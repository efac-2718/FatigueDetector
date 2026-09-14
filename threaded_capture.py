"""
Decouples camera capture from inference so a slow CNN stage never
blocks or backs up the capture pipe.

A background thread continuously reads from RpicamCapture and keeps
only the single latest frame in a shared slot (protected by a lock).
The inference loop asks for "whatever is freshest right now" instead
of pulling from a queue -- so if BlazeFace+FaceMesh take 110ms and a
new frame arrives every 40ms, the loop simply skips frames it can't
keep up with rather than falling behind on stale ones.

This does NOT reduce per-frame inference latency -- it stops the
capture side from becoming an accidental bottleneck for the compute
side. To reduce inference latency itself, see profiler.py.

Usage in pipeline.py:

    from threaded_capture import LatestFrameReader
    reader = LatestFrameReader(RpicamCapture(...)).start()

    try:
        while True:
            frame = reader.get_latest()
            if frame is None:
                continue
            boxes = blazeface.detect(frame)
            ...
    finally:
        reader.stop()
"""

import threading
import time


class LatestFrameReader:
    def __init__(self, capture):
        """capture must expose a blocking .read() -> (ok, frame), matching
        RpicamCapture/StandardCapture's convention (mirrors cv2.VideoCapture)."""
        self._capture = capture
        self._lock = threading.Lock()
        self._frame = None
        self._frame_id = 0
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.dropped_frames = 0
        self._last_delivered_id = -1

    def start(self):
        self._thread.start()
        return self

    def _run(self):
        while not self._stop:
            ok, frame = self._capture.read()  # RpicamCapture/StandardCapture both return (ok, frame)
            if not ok or frame is None:
                time.sleep(0.01)  # avoid busy-spinning if the source stalls or ends
                continue
            with self._lock:
                if self._frame_id != self._last_delivered_id:
                    self.dropped_frames += 1
                self._frame = frame
                self._frame_id += 1

    def get_latest(self, timeout=None):
        """
        Blocks (up to `timeout` seconds, or forever if None) until a frame
        newer than the last one delivered is available, then returns it.
        """
        start = time.perf_counter()
        while True:
            with self._lock:
                if self._frame_id != self._last_delivered_id:
                    self._last_delivered_id = self._frame_id
                    return self._frame
            if timeout is not None and (time.perf_counter() - start) > timeout:
                return None
            time.sleep(0.001)

    def stop(self):
        self._stop = True
        self._thread.join(timeout=1.0)


if __name__ == "__main__":
    # Fake capture to sanity-check the reader on its own, before wiring
    # in the real RpicamCapture.
    class FakeCapture:
        def __init__(self):
            self._n = 0

        def read(self):
            time.sleep(0.04)  # simulate a ~25fps camera
            self._n += 1
            return True, self._n

    reader = LatestFrameReader(FakeCapture()).start()
    try:
        for _ in range(10):
            time.sleep(0.11)  # simulate slow inference (~9fps)
            frame = reader.get_latest()
            print(f"processed frame {frame}, dropped so far: {reader.dropped_frames}")
    finally:
        reader.stop()
