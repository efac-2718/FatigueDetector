"""
Execution pipeline for BlazeFace -> FaceMesh.
Supports standard laptop webcams/video files as well as Raspberry Pi rpicam-vid.
"""

import argparse
import subprocess
import time
import cv2
import numpy as np

from blazeface import BlazeFaceDetector
from facemesh import FaceMeshEstimator


class StandardCapture:
    """Captures frames using standard OpenCV VideoCapture (Ubuntu webcams or video files)."""

    def __init__(self, source):
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {source}")

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()


class RpicamCapture:
    """Captures YUV420 raw frames from rpicam-vid pipe on Raspberry Pi."""

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        self.width = width
        self.height = height
        self.frame_bytes = int(width * height * 1.5)

        cmd = [
            "rpicam-vid",
            "-t", "0",
            "--inline",
            "--width", str(width),
            "--height", str(height),
            "--framerate", str(fps),
            "--codec", "yuv420",
            "-o", "-"
        ]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**7)

    def read(self):
        raw = self.proc.stdout.read(self.frame_bytes)
        if len(raw) != self.frame_bytes:
            return False, None
        
        yuv = np.frombuffer(raw, dtype=np.uint8).reshape((int(self.height * 1.5), self.width))
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
        return True, bgr

    def release(self):
        if self.proc:
            self.proc.terminate()
            self.proc.wait()


def main():
    parser = argparse.ArgumentParser(description="Run BlazeFace + FaceMesh pipeline")
    parser.add_argument("--detector", default="detector.tflite", help="Path to detector tflite")
    parser.add_argument("--landmark", default="face_landmark.tflite", help="Path to landmark tflite")
    parser.add_argument("--input", default="0", help="Webcam ID (e.g. 0) or path to video file")
    parser.add_argument("--output", default="output.mp4", help="Output MP4 file path")
    parser.add_argument("--rpicam", action="store_true", help="Use rpicam-vid (Raspberry Pi only)")
    parser.add_argument("--show", action="store_true", help="Display live window with OpenCV")
    args = parser.parse_args()

    detector = BlazeFaceDetector(args.detector)
    mesh = FaceMeshEstimator(args.landmark)

    if args.rpicam:
        cap = RpicamCapture(width=640, height=480)
    else:
        cap = StandardCapture(args.input)

    writer = None
    print(f"Starting pipeline. Saving to {args.output}. Press Ctrl+C or 'q' to stop.")
    frame_count = 0
    start_time = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            h, w = frame.shape[:2]
            if writer is None:
                writer = cv2.VideoWriter(
                    args.output,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    20.0,
                    (w, h)
                )

            detections = detector.detect(frame)
            for det in detections:
                landmarks = mesh.estimate(frame, det)
                if landmarks is None:
                    continue

                # Draw bounding box
                x1, y1, x2, y2 = det["bbox"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                # Draw 468 landmarks
                for x, y in landmarks[:, :2].astype(int):
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

            writer.write(frame)
            frame_count += 1

            if args.show:
                cv2.imshow("NexGuard Pipeline", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if frame_count % 30 == 0:
                fps = frame_count / (time.time() - start_time)
                print(f"Processed {frame_count} frames (~{fps:.2f} FPS)...")

    except KeyboardInterrupt:
        print("Stopping capture...")
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print(f"Done. Wrote {frame_count} frames to {args.output}")


if __name__ == "__main__":
    main()
