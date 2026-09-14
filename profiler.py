"""
Stage-by-stage timing profiler for the NexGuard pipeline.

Usage in pipeline.py:

    from profiler import StageProfiler
    profiler = StageProfiler()

    for _ in range(100):
        profiler.start_frame()

        with profiler.stage("capture"):
            frame = camera.read()

        with profiler.stage("blazeface"):
            boxes = blazeface.detect(frame)

        with profiler.stage("facemesh_roi"):
            roi, affine = facemesh.make_roi(frame, boxes[0])

        with profiler.stage("facemesh_infer"):
            landmarks = facemesh.infer(roi)

        with profiler.stage("eye_metrics"):
            ear = eye_metrics.update(landmarks)

        profiler.end_frame()

    profiler.report()
"""

import time
from collections import defaultdict
from contextlib import contextmanager


class StageProfiler:
    def __init__(self):
        self.totals = defaultdict(float)
        self.counts = defaultdict(int)
        self._frame_start = None

    @contextmanager
    def stage(self, name):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.totals[name] += time.perf_counter() - t0
            self.counts[name] += 1

    def start_frame(self):
        self._frame_start = time.perf_counter()

    def end_frame(self):
        if self._frame_start is None:
            return
        dt = time.perf_counter() - self._frame_start
        self.totals["_frame_total"] += dt
        self.counts["_frame_total"] += 1
        self._frame_start = None

    def report(self):
        n = self.counts.get("_frame_total", 0)
        total_time = self.totals.get("_frame_total", 0.0)
        if n == 0:
            print("No frames recorded.")
            return

        print(f"\n--- Profile over {n} frames ---")
        header = f"{'stage':<20}{'mean ms':>10}{'total s':>10}{'% frame':>10}"
        print(header)
        print("-" * len(header))

        stage_names = [k for k in self.totals if k != "_frame_total"]
        for name in sorted(stage_names, key=lambda k: -self.totals[k]):
            mean_ms = (self.totals[name] / self.counts[name]) * 1000
            pct = (self.totals[name] / total_time * 100) if total_time else 0.0
            print(f"{name:<20}{mean_ms:>10.2f}{self.totals[name]:>10.3f}{pct:>9.1f}%")

        print("-" * len(header))
        print(f"{'TOTAL / frame':<20}{(total_time/n)*1000:>10.2f}{total_time:>10.3f}{'100.0%':>10}")
        print(f"Effective FPS: {n/total_time:.2f}")


if __name__ == "__main__":
    # Self-test with fake work so you can confirm the harness itself
    # before wiring it into pipeline.py
    import random

    p = StageProfiler()
    for _ in range(30):
        p.start_frame()
        with p.stage("capture"):
            time.sleep(random.uniform(0.01, 0.02))
        with p.stage("blazeface"):
            time.sleep(random.uniform(0.03, 0.05))
        with p.stage("facemesh"):
            time.sleep(random.uniform(0.04, 0.06))
        p.end_frame()
    p.report()
