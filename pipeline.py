"""
Minimal driver wiring BlazeFace -> FaceMesh together.

This is intentionally thin right now. Once facemesh.py's stubs are filled
in and you're ready to retire the standalone BlazeFace-only loop, move
RpicamCapture / visualize() here from BlazeFace.py too, so this file owns
all the "app" plumbing and blazeface.py / facemesh.py stay pure model code.
"""

import cv2

from blazeface import BlazeFaceDetector   # rename BlazeFace.py -> blazeface.py first
from facemesh import FaceMeshEstimator

DETECTOR_MODEL = "blaze_face_short_range.tflite"
LANDMARK_MODEL = "face_landmark.tflite"


def main():
    detector = BlazeFaceDetector(DETECTOR_MODEL)
    mesh = FaceMeshEstimator(LANDMARK_MODEL)

    # Placeholder capture -- swap for RpicamCapture (currently in
    # BlazeFace.py) once you move it here. cv2.VideoCapture(0) is only
    # useful for testing on a machine with a normal webcam/V4L2 device,
    # NOT on the Pi Zero 2W where the OV5647 needs the rpicam-vid subprocess.
    cap = cv2.VideoCapture(0)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            detections = detector.detect(frame)
            for det in detections:
                # NOTE: this will raise NotImplementedError right now --
                # _compute_rotation etc. are still stubs. That's expected;
                # this file exists to prove the wiring, not to run end to end
                # yet.
                landmarks = mesh.estimate(frame, det)
                if landmarks is None:
                    continue
                for x, y in landmarks[:, :2].astype(int):
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

            cv2.imshow("NexGuard", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
