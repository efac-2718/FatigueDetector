"""
FaceMesh landmark regression, run DIRECTLY through the TFLite interpreter --
no MediaPipe graph involved. Consumes a single detection dict as produced by
BlazeFaceDetector.detect() (see blazeface.py) and returns 468 landmarks
mapped back into the ORIGINAL frame's pixel coordinates.

In mediapipe's own graph, this stage is not one calculator but several:
    - a rotation calculator (computes face angle from two eye keypoints)
    - a rect/ROI calculator (turns the BlazeFace box into a square, scaled
      and rotated crop region)
    - ImageToTensorCalculator (warps that region into a fixed-size tensor)
    - FaceLandmarkCalculator (runs face_landmark.tflite itself)
    - a calculator that maps the tensor-space landmarks back through the
      inverse of the warp to land in the original image

This file re-implements that chain by hand, the same way blazeface.py
re-implements SsdAnchorsCalculator + TfLiteTensorsToDetectionsCalculator.

Model:
    wget -q -O face_landmark.tflite \
        https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmark/float16/1/face_landmark.tflite
    (or the equivalent path for whichever face_landmark.tflite version you're
    already using -- confirm output tensor shapes match the constants below)

Usage (once filled in):
    from blazeface import BlazeFaceDetector
    from facemesh import FaceMeshEstimator

    detector = BlazeFaceDetector("detector.tflite")
    mesh = FaceMeshEstimator("face_landmark.tflite")

    detections = detector.detect(frame_bgr)
    for det in detections:
        landmarks = mesh.estimate(frame_bgr, det)  # (468, 2) or (468, 3) px coords
"""

from typing import Optional, Tuple
from math import atan2, pi, degrees

import cv2
import numpy as np

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite import Interpreter  # fallback to full TensorFlow


# ---------------------------------------------------------------------------
# FaceMesh crop/warp configuration. Mirrors mediapipe's own defaults for the
# face landmark ROI -- confirm against your notes before trusting these.
# ---------------------------------------------------------------------------
CROP_SIZE = 192              # face_landmark.tflite expects a 192x192 input
NUM_LANDMARKS = 468
SQUARE_SCALE_FACTOR = 1.5    # symmetric expansion applied to the BlazeFace
                             # box before warping, per your notes

# Indices into the 6 BlazeFace keypoints (det["keypoints"]), used to compute
# rotation. This ordering mirrors mediapipe's default BlazeFace keypoint
# order -- VERIFY this against what your model/graph actually emits before
# relying on it, since getting left/right swapped here silently produces a
# rotation of the wrong sign.
RIGHT_EYE_KP_INDEX = 0
LEFT_EYE_KP_INDEX = 1

# mediapipe's rotation calculator targets a specific angle (commonly 0, i.e.
# "eyes horizontal") rather than raw atan2 output -- fill in once you've
# checked your notes on the exact target-angle convention used.
TARGET_ANGLE_RAD = 0.0


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def _normalize_radians(angle: float) -> float:
    return (angle + pi) % (2*pi) - pi

class FaceMeshEstimator:
    """Runs face_landmark.tflite directly and manually handles the
    rotation-correction crop/warp that mediapipe normally does via
    surrounding calculator nodes -- no MediaPipe graph involved."""

    def __init__(self, model_path: str):
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        self.input_index = input_details[0]["index"]
        self.input_size = input_details[0]["shape"][1]  # expect CROP_SIZE

        # face_landmark.tflite has two outputs: the landmark tensor (last
        # dim/flattened size ~ NUM_LANDMARKS * 3) and a face-presence flag
        # (last dim 1). Identify by shape rather than assuming order, same
        # approach as BlazeFaceDetector.__init__ in blazeface.py.
        self.landmarks_index = None
        self.face_flag_index = None
        for detail in output_details:
            size = int(np.prod(detail["shape"]))
            if size == 1:
                self.face_flag_index = detail["index"]
            else:
                self.landmarks_index = detail["index"]
        if self.landmarks_index is None or self.face_flag_index is None:
            raise RuntimeError(
                "Unexpected model output shapes: "
                f"{[d['shape'] for d in output_details]}. Expected one "
                "output with the flattened landmark tensor and one output "
                "with a single face-presence score -- is this really "
                "face_landmark.tflite?"
            )

    def estimate(self, frame_bgr: np.ndarray, detection: dict) -> Optional[np.ndarray]:
        """Runs FaceMesh on one BlazeFace detection dict (as returned by
        BlazeFaceDetector.detect()).

        Returns an (NUM_LANDMARKS, 2) array of pixel coords in the ORIGINAL
        frame, or None if the model reports low face-presence confidence.
        """
        rotation = self._compute_rotation(detection["keypoints"])
        roi = self._make_square_roi(detection["bbox"], rotation)

        crop, affine_matrix = self._warp_crop(frame_bgr, roi)

        tensor = crop.astype(np.float32) / 255.0  # TODO: confirm normalization
        # matches what face_landmark.tflite actually expects (some versions
        # want [0,1], others [-1,1] -- check against your notes/model card).

        self.interpreter.set_tensor(self.input_index, tensor[np.newaxis])
        self.interpreter.invoke()

        raw_landmarks = self.interpreter.get_tensor(self.landmarks_index)
        raw_flag = self.interpreter.get_tensor(self.face_flag_index)

        face_score = float(_sigmoid(raw_flag).reshape(-1)[0])
        # TODO: threshold face_score and return None below some MIN_FACE_SCORE,
        # the same way BlazeFaceDetector.detect() filters on MIN_SCORE.

        landmarks_crop_space = self._reshape_landmarks(raw_landmarks)
        landmarks_original = self._inverse_transform(landmarks_crop_space, affine_matrix)
        return landmarks_original

    def _compute_rotation(self, keypoints) -> float:
        right_x_coord, right_y_coord = keypoints[RIGHT_EYE_KP_INDEX]
        left_x_coord, left_y_coord = keypoints[LEFT_EYE_KP_INDEX]

        angle = atan2(-(left_y_coord - right_y_coord), left_x_coord - right_x_coord)
        rotation = TARGET_ANGLE_RAD - angle
        return _normalize_radians(rotation)

    def _make_square_roi(self, bbox, rotation: float) -> dict:
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2)/2
        cy = (y1 + y2)/2

        long_side = max(x2 - x1, y2 - y1)
        size = long_side * SQUARE_SCALE_FACTOR
        return {"cx": cx, "cy": cy, "size": size, "rotation": rotation}


    def _warp_crop(self, frame_bgr: np.ndarray, roi: dict) -> Tuple[np.ndarray, np.ndarray]:
        scale = CROP_SIZE/roi["size"]
        angle_deg = degrees(roi["rotation"])

        matrix = cv2.getRotationMatrix2D((roi["cx"], roi["cy"]), angle_deg, scale)
        matrix[0, 2] += CROP_SIZE/2 - roi["cx"]
        matrix[1, 2] += CROP_SIZE/2 - roi["cy"]

        crop = cv2.warpAffine(frame_bgr, matrix, (CROP_SIZE, CROP_SIZE), borderMode=cv2.BORDER_REPLICATE)
        return crop, matrix

    def _reshape_landmarks(self, raw_landmarks: np.ndarray) -> np.ndarray:
        landmarks = raw_landmarks.reshape(-1, 3)
        return landmarks/CROP_SIZE


    def _inverse_transform(self, landmarks: np.ndarray, affine_matrix: np.ndarray) -> np.ndarray:
        inv_matrix = cv2.invertAffineTransform(affine_matrix)
        xy_crop_px = landmarks[:, :2] * CROP_SIZE
        ones = np.ones((xy_crop_px.shape[0], 1), dtype=xy_crop_px.dtype)
        homogeneous = np.hstack([xy_crop_px, ones])
        
        return homogeneous @ inv_matrix.T
