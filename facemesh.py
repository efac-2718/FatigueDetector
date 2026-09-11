"""
FaceMesh landmark regression, run DIRECTLY through the TFLite interpreter.
"""

from typing import Optional, Tuple
from math import atan2, pi, degrees

import cv2
import numpy as np

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite import Interpreter


# ---------------------------------------------------------------------------
# FaceMesh crop/warp configuration.
# ---------------------------------------------------------------------------
CROP_SIZE = 192              # face_landmark.tflite expects a 192x192 input
NUM_LANDMARKS = 468
SQUARE_SCALE_FACTOR = 1.5    # symmetric expansion applied to the BlazeFace box
MIN_FACE_SCORE = 0.5        # threshold for landmark confidence

RIGHT_EYE_KP_INDEX = 0
LEFT_EYE_KP_INDEX = 1
TARGET_ANGLE_RAD = 0.0


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _normalize_radians(angle: float) -> float:
    return (angle + pi) % (2 * pi) - pi


class FaceMeshEstimator:
    """Runs face_landmark.tflite directly with manual rotation-correction crop/warp."""

    def __init__(self, model_path: str, min_face_score: float = MIN_FACE_SCORE):
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        self.input_index = input_details[0]["index"]
        self.input_size = input_details[0]["shape"][1]
        self.min_face_score = min_face_score

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
                f"Unexpected model output shapes: {[d['shape'] for d in output_details]}"
            )

    def estimate(self, frame_bgr: np.ndarray, detection: dict) -> Optional[np.ndarray]:
        """Runs FaceMesh on a detection dict. Returns (NUM_LANDMARKS, 2) array or None."""
        rotation = self._compute_rotation(detection["keypoints"])
        roi = self._make_square_roi(detection["bbox"], rotation)
        crop, affine_matrix = self._warp_crop(frame_bgr, roi)

        # face_landmark.tflite expects RGB, [0, 1] -- confirmed against real
        # snapshots (test_norm01 landmarks land on the face correctly;
        # test_norm11's [-1, 1] variant did not).
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
        tensor = crop_rgb / 255.0

        self.interpreter.set_tensor(self.input_index, tensor[np.newaxis])
        self.interpreter.invoke()

        raw_landmarks = self.interpreter.get_tensor(self.landmarks_index)
        raw_flag = self.interpreter.get_tensor(self.face_flag_index)

        face_score = float(_sigmoid(raw_flag).reshape(-1)[0])

        # Filter out false positives / empty crops
        if face_score < self.min_face_score:
            return None

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
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        long_side = max(x2 - x1, y2 - y1)
        size = long_side * SQUARE_SCALE_FACTOR
        return {"cx": cx, "cy": cy, "size": size, "rotation": rotation}

    def _warp_crop(self, frame_bgr: np.ndarray, roi: dict) -> Tuple[np.ndarray, np.ndarray]:
        scale = CROP_SIZE / roi["size"]
        angle_deg = degrees(roi["rotation"])

        matrix = cv2.getRotationMatrix2D((roi["cx"], roi["cy"]), angle_deg, scale)
        matrix[0, 2] += CROP_SIZE / 2 - roi["cx"]
        matrix[1, 2] += CROP_SIZE / 2 - roi["cy"]

        crop = cv2.warpAffine(
            frame_bgr, matrix, (CROP_SIZE, CROP_SIZE), borderMode=cv2.BORDER_REPLICATE
        )
        return crop, matrix

    def _reshape_landmarks(self, raw_landmarks: np.ndarray) -> np.ndarray:
        landmarks = raw_landmarks.reshape(-1, 3)
        return landmarks / CROP_SIZE

    def _inverse_transform(self, landmarks: np.ndarray, affine_matrix: np.ndarray) -> np.ndarray:
        inv_matrix = cv2.invertAffineTransform(affine_matrix)
        xy_crop_px = landmarks[:, :2] * CROP_SIZE
        ones = np.ones((xy_crop_px.shape[0], 1), dtype=xy_crop_px.dtype)
        homogeneous = np.hstack([xy_crop_px, ones])

        return homogeneous @ inv_matrix.T
