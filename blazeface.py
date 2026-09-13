"""
Real-time face detection using the BlazeFace short-range TFLite model,
run DIRECTLY through the TFLite interpreter -- no MediaPipe, no
tflite_support task library.

The model only outputs raw SSD-style box regressors + score logits per
anchor; something has to turn those into boxes/keypoints. Normally that's
MediaPipe's C++ post-processing (SsdAnchorsCalculator +
TfLiteTensorsToDetectionsCalculator). This script re-implements that exact
math in Python/NumPy, so the only runtime ML dependency is a TFLite
interpreter (tflite-runtime or tensorflow.lite).

Model:
    wget -q -O detector.tflite \
        https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite

Install (pick ONE interpreter backend):
    pip install tflite-runtime opencv-python-headless numpy      # lightweight
    # -- or, if tflite-runtime has no prebuilt wheel for your platform --
    pip install tensorflow opencv-python-headless numpy          # heavier, always available

Usage:
    python blazeface.py --model detector.tflite --outputPath output.mp4
"""

import argparse
from typing import List, Tuple

import cv2
import numpy as np

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite import Interpreter  # fallback to full TensorFlow


# ---------------------------------------------------------------------------
# Anchor configuration for the BlazeFace SHORT-RANGE model. These values are
# not arbitrary -- they mirror mediapipe's own config file:
#   mediapipe/modules/face_detection/face_detection_short_range_common.pbtxt
# Using the wrong numbers here silently produces garbage boxes, since the
# model's raw output only makes sense relative to this exact anchor grid.
# ---------------------------------------------------------------------------
SSD_OPTIONS = {
    "num_layers": 4,
    "input_size": 128,          # model expects a 128x128 input
    "anchor_offset_x": 0.5,
    "anchor_offset_y": 0.5,
    "strides": [8, 16, 16, 16],
    "interpolated_scale_aspect_ratio": 1.0,
}
MIN_SCORE = 0.5              # confidence threshold
MIN_SUPPRESSION_IOU = 0.3    # NMS IoU threshold
RAW_SCORE_LIMIT = 80         # clip logits before sigmoid to avoid float overflow

# Visualization parameters
MARGIN = 10
ROW_SIZE = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red (BGR)


def _generate_anchors(opts: dict) -> np.ndarray:
    """Re-creates mediapipe's SsdAnchorsCalculator for this model's config.

    Produces 896 (x_center, y_center) anchor points for the short-range
    model: 512 from the stride-8 feature map (16x16 cells x 2 anchors) plus
    384 from the stride-16 feature map (8x8 cells x 6 anchors).
    """
    strides = opts["strides"]
    num_layers = opts["num_layers"]
    input_size = opts["input_size"]
    anchors = []
    layer_id = 0
    while layer_id < num_layers:
        last_same_stride_layer = layer_id
        repeats = 0
        while (
            last_same_stride_layer < num_layers
            and strides[last_same_stride_layer] == strides[layer_id]
        ):
            last_same_stride_layer += 1
            repeats += 2 if opts["interpolated_scale_aspect_ratio"] == 1.0 else 1
        stride = strides[layer_id]
        feature_map_size = input_size // stride
        for y in range(feature_map_size):
            y_center = (y + opts["anchor_offset_y"]) / feature_map_size
            for x in range(feature_map_size):
                x_center = (x + opts["anchor_offset_x"]) / feature_map_size
                for _ in range(repeats):
                    anchors.append((x_center, y_center))
        layer_id = last_same_stride_layer
    return np.array(anchors, dtype=np.float32)


def _letterbox(
    frame_bgr: np.ndarray, size: int
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """Resizes a frame to fit in a size x size square, centered, zero-padded.

    Returns the square canvas as an RGB float32 tensor normalized to
    [-1, 1] (what the model expects), plus the (left, top, right, bottom)
    padding fractions of the canvas, so detections can be mapped back to
    the original frame afterwards.
    """
    h, w = frame_bgr.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    x_off = (size - new_w) // 2
    y_off = (size - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized

    left = x_off / size
    right = (size - new_w - x_off) / size
    top = y_off / size
    bottom = (size - new_h - y_off) / size

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32)
    normalized = rgb / 127.5 - 1.0
    return normalized, (left, top, right, bottom)


def _remove_letterbox(
    xy: np.ndarray, padding: Tuple[float, float, float, float]
) -> np.ndarray:
    """Maps normalized (x, y) coords from the padded square back to the
    original (unpadded) frame's normalized [0, 1] coordinate space."""
    left, top, right, bottom = padding
    out = xy.copy()
    out[..., 0] = (xy[..., 0] - left) / (1 - left - right)
    out[..., 1] = (xy[..., 1] - top) / (1 - top - bottom)
    return out


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class BlazeFaceDetector:
    """Runs the BlazeFace short-range TFLite model directly and manually
    decodes its raw SSD-anchor output -- no MediaPipe, no tflite_support
    task library involved."""

    def __init__(self, model_path: str):
        self.interpreter = Interpreter(model_path=model_path, num_threads=2)
        self.interpreter.allocate_tensors()

        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        self.input_index = input_details[0]["index"]
        self.input_size = input_details[0]["shape"][1]  # 128 for this model

        # Identify which output is the box regressors (last dim 16) vs.
        # the score logits (last dim 1) rather than assuming an order.
        self.boxes_index = None
        self.scores_index = None
        for detail in output_details:
            last_dim = detail["shape"][-1]
            if last_dim == 16:
                self.boxes_index = detail["index"]
            elif last_dim == 1:
                self.scores_index = detail["index"]
        if self.boxes_index is None or self.scores_index is None:
            raise RuntimeError(
                "Unexpected model output shapes: "
                f"{[d['shape'] for d in output_details]}. Expected one "
                "output ending in 16 (box regressors) and one ending in 1 "
                "(score logits) -- is this really the short-range BlazeFace "
                "model?"
            )

        self.anchors = _generate_anchors(SSD_OPTIONS)

    def detect(self, frame_bgr: np.ndarray) -> List[dict]:
        """Runs detection on a single BGR frame (as read by OpenCV).

        Returns a list of dicts, each with pixel-space 'bbox'
        (x1, y1, x2, y2), 'keypoints' (list of 6 (x, y) pixel tuples), and
        'score'.
        """
        height, width = frame_bgr.shape[:2]
        tensor, padding = _letterbox(frame_bgr, self.input_size)

        self.interpreter.set_tensor(self.input_index, tensor[np.newaxis])
        self.interpreter.invoke()

        raw_boxes = self.interpreter.get_tensor(self.boxes_index)[0]    # (896, 16)
        raw_scores = self.interpreter.get_tensor(self.scores_index)[0]  # (896, 1)

        boxes = self._decode_boxes(raw_boxes)  # (896, 8, 2) normalized, padded-square space
        scores = _sigmoid(
            np.clip(raw_scores, -RAW_SCORE_LIMIT, RAW_SCORE_LIMIT)
        ).reshape(-1)

        keep = scores > MIN_SCORE
        boxes, scores = boxes[keep], scores[keep]
        if len(boxes) == 0:
            return []

        # boxes[:, 0] = xmin,ymin  boxes[:, 1] = xmax,ymax  boxes[:, 2:8] = keypoints
        xyxy = np.concatenate([boxes[:, 0], boxes[:, 1]], axis=1)  # (N, 4)

        nms_boxes = [
            [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
            for x1, y1, x2, y2 in xyxy
        ]
        indices = cv2.dnn.NMSBoxes(
            nms_boxes, scores.tolist(), MIN_SCORE, MIN_SUPPRESSION_IOU
        )
        if len(indices) == 0:
            return []
        indices = np.array(indices).reshape(-1)

        results = []
        for i in indices:
            box = boxes[i]  # (8, 2), normalized, padded-square space
            box_unpadded = _remove_letterbox(box, padding)

            x1, y1 = box_unpadded[0]
            x2, y2 = box_unpadded[1]
            keypoints_norm = box_unpadded[2:8]

            results.append({
                "bbox": (
                    int(np.clip(x1, 0, 1) * width),
                    int(np.clip(y1, 0, 1) * height),
                    int(np.clip(x2, 0, 1) * width),
                    int(np.clip(y2, 0, 1) * height),
                ),
                "keypoints": [
                    (int(np.clip(kx, 0, 1) * width), int(np.clip(ky, 0, 1) * height))
                    for kx, ky in keypoints_norm
                ],
                "score": float(scores[i]),
            })
        return results

    def _decode_boxes(self, raw_boxes: np.ndarray) -> np.ndarray:
        """Ports mediapipe's TfLiteTensorsToDetectionsCalculator box decode
        (the part that turns per-anchor offsets into actual coordinates)."""
        scale = self.input_size
        num_points = raw_boxes.shape[-1] // 2  # 8: center, size, 6 keypoints
        boxes = raw_boxes.reshape(-1, num_points, 2) / scale

        boxes[:, 0] += self.anchors  # center x,y is anchor-relative
        for i in range(2, num_points):
            boxes[:, i] += self.anchors  # keypoints are anchor-relative too

        center = boxes[:, 0].copy()
        half_size = boxes[:, 1] / 2  # width,height (NOT anchor-relative)
        boxes[:, 0] = center - half_size
        boxes[:, 1] = center + half_size
        return boxes


def visualize(image: np.ndarray, detections: List[dict]) -> np.ndarray:
    """Draws bounding boxes, keypoints, and scores onto a copy of image."""
    annotated = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), TEXT_COLOR, 3)

        for kx, ky in det["keypoints"]:
            cv2.circle(annotated, (kx, ky), 2, (0, 255, 0), 2)

        result_text = f"face ({round(det['score'], 2)})"
        text_location = (MARGIN + x1, MARGIN + ROW_SIZE + y1)
        cv2.putText(
            annotated, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
            FONT_SIZE, TEXT_COLOR, FONT_THICKNESS,
        )
    return annotated


def run(model: str, camera_id: int, width: int, height: int, output_path: str) -> None:
    """Runs face detection on webcam frames and writes the annotated video to disk."""

    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera with id {camera_id}")

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or width
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or height
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (actual_width, actual_height))

    detector = BlazeFaceDetector(model)

    print(f"Recording to '{output_path}'. Press Ctrl+C to stop early.")
    frame_count = 0
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Camera frame not available. Exiting.")
                break

            detections = detector.detect(frame)
            annotated = visualize(frame, detections)
            writer.write(annotated)
            frame_count += 1
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        cap.release()
        writer.release()
        print(f"Done. Wrote {frame_count} frames to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Detect faces from a webcam using a raw TFLite interpreter "
        "(no MediaPipe) and save the annotated video to a file."
    )
    parser.add_argument(
        "--model", help="Path to the BlazeFace short-range .tflite model.",
        default="detector.tflite",
    )
    parser.add_argument(
        "--cameraId", help="Id of the camera to capture from.", type=int, default=0
    )
    parser.add_argument(
        "--frameWidth", help="Requested capture width.", type=int, default=1280
    )
    parser.add_argument(
        "--frameHeight", help="Requested capture height.", type=int, default=720
    )
    parser.add_argument(
        "--outputPath", help="Path to save the annotated output video.",
        default="output.mp4",
    )
    args = parser.parse_args()

    run(args.model, args.cameraId, args.frameWidth, args.frameHeight, args.outputPath)


if __name__ == "__main__":
    main()
