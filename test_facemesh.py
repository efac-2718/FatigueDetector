"""
Standalone test harness for FaceMeshEstimator's geometry methods --
_make_square_roi() and _warp_crop() -- with NO model file and NO camera.

WHY THIS EXISTS
----------------
Those two methods are pure geometry: a bbox + a rotation angle go in, a
square crop comes out. Running them through the full detect() -> estimate()
pipeline mixes their correctness together with the model, the camera, and
(once written) _compute_rotation -- so a bug anywhere shows up as "the crop
looks wrong" with no clue which function caused it. Here we build a fake
frame with a KNOWN, hand-drawn tilt and call the two methods directly, so
you already know what the "correct" output looks like before you look at it.

WHAT IT BUILDS
---------------
A synthetic frame with a rectangle standing in for a face, tilted by
FACE_TILT_DEG, plus a red dot marking one corner so you can tell orientation
(upright vs. flipped vs. mirrored) in the output. The rectangle's
axis-aligned bounding box is computed and used as the fake "bbox" -- this is
exactly what BlazeFaceDetector.detect() would hand you for a tilted face,
since its bbox is always axis-aligned even when the face inside isn't.

OUTPUT
-------
- test_roi_overlay.png: the synthetic frame with the ROI your
  _make_square_roi() computed drawn on top of it, for a rough visual check
  that it's centered on the face and padded, not just fit to the tight bbox.
- test_crop.png: the actual 192x192 output of _warp_crop(). THIS is the real
  check -- if the rotation math is right, the tilted rectangle should look
  UPRIGHT here, with the red dot in a consistent spot. If it's sideways or
  upside down, flip the rotation sign in _warp_crop (or in ROTATION_RAD
  below, to isolate which side the bug is on).
- roi dict and affine matrix, printed to the terminal.

Until _make_square_roi/_warp_crop are implemented, this will raise
NotImplementedError -- that's expected. Fill in one, rerun, check the
printed values; fill in the other, rerun, check the images.
"""

import math

import cv2
import numpy as np

from facemesh import FaceMeshEstimator, SQUARE_SCALE_FACTOR, CROP_SIZE

# Flip to True once _compute_rotation is implemented, to test it in the same
# pass instead of using the hardcoded angle below.
TEST_REAL_ROTATION = True

CANVAS_SIZE = (480, 640)     # (height, width) of the fake frame
FACE_CENTER = (320, 240)     # (x, y) pixel center of the fake face
FACE_WIDTH, FACE_HEIGHT = 120, 160
FACE_TILT_DEG = 20           # the "true" tilt built into the synthetic face
ROTATION_RAD = -math.radians(FACE_TILT_DEG)  # sign is a guess -- flip if the crop comes out backwards


def _rotated_rect_corners(center, width, height, angle_deg):
    """4 corners of a width x height rectangle centered at `center`,
    rotated by angle_deg using the standard 2D rotation matrix."""
    cx, cy = center
    angle = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    half_w, half_h = width / 2, height / 2
    local = [(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)]
    return [(cx + lx * cos_a - ly * sin_a, cy + lx * sin_a + ly * cos_a) for lx, ly in local]


def build_synthetic_frame():
    frame = np.full((*CANVAS_SIZE, 3), 200, dtype=np.uint8)  # light gray background

    corners = _rotated_rect_corners(FACE_CENTER, FACE_WIDTH, FACE_HEIGHT, FACE_TILT_DEG)
    pts = np.array(corners, dtype=np.int32)
    cv2.fillPoly(frame, [pts], color=(180, 140, 100))  # skin-tone-ish fill

    # Orientation marker at the "top of head" corner -- lets you tell
    # upright from upside-down/mirrored in the output crop.
    top_left = corners[0]
    cv2.circle(frame, (int(top_left[0]), int(top_left[1])), 8, (0, 0, 255), -1)

    xs, ys = [c[0] for c in corners], [c[1] for c in corners]
    bbox = (min(xs), min(ys), max(xs), max(ys))  # axis-aligned, like BlazeFace's real output
    return frame, bbox


def main():
    frame, bbox = build_synthetic_frame()

    # Fake keypoints (not used by _make_square_roi/_warp_crop directly --
    # only included so `detection` has the same shape a real one would).
    cx, cy = FACE_CENTER
    a = math.radians(FACE_TILT_DEG)
    offset = 25
    right_eye = (cx - offset * math.cos(a), cy - offset * math.sin(a))
    left_eye = (cx + offset * math.cos(a), cy + offset * math.sin(a))
    detection = {
        "bbox": bbox,
        "keypoints": [right_eye, left_eye, (cx, cy), (cx, cy + 40), (cx - 40, cy), (cx + 40, cy)],
        "score": 0.99,
    }

    # Skip __init__ entirely -- no .tflite model needed to test pure geometry.
    mesh = FaceMeshEstimator.__new__(FaceMeshEstimator)

    if TEST_REAL_ROTATION:
        rotation = mesh._compute_rotation(detection["keypoints"])
    else:
        rotation = ROTATION_RAD
    print(f"rotation (rad): {rotation:.4f}")

    roi = mesh._make_square_roi(detection["bbox"], rotation)
    print(f"roi: {roi}")

    crop, affine_matrix = mesh._warp_crop(frame, roi)
    print(f"affine_matrix:\n{affine_matrix}")
    print(f"crop shape: {crop.shape} (expected ({CROP_SIZE}, {CROP_SIZE}, 3))")

    overlay = frame.copy()
    roi_corners = _rotated_rect_corners(
        (roi["cx"], roi["cy"]), roi["size"], roi["size"], math.degrees(roi["rotation"])
    )
    cv2.polylines(overlay, [np.array(roi_corners, dtype=np.int32)], True, (0, 255, 0), 2)
    cv2.imwrite("test_roi_overlay.png", overlay)
    cv2.imwrite("test_crop.png", crop)

    print("\nWrote test_roi_overlay.png and test_crop.png. Check:")
    print(f"  1. overlay: green square centered on the face, ~{SQUARE_SCALE_FACTOR}x its bbox.")
    print("  2. crop: face should look UPRIGHT with the red dot near the top --")
    print("     if it's sideways/upside down, flip the rotation sign.")


if __name__ == "__main__":
    main()
