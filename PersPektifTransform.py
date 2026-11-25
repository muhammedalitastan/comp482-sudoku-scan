"""Stage 3 - Rectification to a 450x450 square (50 px per cell).

A3: 3 points (TL, TR, BL) via getAffineTransform.
A4: 4 points via estimateAffine2D (least squares over all corners).
H4: 4 points via getPerspectiveTransform.
"""

import cv2
import numpy as np

TARGET_SIZE = 450
CELL_SIZE = TARGET_SIZE // 9  # 50 px per Sudoku cell


def ideal_grid_lines() -> list[int]:
    """The 10 grid lines, one every 50 px: 0, 50, ..., 450.

    450 is the outer edge of the 450 px square. The image itself only contains
    pixels 0..449, so a border that lands on that edge is measured at pixel 449.
    """
    return [i * CELL_SIZE for i in range(TARGET_SIZE // CELL_SIZE)] + [TARGET_SIZE]


def _destination_points() -> np.ndarray:
    """Destination corners in [TL, TR, BR, BL] order.

    The square runs from (0, 0) to (450, 450). That is 50 px per cell, and it
    is the point the brief uses for the A3 fourth-corner error. Pixel 449 is
    the last sample inside that square.
    """
    return np.array(
        [
            [0, 0],
            [TARGET_SIZE, 0],
            [TARGET_SIZE, TARGET_SIZE],
            [0, TARGET_SIZE],
        ],
        dtype=np.float32,
    )


def opposite_side_ratio(corners) -> float:
    """Obliqueness of the detected quad: larger opposite-side ratio.

    A square seen under an affine (weak-perspective) view keeps opposite sides
    equal, so the ratio is 1. Perspective makes one pair unequal.
    """
    pts = _as_corners(corners)
    edges = np.linalg.norm(pts - np.roll(pts, -1, axis=0), axis=1)
    top, right, bottom, left = (float(v) for v in edges)

    def _ratio(a: float, b: float) -> float:
        short = min(a, b)
        if short <= 1e-6:
            return float("inf")
        return max(a, b) / short

    return float(max(_ratio(top, bottom), _ratio(left, right)))


def _as_corners(corners) -> np.ndarray:
    return np.asarray(corners, dtype=np.float32).reshape(4, 2)


def rectify_A3_affine(image: np.ndarray, corners) -> tuple:
    """Affine from TL, TR, BL. Returns (warped, projected_br, br_error_px, M).

    br_error_px is the distance from the projected BR to (450, 450).
    """
    pts = _as_corners(corners)
    src = np.array([pts[0], pts[1], pts[3]], dtype=np.float32)  # TL, TR, BL
    dst = _destination_points()[[0, 1, 3]]

    M = cv2.getAffineTransform(src, dst)
    warped = cv2.warpAffine(image, M, (TARGET_SIZE, TARGET_SIZE))

    projected_br = (M @ np.array([pts[2, 0], pts[2, 1], 1.0], dtype=np.float64))[:2]
    target_br = _destination_points()[2].astype(np.float64)
    br_error = float(np.linalg.norm(projected_br - target_br))

    return warped, projected_br, br_error, M


def rectify_A4_affine(image: np.ndarray, corners) -> tuple:
    """Affine from all 4 corners, least squares. Returns (warped, rms_px, M)."""
    pts = _as_corners(corners)
    dst = _destination_points()

    # With exactly 4 correspondences and no outliers expected, a large
    # reprojection threshold makes every corner an inlier, so the refined
    # estimateAffine2D fit is the least-squares solution over all 4 points.
    # The default RANSAC threshold would reject the 4th corner as an outlier
    # under perspective distortion and return the 3-point A3 transform.
    M, _ = cv2.estimateAffine2D(pts, dst, ransacReprojThreshold=1e6)
    if M is None:
        raise RuntimeError("cv2.estimateAffine2D failed to estimate a transform")

    warped = cv2.warpAffine(image, M, (TARGET_SIZE, TARGET_SIZE))

    src_homogeneous = np.column_stack([pts, np.ones(4)])
    projected = (M @ src_homogeneous.T).T
    # RMS of the four corner distances, not the RMS of the eight x/y scalars.
    distances = np.linalg.norm(projected - dst, axis=1)
    rms_error = float(np.sqrt(np.mean(distances ** 2)))

    return warped, rms_error, M


def rectify_H4_homography(image: np.ndarray, corners) -> tuple:
    """Homography from all 4 corners. Returns (warped, H, max_reproj_px).

    max_reproj_px is the largest corner error after the homography. With four
    exact correspondences it is numerical noise, which confirms that H4 lands
    on (0, 0), (450, 0), (450, 450) and (0, 450).
    """
    pts = _as_corners(corners)
    dst = _destination_points().astype(np.float64)

    H = cv2.getPerspectiveTransform(pts, dst.astype(np.float32))
    warped = cv2.warpPerspective(image, H, (TARGET_SIZE, TARGET_SIZE))

    ones = np.ones((4, 1), dtype=np.float64)
    homogeneous = np.hstack([pts.astype(np.float64), ones])
    projected = (H.astype(np.float64) @ homogeneous.T).T
    projected = projected[:, :2] / projected[:, 2:3]
    max_error = float(np.max(np.linalg.norm(projected - dst, axis=1)))

    return warped, H, max_error
