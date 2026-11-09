"""Stage 3 - Rectification to a 450x450 square (50 px per cell).

A3: 3 points (TL, TR, BL) via getAffineTransform.
A4: 4 points via estimateAffine2D (least squares over all corners).
H4: 4 points via getPerspectiveTransform.
"""

import cv2
import numpy as np

TARGET_SIZE = 450
CELL_SIZE = TARGET_SIZE // 9  # 50 px per Sudoku cell


def _destination_points() -> np.ndarray:
    """Destination corners in [TL, TR, BR, BL] order."""
    return np.array(
        [
            [0, 0],
            [TARGET_SIZE - 1, 0],
            [TARGET_SIZE - 1, TARGET_SIZE - 1],
            [0, TARGET_SIZE - 1],
        ],
        dtype=np.float32,
    )


def _as_corners(corners) -> np.ndarray:
    return np.asarray(corners, dtype=np.float32).reshape(4, 2)


def rectify_A3_affine(image: np.ndarray, corners) -> tuple:
    """Affine from TL, TR, BL. Returns (warped, projected_br, br_error_px, M)."""
    pts = _as_corners(corners)
    src = np.array([pts[0], pts[1], pts[3]], dtype=np.float32)  # TL, TR, BL
    dst = _destination_points()[[0, 1, 3]]

    M = cv2.getAffineTransform(src, dst)
    warped = cv2.warpAffine(image, M, (TARGET_SIZE, TARGET_SIZE))

    projected_br = (M @ np.array([pts[2, 0], pts[2, 1], 1.0]))[:2]
    br_error = float(np.linalg.norm(projected_br - _destination_points()[2][:2]))

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
    M, _inliers = cv2.estimateAffine2D(pts, dst, ransacReprojThreshold=1e6)
    if M is None:
        raise RuntimeError("cv2.estimateAffine2D failed to estimate a transform")

    warped = cv2.warpAffine(image, M, (TARGET_SIZE, TARGET_SIZE))

    src_homogeneous = np.column_stack([pts, np.ones(4)])
    projected = (M @ src_homogeneous.T).T
    rms_error = float(np.sqrt(np.mean((projected - dst) ** 2)))

    return warped, rms_error, M


def rectify_H4_homography(image: np.ndarray, corners) -> tuple:
    """Homography from all 4 corners. Returns (warped, H)."""
    pts = _as_corners(corners)
    dst = _destination_points()

    H = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image, H, (TARGET_SIZE, TARGET_SIZE))

    return warped, H
