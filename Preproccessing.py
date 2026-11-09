"""Stage 1 - Preprocessing.

Returns the representations the later stages need: gray (rectification +
OCR), blurred (Harris), threshold (grid region) and edges. Every parameter
is relative to the image width so any photo resolution works.
"""

import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class PreprocessedImage:
    original: np.ndarray  # BGR image
    gray: np.ndarray      # grayscale
    blurred: np.ndarray   # denoised grayscale
    threshold: np.ndarray  # adaptive binary (inverted)
    edges: np.ndarray     # dilated Canny edges


def _odd(value: int) -> int:
    """Nearest odd integer >= value, at least 3."""
    value = max(3, value)
    return value if value % 2 == 1 else value + 1


def preprocess_image(image_path: str) -> PreprocessedImage:
    """Load an image and build gray / blurred / threshold / edges."""
    # cv2.imread fails on non-ASCII paths (e.g. Turkish filenames), so read
    # the bytes ourselves and decode.
    try:
        data = np.fromfile(image_path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except (OSError, ValueError):
        img = None
    if img is None:
        raise FileNotFoundError(f"Cannot read image file: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, w = gray.shape

    # Gaussian blur, kernel ~1% of image width (clamped to [3, 15], odd)
    blur_k = min(15, _odd(int(w * 0.01)))
    blurred = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)

    # Adaptive threshold, block size ~2% of image width (odd, >= 3)
    block_size = _odd(int(w * 0.02))
    threshold = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        2,
    )

    # Canny edges with thresholds derived from the median intensity
    median = float(np.median(blurred))
    lower = max(0, int(0.5 * median))
    upper = min(255, max(lower + 1, int(1.5 * median)))
    edges = cv2.Canny(blurred, lower, upper)

    # Dilate edges so the grid lines become connected (~0.5% of image width)
    dil_k = max(3, int(w * 0.005))
    kernel = np.ones((dil_k, dil_k), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    return PreprocessedImage(img, gray, blurred, threshold, edges)
