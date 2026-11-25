"""Stage 1 - Preprocessing.

Returns the representations the later stages need: gray (rectification,
cornerSubPix and OCR), blurred (Harris) and threshold (grid region).
Kernel sizes are fractions of the image width, with an upper cap so a
4000 px photograph stays inside the time limit.
"""

import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class PreprocessedImage:
    original: np.ndarray  # BGR image, as decoded
    gray: np.ndarray      # grayscale, used for warping and cornerSubPix
    blurred: np.ndarray   # denoised grayscale, used for Harris
    threshold: np.ndarray  # adaptive binary (inverted), used as the grid region


def _odd(value: int) -> int:
    """Nearest odd integer >= value, at least 3."""
    value = max(3, value)
    return value if value % 2 == 1 else value + 1


def _adaptive_block_size(height: int, width: int) -> int:
    """Odd Gaussian window, ~2% of width, never larger than the image or 81."""
    if min(height, width) < 3:
        raise ValueError(f"image is too small to threshold: {width}x{height}")
    block = _odd(int(width * 0.02))
    image_cap = min(height, width)
    if image_cap % 2 == 0:
        image_cap -= 1
    # 81 is 2% of a 4000 px side, the largest input the assignment allows.
    return min(block, 81, max(3, image_cap))


def preprocess_image(image_path: str) -> PreprocessedImage:
    """Load an image and build gray / blurred / threshold."""
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
    h, w = gray.shape

    # Gaussian blur, kernel ~1% of image width (clamped to [3, 15], odd)
    blur_k = min(15, _odd(int(w * 0.01)))
    blurred = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)

    # Adaptive threshold, block size ~2% of image width (odd, >= 3, capped)
    block_size = _adaptive_block_size(h, w)
    threshold = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        2,
    )

    return PreprocessedImage(img, gray, blurred, threshold)
