"""Stage 5 - Tesseract OCR on the rectified grid.

Empty cells are filtered by ink ratio before OCR, and recognition is
restricted to digits 1-9. Non-empty cells are packed into one montage image
per pass (a second sparse-text pass retries the missed ones) to stay far
below the 20 s limit. prepared=True normalizes each cell (crop, white
background, centered); prepared=False feeds the raw cell for comparison.
"""

import os

import cv2
import numpy as np
import pytesseract

# Use the Windows Tesseract install when present; otherwise PATH is used.
if os.path.exists(r"C:\Program Files\Tesseract-OCR\tesseract.exe"):
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

_MIN_INK_RATIO = 0.02
_DIGITS = "123456789"


def is_cell_empty(cell, min_ink_ratio: float = _MIN_INK_RATIO) -> bool:
    """True when the cell has no digit: flat background or almost no ink."""
    if cell is None or cell.size == 0:
        return True
    # A nearly uniform cell (shaded background, shadow) cannot contain a
    # digit; Otsu would split its noise ~50/50 and fake a high ink ratio.
    if float(cell.std()) < 10.0:
        return True
    _, binary = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink_ratio = cv2.countNonZero(binary) / cell.size
    return ink_ratio < min_ink_ratio


def preprocess_cell(cell: np.ndarray) -> np.ndarray:
    """Crop to the digit, force the background white, center on a 28x28 canvas."""
    if cell is None or cell.size == 0:
        return cell

    _, binary = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    coords = cv2.findNonZero(binary)
    if coords is None:
        return cell

    x, y, w, h = cv2.boundingRect(coords)
    digit = cell[y : y + h, x : x + w]

    _, mask = cv2.threshold(digit, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    digit = np.where(mask > 0, digit, 255).astype(np.uint8)

    canvas = np.full((28, 28), 255, np.uint8)

    if max(w, h) > 24:
        scale = 24 / max(w, h)
        digit = cv2.resize(digit, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        h, w = digit.shape

    x_off = (28 - w) // 2
    y_off = (28 - h) // 2
    canvas[y_off : y_off + h, x_off : x_off + w] = digit

    return canvas


# Montage layout: one 64px slot per cell, 16px gaps, 48px digit canvas.
_MONTAGE_CELL = 64
_MONTAGE_GAP = 16
_MONTAGE_INNER = 48
_MONTAGE_SIZE = 9 * _MONTAGE_CELL + 10 * _MONTAGE_GAP

_MONTAGE_CONFIG = r"--oem 3 --psm 6 -c tessedit_char_whitelist=123456789"
_MONTAGE_RETRY_CONFIG = r"--oem 3 --psm 11 -c tessedit_char_whitelist=123456789"
# Raw cells keep their gray photo background; Tesseract's default Otsu
# thresholding drowns in it, so unprepared passes need adaptive thresholding.
_UNPREPARED_CONFIG = (
    r"--oem 3 --psm 6 -c tessedit_char_whitelist=123456789"
    r" -c thresholding_method=1"
)
_UNPREPARED_RETRY_CONFIG = (
    r"--oem 3 --psm 11 -c tessedit_char_whitelist=123456789"
    r" -c thresholding_method=1"
)
_SINGLE_CELL_CONFIG = r"--oem 3 --psm 10 -c tessedit_char_whitelist=123456789"


def _digit_canvas(cell: np.ndarray, prepared: bool):
    """Normalized digit image (_MONTAGE_INNER square) or None."""
    if prepared:
        canvas = preprocess_cell(cell)
        if canvas is None or canvas.size == 0:
            return None
        return cv2.resize(canvas, (_MONTAGE_INNER, _MONTAGE_INNER),
                          interpolation=cv2.INTER_NEAREST)
    return cv2.resize(cell, (_MONTAGE_INNER, _MONTAGE_INNER), interpolation=cv2.INTER_AREA)


def _build_montage(cells, prepared: bool, only: np.ndarray | None = None) -> np.ndarray:
    """Pack non-empty cells onto one white montage; `only` restricts positions."""
    montage = np.full((_MONTAGE_SIZE, _MONTAGE_SIZE), 255, np.uint8)
    offset = (_MONTAGE_CELL - _MONTAGE_INNER) // 2

    for row in range(9):
        for col in range(9):
            if only is not None and not only[row, col]:
                continue
            cell = cells[row][col]
            if is_cell_empty(cell):
                continue
            digit = _digit_canvas(cell, prepared)
            if digit is None:
                continue
            x0 = _MONTAGE_GAP + col * (_MONTAGE_CELL + _MONTAGE_GAP) + offset
            y0 = _MONTAGE_GAP + row * (_MONTAGE_CELL + _MONTAGE_GAP) + offset
            montage[y0 : y0 + _MONTAGE_INNER, x0 : x0 + _MONTAGE_INNER] = digit

    return montage


def _cell_index(center: float, gap: int = _MONTAGE_GAP) -> int:
    """Map a montage coordinate to a cell index 0..8."""
    return int(round((center - gap - _MONTAGE_CELL / 2) / (_MONTAGE_CELL + _MONTAGE_GAP)))


def _read_montage(montage: np.ndarray, config: str = _MONTAGE_CONFIG) -> np.ndarray:
    """One Tesseract call; map character boxes back to cells via their centers."""
    grid = np.zeros((9, 9), dtype=int)
    height = montage.shape[0]

    try:
        # image_to_boxes lines: <char> <left> <bottom> <right> <top> <page>,
        # y measured from the bottom of the image.
        boxes = pytesseract.image_to_boxes(montage, config=config)
    except Exception:
        return grid

    for line in boxes.strip().split("\n"):
        parts = line.split()
        if len(parts) != 6:
            continue
        char = parts[0].strip()
        if len(char) != 1 or char not in _DIGITS:
            continue
        try:
            left, bottom, right, top = (float(v) for v in parts[1:5])
        except ValueError:
            continue

        cx = (left + right) / 2
        cy = height - (bottom + top) / 2
        col = _cell_index(cx)
        row = _cell_index(cy)
        if 0 <= row < 9 and 0 <= col < 9:
            grid[row, col] = int(char)

    return grid


def predict_sudoku_grid(cells, prepared: bool = True) -> np.ndarray:
    """Recognize digits in a 9x9 cell matrix; 0 for empty/unread cells."""
    config = _MONTAGE_CONFIG if prepared else _UNPREPARED_CONFIG
    retry_config = _MONTAGE_RETRY_CONFIG if prepared else _UNPREPARED_RETRY_CONFIG

    grid = _read_montage(_build_montage(cells, prepared), config=config)

    # Second pass: cells the first montage missed, isolated on their own
    # montage in sparse-text mode.
    missed = np.zeros((9, 9), dtype=bool)
    for row in range(9):
        for col in range(9):
            if grid[row, col] == 0 and not is_cell_empty(cells[row][col]):
                missed[row, col] = True
    if missed.any():
        retry = _read_montage(
            _build_montage(cells, prepared, only=missed), config=retry_config
        )
        grid = np.where((grid == 0) & missed, retry, grid)

    return grid


def ocr_single_cell(cell: np.ndarray) -> int:
    """Single-character OCR for one prepared cell; 0 when unreadable."""
    canvas = preprocess_cell(cell)
    if canvas is None or canvas.size == 0:
        return 0
    canvas = cv2.resize(canvas, (64, 64), interpolation=cv2.INTER_NEAREST)
    try:
        text = pytesseract.image_to_string(canvas, config=_SINGLE_CELL_CONFIG)
    except Exception:
        return 0
    text = text.strip()
    return int(text) if len(text) == 1 and text in _DIGITS else 0
