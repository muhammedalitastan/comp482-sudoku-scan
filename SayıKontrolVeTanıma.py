"""
Stage 5 - digit recognition with Tesseract OCR.

Every cell is first checked with an ink-ratio criterion so empty cells never
reach the OCR engine. Recognition is restricted to the digits 1-9 (Sudoku
grids contain no zeros).

All non-empty cells are placed on a single montage image and recognized
with one Tesseract call per pass (two passes per mode at most), which
keeps the runtime far below the 20 s limit. Character boxes are mapped
back to cells by their center position.

Two OCR modes are supported:
  prepared=True  -> crop to the digit bounding box, force the background
                    to white and center the digit on a canvas (ink keeps
                    its gray value; only the background is normalized)
  prepared=False -> raw cell image (used for the accuracy comparison)
"""

import os

import cv2
import numpy as np
import pytesseract

_DEFAULT_TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(_DEFAULT_TESSERACT_CMD):
    pytesseract.pytesseract.tesseract_cmd = _DEFAULT_TESSERACT_CMD

_MONTAGE_CONFIG = r"--oem 3 --psm 6 -c tessedit_char_whitelist=123456789"
_MONTAGE_RETRY_CONFIG = r"--oem 3 --psm 11 -c tessedit_char_whitelist=123456789"
_MIN_INK_RATIO = 0.02
_DIGITS = "123456789"

_MONTAGE_CELL = 64  # slot size per cell in the montage
_MONTAGE_GAP = 16  # white gap between slots
_MONTAGE_INNER = 48  # digit canvas placed inside a slot
_MONTAGE_SIZE = 9 * _MONTAGE_CELL + 10 * _MONTAGE_GAP


def is_cell_empty(cell, min_ink_ratio: float = _MIN_INK_RATIO) -> bool:
    """True when the fraction of dark pixels is below the threshold."""
    if cell is None or cell.size == 0:
        return True
    _, binary = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink_ratio = cv2.countNonZero(binary) / cell.size
    return ink_ratio < min_ink_ratio


def preprocess_cell(cell: np.ndarray) -> np.ndarray:
    """
    Crop the cell to the digit bounding box and center it on a 28x28
    canvas. Inside the crop, Otsu binarization separates the dark ink from
    the paper; ink pixels keep their gray value and the background is
    forced to pure white so dark or unevenly lit photos produce the same
    high-contrast input as clean scans.
    """
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
    """
    Place every non-empty cell on one white montage image.
    `only` optionally restricts which (row, col) cells are placed.
    """
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
    """
    One Tesseract call on the montage; map character boxes back to cells.

    image_to_boxes returns one line per character:
        <char> <left> <bottom> <right> <top> <page>
    with y measured from the BOTTOM of the image.
    """
    grid = np.zeros((9, 9), dtype=int)
    height = montage.shape[0]

    try:
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
    """
    Recognize digits in a 9x9 matrix of cell images.

    Two montage passes: the first reads all non-empty cells at once; cells
    it missed are retried on their own montage with sparse-text mode, so
    faint digits are not lost while the runtime stays near one Tesseract
    call per pass.

    Returns a 9x9 numpy array (0 for empty cells).
    """
    grid = _read_montage(_build_montage(cells, prepared))

    missed = np.zeros((9, 9), dtype=bool)
    for row in range(9):
        for col in range(9):
            if grid[row, col] == 0 and not is_cell_empty(cells[row][col]):
                missed[row, col] = True
    if missed.any():
        retry = _read_montage(
            _build_montage(cells, prepared, only=missed), config=_MONTAGE_RETRY_CONFIG
        )
        grid = np.where((grid == 0) & missed, retry, grid)

    return grid
