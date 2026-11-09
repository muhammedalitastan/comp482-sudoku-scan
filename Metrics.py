"""Stage 4 metrics - grid-line straightness on the 450x450 rectified image.

Morphological opening removes digit strokes so only the grid lines remain;
each of the 10 horizontal + 10 vertical lines is matched to its ideal 50 px
position and the deviation is measured.
"""

import cv2
import numpy as np

TARGET_SIZE = 450
CELL_SIZE = TARGET_SIZE // 9  # 50 px per cell -> 10 lines per axis
_SEARCH_RADIUS = 25  # px around each ideal position to look for the line
_LINE_KERNEL = 35  # structuring element length; longer than any digit stroke
_MIN_PEAK = 255 * 30  # minimum dark-pixel count for a line to be "found"


def _binarize(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def _line_profile(binary: np.ndarray, orientation: str) -> np.ndarray:
    """Dark-pixel count per row ("h") or per column ("v") after line isolation."""
    if orientation == "h":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (_LINE_KERNEL, 1))
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, _LINE_KERNEL))
    lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return lines.sum(axis=1 if orientation == "h" else 0)


def _measure_axis(profile: np.ndarray, ideal_positions) -> dict:
    """Match each ideal line position to the strongest nearby line; report deviations."""
    ideal_list = [int(p) for p in ideal_positions]
    detected, deviations = [], []

    for ideal in ideal_list:
        ideal = min(ideal, len(profile) - 1)
        lo = max(0, ideal - _SEARCH_RADIUS)
        hi = min(len(profile), ideal + _SEARCH_RADIUS + 1)
        segment = profile[lo:hi]

        if int(segment.max()) < _MIN_PEAK:
            detected.append(None)
            deviations.append(None)
            continue

        actual = lo + int(segment.argmax())
        detected.append(int(actual))
        deviations.append(float(abs(actual - ideal)))

    found = [d for d in deviations if d is not None]
    return {
        "ideal": ideal_list,
        "detected": detected,
        "deviations": deviations,
        "lines_found": len(found),
        "mean": float(np.mean(found)) if found else None,
    }


def calculate_grid_line_straightness(warped: np.ndarray) -> dict:
    """Deviation of the 10+10 grid lines from their ideal 50 px positions."""
    binary = _binarize(warped)
    ideal = list(range(0, TARGET_SIZE + 1, CELL_SIZE))  # 0, 50, ..., 450

    horizontal = _measure_axis(_line_profile(binary, "h"), ideal)
    vertical = _measure_axis(_line_profile(binary, "v"), ideal)

    means = [m for m in (horizontal["mean"], vertical["mean"]) if m is not None]

    return {
        "horizontal": horizontal,
        "vertical": vertical,
        "overall_mean": float(np.mean(means)) if means else None,
    }


def calculate_ocr_accuracy(grid: np.ndarray, ground_truth=None):
    """
    OCR accuracy against a ground-truth grid, or basic grid statistics when
    no ground truth is available.
    """
    grid = np.asarray(grid)

    if ground_truth is None:
        non_empty = int(np.sum(grid != 0))
        return {
            "non_empty_cells": non_empty,
            "fill_ratio": non_empty / 81.0,
            "unique_digits": int(len(np.unique(grid[grid != 0]))),
        }

    ground_truth = np.asarray(ground_truth)
    correct = int(np.sum(grid == ground_truth))
    total = int(np.sum(ground_truth != 0))
    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "correct": correct,
        "total": total,
    }
