"""Stage 4 metrics - grid-line straightness on the 450x450 rectified image.

Morphological opening removes digit strokes so only the grid lines remain;
each of the 10 horizontal + 10 vertical lines is matched to its ideal 50 px
position and the deviation is measured.
"""

import cv2
import numpy as np

from PersPektifTransform import CELL_SIZE, TARGET_SIZE, ideal_grid_lines

_SEARCH_RADIUS = 25  # px around each ideal position to look for the line
_LINE_KERNEL = 35  # structuring element length; longer than any digit stroke
_MIN_PEAK = 255 * 30  # minimum dark-pixel count for a line to be "found"
_PEAK_GAP = 10  # peaks closer than this are the same physical line


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


def _line_peaks(profile: np.ndarray) -> list[int]:
    """Strong local ridges, one index each. Nearby ridges collapse into one peak."""
    peaks: list[int] = []
    for idx in np.argsort(profile)[::-1]:
        strength = int(profile[idx])
        if strength < _MIN_PEAK:
            break
        index = int(idx)
        if any(abs(index - kept) < _PEAK_GAP for kept in peaks):
            continue
        peaks.append(index)
    return peaks


def _measure_axis(profile: np.ndarray, ideal_positions) -> dict:
    """Match each ideal line to a distinct nearby peak; report deviations.

    A peak is given to at most one ideal position. Ideal 450 is kept as 450
    even though the profile ends at pixel 449; the border ink on that last
    pixel is then 1 px from the outer edge.
    """
    ideal_list = [int(p) for p in ideal_positions]
    peaks = _line_peaks(profile)

    pairs = []
    for index, ideal in enumerate(ideal_list):
        for peak in peaks:
            distance = abs(peak - ideal)
            if distance <= _SEARCH_RADIUS:
                pairs.append((distance, index, peak))
    pairs.sort()

    chosen: dict[int, tuple[int, int]] = {}
    used_peaks: set[int] = set()
    for distance, index, peak in pairs:
        if index in chosen or peak in used_peaks:
            continue
        chosen[index] = (peak, distance)
        used_peaks.add(peak)

    detected, deviations = [], []
    for index, ideal in enumerate(ideal_list):
        if index not in chosen:
            detected.append(None)
            deviations.append(None)
            continue
        actual, distance = chosen[index]
        detected.append(int(actual))
        deviations.append(float(distance))

    found = [d for d in deviations if d is not None]
    # A missed line counts as one full search radius, so a single lucky hit
    # cannot report a near-zero error. `mean` stays the error of lines that
    # were actually found.
    penalized = [d if d is not None else float(_SEARCH_RADIUS) for d in deviations]
    return {
        "ideal": ideal_list,
        "detected": detected,
        "deviations": deviations,
        "lines_found": len(found),
        "lines_expected": len(ideal_list),
        "mean": float(np.mean(found)) if found else None,
        "mean_penalized": float(np.mean(penalized)) if penalized else None,
    }


def calculate_grid_line_straightness(warped: np.ndarray) -> dict:
    """Deviation of the 10+10 grid lines from their ideal 50 px positions."""
    binary = _binarize(warped)
    ideal = ideal_grid_lines()  # 0, 50, ..., 400, 450

    horizontal = _measure_axis(_line_profile(binary, "h"), ideal)
    vertical = _measure_axis(_line_profile(binary, "v"), ideal)

    means = [m for m in (horizontal["mean"], vertical["mean"]) if m is not None]
    penalized = [
        m for m in (horizontal["mean_penalized"], vertical["mean_penalized"]) if m is not None
    ]

    return {
        "horizontal": horizontal,
        "vertical": vertical,
        "overall_mean": float(np.mean(means)) if means else None,
        "overall_mean_penalized": float(np.mean(penalized)) if penalized else None,
    }


def _empty_metrics():
    return {
        "cell_accuracy": None,
        "given_accuracy": None,
        "false_positives": None,
        "empty_precision": None,
        "empty_recall": None,
        "empty_f1": None,
        "exact_match": None,
        "correct": None,
        "total": None,
    }


def recognition_metrics(grid, ground_truth) -> dict:
    """Section 7 recognition metrics for one image.

    cell_accuracy is the fraction of all 81 cells, empty or filled, that match.
    given_accuracy is that fraction restricted to non-zero ground-truth cells.
    empty_f1 scores the empty-versus-filled decision with empty as the positive
    class. It is None when the ground truth has no empty cell, because recall
    is then undefined. given_accuracy is None when the ground truth has no givens.
    """
    if grid is None or ground_truth is None:
        return _empty_metrics()

    pred = np.asarray(grid)
    gt = np.asarray(ground_truth)
    if pred.shape != gt.shape:
        raise ValueError(f"grid shape {pred.shape} does not match ground truth {gt.shape}")

    correct = int(np.sum(pred == gt))
    total = int(gt.size)
    givens = gt != 0
    n_givens = int(np.sum(givens))
    given_correct = int(np.sum((pred == gt) & givens))
    false_positives = int(np.sum((pred != 0) & ~givens))

    pred_empty = pred == 0
    gt_empty = ~givens
    true_positive = int(np.sum(pred_empty & gt_empty))
    false_empty = int(np.sum(pred_empty & ~gt_empty))
    missed_empty = int(np.sum(~pred_empty & gt_empty))
    if true_positive + missed_empty == 0:
        precision = recall = f1 = None
    else:
        precision = true_positive / (true_positive + false_empty) if (true_positive + false_empty) else 0.0
        recall = true_positive / (true_positive + missed_empty)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "cell_accuracy": (correct / total) if total else None,
        "given_accuracy": (given_correct / n_givens) if n_givens else None,
        "false_positives": false_positives,
        "empty_precision": precision,
        "empty_recall": recall,
        "empty_f1": f1,
        "exact_match": bool(total and correct == total),
        "correct": correct,
        "total": total,
    }


def calculate_ocr_accuracy(grid: np.ndarray, ground_truth=None):
    """OCR statistics, or Section 7 accuracy when a ground-truth grid is given.

    With ground truth, accuracy is the fraction of all cells (digit or empty).
    It is not divided by the number of givens, which previously counted
    empty-empty matches in the numerator and could exceed 1.
    """
    grid = np.asarray(grid)

    if ground_truth is None:
        non_empty = int(np.sum(grid != 0))
        present = grid[grid != 0]
        return {
            "non_empty_cells": non_empty,
            "fill_ratio": non_empty / float(grid.size if grid.size else 81),
            "unique_digits": int(len(np.unique(present))) if non_empty else 0,
        }

    metrics = recognition_metrics(grid, ground_truth)
    return {
        "accuracy": metrics["cell_accuracy"],
        "correct": metrics["correct"],
        "total": metrics["total"],
        "given_accuracy": metrics["given_accuracy"],
        "empty_f1": metrics["empty_f1"],
        "exact_match": metrics["exact_match"],
    }
