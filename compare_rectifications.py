#!/usr/bin/env python3
"""Stage 4 metric script - A3 vs A4 vs H4 comparison.

Per image: corner residuals (from the result JSON), grid-line straightness
(from the result JSON) and the SAME prepared-cell OCR run on all three
rectifications, scored against ground truth. Obliqueness is joined from an
optional CSV index (columns "image" + "side_ratio" or "obliqueness_deg").

Usage (dev set):
    python compare_rectifications.py
Usage (personal set):
    python compare_rectifications.py --results out/ --annotations labels/ \
        --index my_index.csv --out comparison.csv
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from GRİDsEGMENTATİON import extract_sudoku_cells
from SayıKontrolVeTanıma import predict_sudoku_grid

HERE = Path(__file__).resolve().parent


def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def ocr_accuracy(gt_grid, pred_grid) -> float | None:
    """Fraction of given (non-zero) cells recognized correctly."""
    if pred_grid is None:
        return None
    gt = np.asarray(gt_grid)
    pred = np.asarray(pred_grid)
    givens = gt != 0
    if not givens.any():
        return None
    return float(np.sum((pred == gt) & givens) / givens.sum())


def _imread_gray(path: Path):
    """Read a grayscale image; imdecode/fromfile handles non-ASCII paths."""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


def read_rect_grid(rect_path: Path):
    """Extract cells from a saved 450x450 rectification and run prepared OCR."""
    image = _imread_gray(rect_path)
    if image is None:
        return None
    cells = extract_sudoku_cells(image)
    return predict_sudoku_grid(cells, prepared=True)


def collect_rows(results_dir: Path, annotations_dir: Path, index: dict) -> list:
    rows = []
    for ann_path in sorted(annotations_dir.glob("*.json")):
        ann = load_json(ann_path)
        name = Path(ann["image"]).stem
        result_path = results_dir / f"{name}_result.json"
        if not result_path.exists():
            print(f"SKIP {name}: no result JSON in {results_dir}")
            continue

        result = load_json(result_path)
        dev = result.get("grid_line_deviation") or {}
        meta = index.get(ann["image"], {})

        obliqueness = meta.get("side_ratio", meta.get("obliqueness_deg", ""))
        try:
            obliqueness = float(obliqueness)
        except (TypeError, ValueError):
            obliqueness = None

        row = {
            "image": ann["image"],
            "obliqueness": obliqueness,
            "A3_4th_corner_error_px": result.get("a3_corner_err_px"),
            "A4_rms_error_px": result.get("a4_rms_px"),
            "line_dev_A3": (dev.get("A3") or {}).get("overall_mean"),
            "line_dev_A4": (dev.get("A4") or {}).get("overall_mean"),
            "line_dev_H4": (dev.get("H4") or {}).get("overall_mean"),
        }

        for method in ("A3", "A4", "H4"):
            rect_path = results_dir / f"{name}_rect_{method}.png"
            grid = read_rect_grid(rect_path) if rect_path.exists() else None
            row[f"ocr_accuracy_{method}"] = ocr_accuracy(ann["grid"], grid)

        rows.append(row)
        print(
            f"{name}: A3err={_f(row['A3_4th_corner_error_px'], 1)} "
            f"A4rms={_f(row['A4_rms_error_px'], 1)} "
            f"ocr A3/A4/H4={_f(row['ocr_accuracy_A3'], 3)}/"
            f"{_f(row['ocr_accuracy_A4'], 3)}/{_f(row['ocr_accuracy_H4'], 3)}"
        )
    return rows


def print_summary(rows: list) -> None:
    keys = [
        "A3_4th_corner_error_px",
        "A4_rms_error_px",
        "line_dev_A3",
        "line_dev_A4",
        "line_dev_H4",
        "ocr_accuracy_A3",
        "ocr_accuracy_A4",
        "ocr_accuracy_H4",
    ]
    print("\nMeans over evaluated images:")
    for key in keys:
        vals = [r[key] for r in rows if r[key] is not None]
        mean = float(np.mean(vals)) if vals else None
        print(f"  {key:28s} {_f(mean, 4)}")

    # Corner residual vs obliqueness (sorted for the report figure)
    print("\nCorner residual vs obliqueness (sorted):")
    print(f"  {'image':10s} {'obliqueness':>12s} {'A3err':>8s} {'A4rms':>8s}")
    for r in sorted(rows, key=lambda r: _float(r["obliqueness"])):
        print(
            f"  {r['image']:10s} {_f(r['obliqueness'], 3):>12s} "
            f"{_f(r['A3_4th_corner_error_px'], 1):>8s} {_f(r['A4_rms_error_px'], 1):>8s}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=str(HERE / "devset_results"), help="directory with *_result.json and *_rect_*.png")
    parser.add_argument("--annotations", default=str(HERE.parent / "SudokuScan_devset" / "annotations"), help="directory with annotation JSON files")
    parser.add_argument("--index", default=str(HERE.parent / "SudokuScan_devset" / "index.csv"), help="CSV with image + side_ratio (optional join)")
    parser.add_argument("--out", default=str(HERE / "devset_results" / "rectification_comparison.csv"), help="output CSV path")
    args = parser.parse_args()

    index = {}
    index_path = Path(args.index)
    if index_path.exists():
        with open(index_path, newline="", encoding="utf-8") as f:
            index = {row["image"]: row for row in csv.DictReader(f)}
    else:
        print(f"Note: no index CSV at {index_path}; obliqueness column will be empty")

    rows = collect_rows(Path(args.results), Path(args.annotations), index)
    if not rows:
        raise SystemExit("No matching results found.")

    print_summary(rows)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {args.out}")


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("inf")


def _f(value, digits):
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    main()
