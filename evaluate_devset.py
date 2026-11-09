#!/usr/bin/env python3
"""Score the pipeline against the dev-set annotations.

Joins each result JSON with its ground-truth annotation and reports corner
error, A3/A4 rectification error, grid-line deviation, OCR accuracy
(prepared vs unprepared) and verdict. Writes devset_evaluation.csv.

Usage:
    python sudoku_scan.py --batch ../SudokuScan_devset/images --outdir devset_results
    python evaluate_devset.py
"""

import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEVSET = HERE.parent / "SudokuScan_devset"
RESULTS = HERE / "devset_results"


def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def corner_errors(gt_corners, pred_corners):
    """Mean and max Euclidean distance between matched corners."""
    if pred_corners is None:
        return None, None
    gt = np.asarray(gt_corners, dtype=float)
    pred = np.asarray(pred_corners, dtype=float)
    dists = np.linalg.norm(gt - pred, axis=1)
    return float(dists.mean()), float(dists.max())


def ocr_accuracy(gt_grid, pred_grid):
    """Fraction of given (non-zero) cells recognized correctly, plus false positives."""
    if pred_grid is None:
        return None, None
    gt = np.asarray(gt_grid)
    pred = np.asarray(pred_grid)
    givens = gt != 0
    correct = int(np.sum((pred == gt) & givens))
    false_positives = int(np.sum((pred != 0) & (gt == 0)))
    return correct / int(givens.sum()), false_positives


def main():
    annotations = sorted((DEVSET / "annotations").glob("*.json"))
    if not annotations:
        raise SystemExit(f"No annotations found in {DEVSET / 'annotations'}")

    with open(DEVSET / "index.csv", newline="", encoding="utf-8") as f:
        index = {row["image"]: row for row in csv.DictReader(f)}

    rows = []
    for ann_path in annotations:
        ann = load_json(ann_path)
        name = Path(ann["image"]).stem
        result_path = RESULTS / f"{name}_result.json"
        if not result_path.exists():
            print(f"SKIP {name}: no result JSON in {RESULTS}")
            continue
        result = load_json(result_path)

        gt_corners = ann["corners"]
        pred_corners = result.get("corners")
        err_mean, err_max = corner_errors(gt_corners, pred_corners)

        acc_prepared, fp_prepared = ocr_accuracy(ann["grid"], result.get("grid"))
        acc_unprepared, fp_unprepared = ocr_accuracy(ann["grid"], result.get("grid_unprepared"))

        meta = index.get(ann["image"], {})
        dev = result.get("grid_line_deviation") or {}
        rows.append(
            {
                "image": ann["image"],
                "conditions": meta.get("conditions", ""),
                "side_ratio": meta.get("side_ratio", ""),
                "verdict": result.get("verdict", ""),
                "corner_err_mean_px": err_mean,
                "corner_err_max_px": err_max,
                "A3_BR_error_px": result.get("a3_corner_err_px"),
                "A4_rms_error_px": result.get("a4_rms_px"),
                "line_dev_A3": (dev.get("A3") or {}).get("overall_mean"),
                "line_dev_A4": (dev.get("A4") or {}).get("overall_mean"),
                "line_dev_H4": (dev.get("H4") or {}).get("overall_mean"),
                "ocr_accuracy_prepared": acc_prepared,
                "ocr_accuracy_unprepared": acc_unprepared,
                "false_positives_prepared": fp_prepared,
                "false_positives_unprepared": fp_unprepared,
                "runtime_s": result.get("runtime_s"),
            }
        )

    if not rows:
        raise SystemExit("No matching results found.")

    # Console summary
    header = [
        "image", "cond", "side_ratio", "verdict",
        "corner_mean", "corner_max", "A3err", "A4rms",
        "devA3", "devA4", "devH4", "ocr_pre", "ocr_unpre", "runtime",
    ]
    print(("{:9s} {:12s} {:>10s} {:>10s} {:>11s} {:>10s} {:>7s} {:>7s} {:>6s} {:>6s} {:>6s} {:>8s} {:>9s} {:>7s}").format(*header))
    for r in rows:
        print(
            "{:9s} {:12s} {:>10s} {:>10s} {:>11} {:>10} {:>7} {:>7} {:>6} {:>6} {:>6} {:>8} {:>9} {:>7}".format(
                r["image"], r["conditions"], r["side_ratio"], r["verdict"],
                _f(r["corner_err_mean_px"], 1), _f(r["corner_err_max_px"], 1),
                _f(r["A3_BR_error_px"], 1), _f(r["A4_rms_error_px"], 1),
                _f(r["line_dev_A3"], 2), _f(r["line_dev_A4"], 2), _f(r["line_dev_H4"], 2),
                _f(r["ocr_accuracy_prepared"], 3), _f(r["ocr_accuracy_unprepared"], 3),
                _f(r["runtime_s"], 1),
            )
        )

    def mean(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return float(np.mean(vals)) if vals else None

    print("\nMeans over evaluated images:")
    for key, label in [
        ("corner_err_mean_px", "corner error (mean)"),
        ("A3_BR_error_px", "A3 4th-corner error"),
        ("A4_rms_error_px", "A4 RMS error"),
        ("line_dev_A3", "grid-line deviation A3"),
        ("line_dev_A4", "grid-line deviation A4"),
        ("line_dev_H4", "grid-line deviation H4"),
        ("ocr_accuracy_prepared", "OCR accuracy prepared"),
        ("ocr_accuracy_unprepared", "OCR accuracy unprepared"),
    ]:
        print(f"  {label:32s} {mean(key)}")

    verdicts = {}
    for r in rows:
        verdicts[r["verdict"]] = verdicts.get(r["verdict"], 0) + 1
    print(f"\nVerdicts: {verdicts}")

    out_csv = RESULTS / "devset_evaluation.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {out_csv}")


def _f(value, digits):
    return f"{value:.{digits}f}" if value is not None else "n/a"


if __name__ == "__main__":
    main()
