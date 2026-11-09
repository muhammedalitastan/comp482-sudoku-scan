#!/usr/bin/env python3
"""SudokuScan - COMP 482 Computer Vision Project 1.

Sudoku photo -> 9x9 digit matrix via Harris corners, A3/A4/H4
rectification, Tesseract OCR and a validity check.

Usage:
    python sudoku_scan.py input photo.jpg --outdir out/
    python sudoku_scan.py --batch images_dir/ --outdir results/
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from ContourAndCornerDetection import HarrisParams, find_sudoku_grid
from GRİDsEGMENTATİON import extract_sudoku_cells
from Metrics import calculate_grid_line_straightness
from PersPektifTransform import rectify_A3_affine, rectify_A4_affine, rectify_H4_homography
from Preproccessing import preprocess_image
from SayıKontrolVeTanıma import ocr_single_cell, predict_sudoku_grid
from ValidityCheck import check_sudoku_validity

CELL_SIZE = 50  # 450 / 9
MIN_SUDOKU_GIVENS = 17  # minimum givens for a unique Sudoku solution


def create_debug_image(warped: np.ndarray, grid: np.ndarray, conflicts) -> np.ndarray:
    """H4 image with grid lines, recognized digits and red conflict cells."""
    debug = (
        cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
        if warped.ndim == 2
        else warped.copy()
    )

    # Grid lines every 50 px
    for i in range(10):
        p = i * CELL_SIZE
        cv2.line(debug, (p, 0), (p, 449), (180, 180, 180), 1)
        cv2.line(debug, (0, p), (449, p), (180, 180, 180), 1)

    # Recognized digits, centered in their cells
    for row in range(9):
        for col in range(9):
            value = int(grid[row, col])
            if value:
                cx, cy = col * CELL_SIZE + 25, row * CELL_SIZE + 25
                cv2.putText(
                    debug, str(value), (cx - 9, cy + 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2,
                )

    # Conflicting cells: translucent red fill plus a red border
    overlay = debug.copy()
    for row, col in conflicts:
        x0, y0 = col * CELL_SIZE, row * CELL_SIZE
        cv2.rectangle(overlay, (x0, y0), (x0 + CELL_SIZE - 1, y0 + CELL_SIZE - 1), (0, 0, 255), -1)
    cv2.addWeighted(overlay, 0.25, debug, 0.75, 0, debug)
    for row, col in conflicts:
        x0, y0 = col * CELL_SIZE, row * CELL_SIZE
        cv2.rectangle(debug, (x0, y0), (x0 + CELL_SIZE - 1, y0 + CELL_SIZE - 1), (0, 0, 255), 2)

    return debug


def _empty_result(image_path: str) -> dict:
    """Result template; fields are filled in as the pipeline progresses."""
    return {
        # Required output fields (assignment spec)
        "corners": None,
        "A3": None,
        "A4": None,
        "H4": None,
        "a3_corner_err_px": None,
        "a4_rms_px": None,
        "grid": None,
        "verdict": "not_found",
        "runtime_s": 0.0,
        # Additional analysis fields
        "image_path": image_path,
        "message": "",
        "corner_selection_method": None,
        "grid_unprepared": None,
        "grid_line_deviation": None,
        "validity": {"is_valid": None, "conflicts": []},
        "ocr": {"prepared_non_empty": None, "unprepared_non_empty": None},
        "stage_times": {},
    }


def _write_result_json(result: dict, outdir: str, base_name: str) -> None:
    path = os.path.join(outdir, f"{base_name}_result.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  saved: {path}")


def process_single_image(image_path: str, outdir: str, harris_params: HarrisParams | None = None) -> dict:
    """Run the full pipeline on one image and write all output files."""
    start = time.time()
    base_name = Path(image_path).stem
    os.makedirs(outdir, exist_ok=True)

    result = _empty_result(image_path)
    stage_times = result["stage_times"]

    try:
        # Stage 1: Preprocessing
        t = time.time()
        pre = preprocess_image(image_path)
        stage_times["preprocessing"] = round(time.time() - t, 3)
        print(f"  image size: {pre.gray.shape[1]}x{pre.gray.shape[0]}")

        # Stage 2: Harris corner detection (contour finds the region only)
        t = time.time()
        try:
            detection = find_sudoku_grid(pre.blurred, pre.threshold, harris_params)
        except Exception as exc:
            # Detection failure must never crash: report it as not_found.
            result["message"] = f"corner detection failed: {type(exc).__name__}: {exc}"
            print(f"  corner detection failed: {exc}")
            _finish(result, outdir, base_name, start)
            return result
        stage_times["corner_detection"] = round(time.time() - t, 3)

        harris_path = os.path.join(outdir, f"{base_name}_harris.png")
        cv2.imwrite(harris_path, detection.harris_vis)
        print(f"  Harris candidates: {len(detection.candidates)}")
        print(f"  saved: {harris_path}")

        if detection.corners is None:
            result["message"] = detection.message
            print(f"  corners not found: {detection.message}")
            _finish(result, outdir, base_name, start)
            return result

        corners = detection.corners
        result["corners"] = corners.tolist()
        result["corner_selection_method"] = detection.method
        print(f"  selection method: {detection.method}")
        for label, corner in zip(["TL", "TR", "BR", "BL"], corners):
            print(f"  {label}: ({corner[0]:.1f}, {corner[1]:.1f})")

        # Stage 3: Rectification (A3, A4, H4)
        t = time.time()
        warped_a3, projected_br, error_a3, m_a3 = rectify_A3_affine(pre.gray, corners)
        warped_a4, rms_error_a4, m_a4 = rectify_A4_affine(pre.gray, corners)
        warped_h4, h_matrix = rectify_H4_homography(pre.gray, corners)
        stage_times["rectification"] = round(time.time() - t, 3)

        result["A3"] = m_a3.tolist()
        result["A4"] = m_a4.tolist()
        result["H4"] = h_matrix.tolist()
        result["a3_corner_err_px"] = float(error_a3)
        result["a4_rms_px"] = float(rms_error_a4)
        print(f"  A3 4th-corner error: {error_a3:.2f} px")
        print(f"  A4 RMS corner error: {rms_error_a4:.2f} px")

        for name, warped in (("A3", warped_a3), ("A4", warped_a4), ("H4", warped_h4)):
            path = os.path.join(outdir, f"{base_name}_rect_{name}.png")
            cv2.imwrite(path, warped)
            print(f"  saved: {path}")

        # Stage 4: Comparison metrics (grid-line straightness)
        t = time.time()

        def _fmt(value):
            return f"{value:.2f}" if value is not None else "n/a"

        straightness = {}
        for name, warped in (("A3", warped_a3), ("A4", warped_a4), ("H4", warped_h4)):
            straightness[name] = calculate_grid_line_straightness(warped)
            s = straightness[name]
            print(
                f"  {name} grid-line deviation: "
                f"H={_fmt(s['horizontal']['mean'])} px ({s['horizontal']['lines_found']}/10 found), "
                f"V={_fmt(s['vertical']['mean'])} px ({s['vertical']['lines_found']}/10 found)"
            )
        result["grid_line_deviation"] = straightness
        stage_times["metrics"] = round(time.time() - t, 3)

        # Stage 5: Cell extraction and OCR (prepared vs raw comparison)
        t = time.time()
        cells = extract_sudoku_cells(warped_h4)
        grid = predict_sudoku_grid(cells, prepared=True)
        grid_unprepared = predict_sudoku_grid(cells, prepared=False)
        stage_times["ocr"] = round(time.time() - t, 3)

        result["grid"] = grid.tolist()
        result["grid_unprepared"] = grid_unprepared.tolist()
        result["ocr"] = {
            "prepared_non_empty": int(np.count_nonzero(grid)),
            "unprepared_non_empty": int(np.count_nonzero(grid_unprepared)),
        }
        print(
            f"  recognized cells: prepared={result['ocr']['prepared_non_empty']}, "
            f"unprepared={result['ocr']['unprepared_non_empty']}"
        )

        # Stage 6: Validity check
        t = time.time()
        is_valid, conflicts = check_sudoku_validity(grid)

        # Conflict-driven re-recognition: cells involved in a rule violation
        # are re-read one by one in single-character mode. The refined grid
        # is kept only when it produces fewer conflicts.
        if conflicts:
            refined = grid.copy()
            for row, col in conflicts:
                digit = ocr_single_cell(cells[row][col])
                if digit:
                    refined[row, col] = digit
            refined_valid, refined_conflicts = check_sudoku_validity(refined)
            if len(refined_conflicts) < len(conflicts):
                grid = refined
                is_valid, conflicts = refined_valid, refined_conflicts
                result["grid"] = grid.tolist()
        stage_times["validity_check"] = round(time.time() - t, 3)

        result["validity"] = {
            "is_valid": bool(is_valid),
            "conflicts": [list(cell) for cell in conflicts],
        }
        recognized = int(np.count_nonzero(grid))
        if recognized < MIN_SUDOKU_GIVENS:
            # A real puzzle has at least 17 givens; fewer means OCR failed,
            # so the grid must not be reported as a valid Sudoku.
            result["verdict"] = "invalid"
            result["message"] = (
                f"only {recognized} digits recognized "
                f"(minimum {MIN_SUDOKU_GIVENS} required) - recognition likely failed"
            )
        else:
            result["verdict"] = "valid" if is_valid else "invalid"
        print(f"  validity: {result['verdict']} ({len(conflicts)} conflicting cells, {recognized} digits)")

        # Debug image on the H4 rectified view
        debug_path = os.path.join(outdir, f"{base_name}_debug.png")
        cv2.imwrite(debug_path, create_debug_image(warped_h4, grid, conflicts))
        print(f"  saved: {debug_path}")

        # Grid as text
        grid_path = os.path.join(outdir, f"{base_name}_grid.txt")
        with open(grid_path, "w", encoding="utf-8") as f:
            for row in grid:
                f.write("".join(map(str, row.tolist())) + "\n")
        print(f"  saved: {grid_path}")

    except Exception as exc:
        result["verdict"] = "error"
        result["message"] = f"{type(exc).__name__}: {exc}"
        print(f"  ERROR: {result['message']}")

    _finish(result, outdir, base_name, start)
    return result


def _finish(result: dict, outdir: str, base_name: str, start: float) -> None:
    result["runtime_s"] = round(time.time() - start, 3)
    print(f"  verdict: {result['verdict']}  runtime: {result['runtime_s']} s")
    _write_result_json(result, outdir, base_name)


def process_batch(input_dir: str, outdir: str, harris_params: HarrisParams | None = None) -> list:
    """Process every image in a directory."""
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    image_files = sorted(
        p for p in Path(input_dir).iterdir() if p.suffix.lower() in extensions
    )

    if not image_files:
        print(f"No image files found in {input_dir}")
        return []

    print(f"Found {len(image_files)} images to process")

    results = []
    for image_file in image_files:
        print(f"\n{'=' * 60}\nProcessing: {image_file}\n{'=' * 60}")
        results.append(process_single_image(str(image_file), outdir, harris_params))

    verdict_counts = {}
    for result in results:
        verdict_counts[result["verdict"]] = verdict_counts.get(result["verdict"], 0) + 1

    summary = {
        "total_images": len(image_files),
        "verdicts": verdict_counts,
        "mean_runtime_seconds": round(
            float(np.mean([r["runtime_s"] for r in results])), 3
        ),
    }
    summary_path = os.path.join(outdir, "batch_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nBatch summary: {summary}")
    print(f"saved: {summary_path}")

    return results


def main() -> None:
    # The assignment CLI writes "input photo.jpg"; accept both that form and
    # the bare "photo.jpg" positional form.
    argv = sys.argv[1:]
    if argv[:1] == ["input"]:
        argv = argv[1:]

    parser = argparse.ArgumentParser(
        description="SudokuScan - convert Sudoku photos to 9x9 digital matrices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sudoku_scan.py input photo.jpg --outdir out/
  python sudoku_scan.py photo.jpg --outdir results/
  python sudoku_scan.py --batch images_dir/ --outdir results/
  python sudoku_scan.py photo.jpg --outdir results/ --blockSize 3 --ksize 5 --threshold 0.02
        """,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("image", nargs="?", help="Input image path")
    input_group.add_argument("--batch", dest="batch_dir", help="Directory containing images")

    parser.add_argument("--outdir", required=True, help="Output directory")

    parser.add_argument("--blockSize", type=int, default=2, help="Harris blockSize (default: 2)")
    parser.add_argument("--ksize", type=int, default=3, help="Harris aperture parameter (default: 3)")
    parser.add_argument("--k", type=float, default=0.04, help="Harris free parameter (default: 0.04)")
    parser.add_argument(
        "--threshold", type=float, default=0.01,
        help="Harris threshold as a fraction of the max response (default: 0.01)",
    )
    parser.add_argument(
        "--nms-distance", type=int, default=None, dest="nms_distance",
        help="Minimum distance between Harris corners (default: auto, 1%% of the smallest image side)",
    )

    args = parser.parse_args(argv)

    harris_params = HarrisParams(
        block_size=args.blockSize,
        ksize=args.ksize,
        k=args.k,
        threshold_ratio=args.threshold,
        nms_distance=args.nms_distance,
    )

    if args.batch_dir:
        process_batch(args.batch_dir, args.outdir, harris_params)
    elif args.image:
        process_single_image(args.image, args.outdir, harris_params)
    else:
        parser.error("either an image path or --batch is required")


if __name__ == "__main__":
    main()
