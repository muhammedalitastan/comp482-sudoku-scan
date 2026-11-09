# SudokuScan - COMP 482 Computer Vision Project 1

## Overview
This project converts printed Sudoku puzzle photos into 9x9 digital matrices using geometric transformations and OCR. The main research question is: "When is affine transformation sufficient, and when is homography (projective) transformation needed?"

## Requirements
- Python 3.10+
- OpenCV (cv2)
- NumPy
- pytesseract (with Tesseract OCR installed)
- argparse

## Installation
```bash
pip install -r requirements.txt
# or: pip install opencv-python numpy pytesseract
```

Install Tesseract OCR for Windows:
- Download from: https://github.com/UB-Mannheim/tesseract/wiki
- Install to default path or update the path in `SayıKontrolVeTanıma.py`

## Usage

### Single Image
```bash
python sudoku_scan.py photo.jpg --outdir results/
```

### Batch Processing
```bash
python sudoku_scan.py --batch images/ --outdir results/
```

### With Custom Harris Parameters
```bash
python sudoku_scan.py photo.jpg --outdir results/ --blockSize 3 --ksize 5 --threshold 0.02
```

## Output Files
For each input image, the following files are generated in the output directory:

1. `<name>_harris.png` - Harris corner response map with detected candidates and the selected 4 corners
2. `<name>_rect_A3.png` - Rectified image using A3-affine (3 points)
3. `<name>_rect_A4.png` - Rectified image using A4-affine (4 points with least squares)
4. `<name>_rect_H4.png` - Rectified image using H4-homography (4 points)
5. `<name>_debug.png` - H4 image with grid lines, recognized digits and conflicting cells highlighted
6. `<name>_grid.txt` - 9 rows of 9 digits, no separators (0 for empty cells)
7. `<name>_result.json` - Required fields: `corners`, `A3`/`A4` (2x3), `H4` (3x3), `a3_corner_err_px`, `a4_rms_px`, `grid`, `verdict` (`valid`/`invalid`/`not_found`), `runtime_s`; plus analysis extras: `grid_unprepared`, `grid_line_deviation`, `validity`, `ocr`, `stage_times`, `corner_selection_method`

## Pipeline Stages

### 1. Preprocessing
- Converts to grayscale
- Applies Gaussian blur with relative kernel size (1% of image width)
- Adaptive thresholding with relative block size (2% of image width)
- Canny edge detection with intensity-based thresholds
- Dilation for edge connection

### 2. Harris Corner Detection
- Uses `cv2.cornerHarris` for corner detection
- Applies non-maximum suppression
- Refines corners with `cv2.cornerSubPix`
- **Automatically selects 4 outer corners** using rule-based approach:
  - TL: minimum (x + y)
  - TR: maximum (x - y)
  - BR: maximum (x + y)
  - BL: minimum (x - y)

### 3. Rectification
Three transformation methods to 450x450 target (50px per cell):

- **A3-affine**: Uses 3 points (TL, TR, BL) with `cv2.getAffineTransform`
  - Reports 4th corner (BR) projection error
  
- **A4-affine**: Uses 4 points with `cv2.estimateAffine2D` (least squares)
  - Reports RMS error on all 4 corners
  
- **H4-homography**: Uses 4 points with `cv2.getPerspectiveTransform`
  - Perfect alignment of all 4 corners

### 4. Comparison Metrics
- **Corner residuals**: A3 4th corner error, A4 RMS error
- **Grid line straightness**: Deviation from ideal 50px intervals
- **OCR accuracy**: Prepared vs unprepared cell preprocessing

### 5. Cell Extraction & OCR
- Extracts 81 cells from 450x450 rectified image
- Applies inner margin (15%) to remove grid lines
- Detects empty cells using ink ratio threshold
- Preprocesses cells: digit bounding-box crop, ink/background separation (background forced to white), centering, padding (grayscale ink kept)
- OCR with Tesseract (digits 1-9 only): all cells on one montage image per pass; cells missed by the first pass are retried on their own montage in sparse-text mode

### 6. Validity Check
- Checks Sudoku rules:
  - No duplicates in rows
  - No duplicates in columns
  - No duplicates in 3x3 blocks
- A grid with fewer than 17 recognized digits is reported as `invalid` (recognition likely failed; a unique Sudoku puzzle needs at least 17 givens)
- Returns valid/invalid with conflict positions

## Module Descriptions

- `Preproccessing.py` - Image preprocessing with relative parameters (returns original, gray, blurred, threshold and edges)
- `ContourAndCornerDetection.py` - Harris corner detection (local maxima + NMS + subpixel refinement) and rule-based outer corner selection with quadrilateral validation
- `PersPektifTransform.py` - A3, A4, and H4 rectification methods
- `GRİDsEGMENTATİON.PY` - Cell extraction from rectified image
- `SayıKontrolVeTanıma.py` - OCR with cell preprocessing (prepared vs raw comparison)
- `ValidityCheck.py` - Sudoku validity checking
- `Metrics.py` - Grid-line straightness (10 horizontal + 10 vertical lines vs ideal 50 px positions) and OCR accuracy metrics
- `sudoku_scan.py` - Main CLI interface

## Dataset Requirements
Collect at least 10 photos from 3 different Sudoku sources with:
- Student ID and date visible on a handwritten card in each photo
- Various conditions: frontal, oblique (>40°), rotated, uneven lighting, complex backgrounds

## Important Notes
- **All corners must come from Harris detection** - no contour-based or manual selection
- **No deep learning models** for corner detection
- **All parameters are relative to image size** for scalability
- **Stage 2 (corner selection) and Stage 4 (comparison) must be written without AI assistance**

## Command Line Arguments
- `image` - Input image path (or use --batch)
- `--batch` - Directory for batch processing
- `--outdir` - Output directory (required)
- `--blockSize` - Harris blockSize (default: 2)
- `--ksize` - Harris aperture parameter (default: 3)
- `--k` - Harris free parameter (default: 0.04)
- `--threshold` - Harris threshold ratio (default: 0.01)
- `--nms-distance` - NMS distance (default: auto, 1% of the smallest image side)
