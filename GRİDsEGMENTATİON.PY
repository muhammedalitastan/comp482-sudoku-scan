"""
Stage 5 helper - cell extraction.

Splits a 450x450 rectified image into 81 cells (50x50 px each) and crops an
inner margin from every cell so the grid lines are excluded before OCR.
"""

GRID_SIZE = 9
TARGET_SIZE = 450
MARGIN_RATIO = 0.15


def extract_sudoku_cells(
    warped,
    grid_size: int = GRID_SIZE,
    target_size: int = TARGET_SIZE,
    margin_ratio: float = MARGIN_RATIO,
):
    """
    Returns a grid_size x grid_size list of cell images with the inner
    margin removed.
    """
    if warped.shape[0] != target_size or warped.shape[1] != target_size:
        raise ValueError(
            f"expected a {target_size}x{target_size} rectified image, "
            f"got {warped.shape[1]}x{warped.shape[0]}"
        )

    cell_size = target_size // grid_size
    margin = int(cell_size * margin_ratio)

    cells = []
    for row in range(grid_size):
        row_cells = []
        for col in range(grid_size):
            y0, x0 = row * cell_size, col * cell_size
            cell = warped[y0 : y0 + cell_size, x0 : x0 + cell_size]
            row_cells.append(cell[margin : cell_size - margin, margin : cell_size - margin])
        cells.append(row_cells)

    return cells
