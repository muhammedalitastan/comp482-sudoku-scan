"""
Stage 6 - Sudoku validity check.

A grid is valid when no digit 1-9 appears twice in the same row, column or
3x3 box. Empty cells (0) are ignored. Returns the conflicting cell
positions so they can be highlighted in the debug image.
"""

import numpy as np


def _units():
    """Yield the 27 Sudoku units (9 rows, 9 columns, 9 boxes) as (row, col) lists."""
    for r in range(9):
        yield [(r, c) for c in range(9)]
    for c in range(9):
        yield [(r, c) for r in range(9)]
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            yield [(br + i, bc + j) for i in range(3) for j in range(3)]


def check_sudoku_validity(grid) -> tuple:
    """
    Returns (is_valid, conflicts) where conflicts is a sorted list of
    (row, col) tuples involved in at least one rule violation.
    """
    grid = np.asarray(grid)
    conflicts = set()

    for unit in _units():
        seen = {}
        for r, c in unit:
            value = grid[r, c]
            if value == 0:
                continue
            if value in seen:
                conflicts.add((r, c))
                conflicts.add(seen[value])
            else:
                seen[value] = (r, c)

    return len(conflicts) == 0, sorted(conflicts)
