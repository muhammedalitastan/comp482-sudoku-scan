"""Stage 6 - Sudoku rule check: no digit repeats in a row, column or 3x3 box."""

import numpy as np


def _units():
    """Yield the 27 Sudoku units as (row, col) lists."""
    for r in range(9):
        yield [(r, c) for c in range(9)]
    for c in range(9):
        yield [(r, c) for r in range(9)]
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            yield [(br + i, bc + j) for i in range(3) for j in range(3)]


def check_sudoku_validity(grid) -> tuple:
    """Return (is_valid, conflicts); conflicts is a sorted (row, col) list."""
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
