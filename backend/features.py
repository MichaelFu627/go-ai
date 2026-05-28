"""
Feature encoding — turn a Go position into the stack of binary planes the
neural network reads as input.

This is the "19x19x48 image stack" idea from the AlphaGo paper, simplified for
a 9x9 from-scratch project. We use a small set of planes that capture the most
important information. All planes are computed RELATIVE TO THE SIDE TO MOVE
(the paper does this too) — "my stones" / "opponent stones" rather than
black/white — so the same network weights work for both colors.

Plane layout (NUM_PLANES total), each is size×size of 0/1 values:

  0           : my stones
  1           : opponent stones
  2           : empty points
  3..6        : my liberties == 1, 2, 3, >=4  (one-hot by liberty count)
  7..10       : opponent liberties == 1, 2, 3, >=4
  11          : my most recent move (the single point I last played)
  12          : opponent most recent move
  13          : legal moves for the side to move (and not filling own eye)
  14          : all-ones constant plane (helps the net sense the board edge)
  15          : colour-to-move is black (all 1 if black to move, else all 0)

That's 16 planes. Easy to extend later (ladders, capture sizes, etc.).
"""

from __future__ import annotations
import numpy as np
from board import Board, BLACK, WHITE, EMPTY, opponent

NUM_PLANES = 16


def encode(board: Board, to_move: int,
           last_move=None, opp_last_move=None) -> np.ndarray:
    """Return a float32 array of shape (NUM_PLANES, size, size).

    last_move / opp_last_move are (r, c) tuples or None.
    """
    size = board.size
    me = to_move
    opp = opponent(me)
    planes = np.zeros((NUM_PLANES, size, size), dtype=np.float32)

    grid = board.grid

    # Precompute liberties for each stone via group BFS, caching per group.
    # We map each occupied point to its group's liberty count.
    lib_of = [[0] * size for _ in range(size)]
    seen = [[False] * size for _ in range(size)]
    for r in range(size):
        for c in range(size):
            if grid[r][c] != EMPTY and not seen[r][c]:
                group, libs = board._group_and_liberties(r, c)
                for (gr, gc) in group:
                    seen[gr][gc] = True
                    lib_of[gr][gc] = libs

    for r in range(size):
        for c in range(size):
            v = grid[r][c]
            if v == me:
                planes[0, r, c] = 1.0
                lc = lib_of[r][c]
                planes[3 + _lib_bucket(lc), r, c] = 1.0
            elif v == opp:
                planes[1, r, c] = 1.0
                lc = lib_of[r][c]
                planes[7 + _lib_bucket(lc), r, c] = 1.0
            else:
                planes[2, r, c] = 1.0

    # Last-move planes.
    if last_move is not None:
        planes[11, last_move[0], last_move[1]] = 1.0
    if opp_last_move is not None:
        planes[12, opp_last_move[0], opp_last_move[1]] = 1.0

    # Legal-and-sensible plane (uses fast legality; excludes obvious eyes).
    for (r, c) in board.fast_legal_moves(me):
        if not _is_obvious_eye(board, r, c, me):
            planes[13, r, c] = 1.0

    # Constant ones plane.
    planes[14, :, :] = 1.0

    # Colour-to-move plane.
    if me == BLACK:
        planes[15, :, :] = 1.0

    return planes


def _lib_bucket(liberties: int) -> int:
    """Map a liberty count to one of 4 buckets: 1, 2, 3, >=4 → index 0..3."""
    if liberties <= 1:
        return 0
    if liberties == 2:
        return 1
    if liberties == 3:
        return 2
    return 3


def _is_obvious_eye(board: Board, r: int, c: int, color: int) -> bool:
    """Same eye heuristic used elsewhere; duplicated here to keep this module
    free of dependencies on the AI package."""
    for nr, nc in board.neighbors(r, c):
        if board.grid[nr][nc] != color:
            return False
    diagonals = [(r-1, c-1), (r-1, c+1), (r+1, c-1), (r+1, c+1)]
    friendly = 0
    off_board = 0
    for dr, dc in diagonals:
        if not board.in_bounds(dr, dc):
            off_board += 1
        elif board.grid[dr][dc] == color:
            friendly += 1
    if off_board > 0:
        return (friendly + off_board) == 4
    return friendly >= 3


def move_to_index(move, size: int) -> int:
    """Map a move to a policy index. move is (r, c) or None (=pass).
    Indices 0..size*size-1 are board points (row-major); size*size is pass."""
    if move is None:
        return size * size
    return move[0] * size + move[1]


def index_to_move(index: int, size: int):
    """Inverse of move_to_index. Returns (r, c) or None for pass."""
    if index == size * size:
        return None
    return (index // size, index % size)


def policy_size(size: int) -> int:
    """Number of policy outputs: every point plus one pass action."""
    return size * size + 1
