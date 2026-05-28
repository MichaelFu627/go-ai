"""Random move selector with light common sense.

The naive 'pick any legal move' AI plays catastrophically badly because it
fills its own eyes (which is suicide in spirit and gives up huge territory).
We screen those out, falling back to pass when only such moves remain — this
also makes the game end naturally.
"""
from __future__ import annotations
import random
from typing import Optional
from game import Game
from board import EMPTY, BLACK, WHITE, opponent
from .base import AIDecision


def _is_obvious_eye(game: Game, r: int, c: int, color: int) -> bool:
    """Heuristic: an 'eye' is an empty point where all orthogonal neighbors
    are our own color (or board edge), AND most diagonal neighbors are ours.
    Filling such a point is almost always self-harm. This is the same rule
    AlphaGo uses for its 'sensibleness' feature.
    """
    board = game.board
    # All orthogonal neighbors must be ours (off-board counts as ours).
    for nr, nc in board.neighbors(r, c):
        if board.grid[nr][nc] != color:
            return False
    # Diagonals: count own + edge as friendly. Most must be friendly.
    diagonals = [(r-1, c-1), (r-1, c+1), (r+1, c-1), (r+1, c+1)]
    friendly = 0
    off_board = 0
    for dr, dc in diagonals:
        if not board.in_bounds(dr, dc):
            off_board += 1
        elif board.grid[dr][dc] == color:
            friendly += 1
    # On the edge, all on-board diagonals must be friendly.
    # In the middle, at most one hostile diagonal allowed.
    if off_board > 0:
        return (friendly + off_board) == 4
    return friendly >= 3


class RandomAI:
    """Picks a uniformly random legal move that isn't an obvious eye-fill.
    Passes when no sensible move is available."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def select_move(self, game: Game) -> AIDecision:
        color = game.turn
        legal = game.board.legal_moves(color)
        sensible = [
            (r, c) for (r, c) in legal
            if not _is_obvious_eye(game, r, c, color)
        ]
        if not sensible:
            return AIDecision(kind="pass", simulations=0)
        r, c = self.rng.choice(sensible)
        return AIDecision(kind="play", row=r, col=c, simulations=0)
