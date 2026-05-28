"""
Go board core logic: stone placement, capture, suicide rule, positional superko.

Design notes:
- Board is an N x N numpy-like 2D list of ints: 0=empty, 1=black, 2=white.
- Groups (connected stones of same color) and their liberties are computed
  on demand via BFS. This is simple and correct; performance is fine for 9x9.
- Ko rule: positional superko (no whole-board position may repeat).
  We use Zobrist hashing to make repeat detection O(1) per move.
"""

from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Optional


EMPTY = 0
BLACK = 1
WHITE = 2


def opponent(color: int) -> int:
    return WHITE if color == BLACK else BLACK


# Zobrist table is generated once at import time and shared across all boards.
# Indexed as ZOBRIST[row][col][color] where color is 1 or 2.
# Using a fixed seed keeps hashes stable across runs (helps debugging).
_RNG = random.Random(0xA1F4607F)
_MAX_SIZE = 25
ZOBRIST = [
    [
        [0, _RNG.getrandbits(64), _RNG.getrandbits(64)]
        for _ in range(_MAX_SIZE)
    ]
    for _ in range(_MAX_SIZE)
]


@dataclass
class MoveResult:
    """Outcome of a legal move."""
    captured: int                    # number of opponent stones captured
    captured_positions: list[tuple[int, int]] = field(default_factory=list)


class IllegalMove(Exception):
    """Raised when a move violates the rules (occupied / suicide / ko)."""
    pass


class Board:
    """A Go board with full rule enforcement.

    Coordinates are (row, col), both 0-indexed, with row 0 at the top.
    """

    __slots__ = ("size", "grid", "_hash", "_history_hashes")

    def __init__(self, size: int = 9):
        if not (2 <= size <= _MAX_SIZE):
            raise ValueError(f"Board size must be in [2, {_MAX_SIZE}]")
        self.size = size
        self.grid: list[list[int]] = [[EMPTY] * size for _ in range(size)]
        self._hash: int = 0
        # Set of all past position hashes (including current) for superko.
        self._history_hashes: set[int] = {0}

    # ---- copying ----------------------------------------------------------

    def copy(self) -> "Board":
        b = Board.__new__(Board)
        b.size = self.size
        b.grid = [row[:] for row in self.grid]
        b._hash = self._hash
        b._history_hashes = set(self._history_hashes)
        return b

    # ---- queries ----------------------------------------------------------

    def __getitem__(self, pos: tuple[int, int]) -> int:
        r, c = pos
        return self.grid[r][c]

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.size and 0 <= c < self.size

    def neighbors(self, r: int, c: int):
        if r > 0: yield r - 1, c
        if r < self.size - 1: yield r + 1, c
        if c > 0: yield r, c - 1
        if c < self.size - 1: yield r, c + 1

    def _group_and_liberties(self, r: int, c: int) -> tuple[list[tuple[int, int]], int]:
        """BFS the connected group containing (r, c) and count its liberties."""
        color = self.grid[r][c]
        if color == EMPTY:
            return [], 0
        seen: set[tuple[int, int]] = set()
        lib_seen: set[tuple[int, int]] = set()
        group: list[tuple[int, int]] = []
        stack = [(r, c)]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            group.append(cur)
            cr, cc = cur
            for nr, nc in self.neighbors(cr, cc):
                v = self.grid[nr][nc]
                if v == color and (nr, nc) not in seen:
                    stack.append((nr, nc))
                elif v == EMPTY:
                    lib_seen.add((nr, nc))
        return group, len(lib_seen)

    # ---- mutation ---------------------------------------------------------

    def _toggle(self, r: int, c: int, color: int) -> None:
        """Place or remove a stone, updating the Zobrist hash."""
        self.grid[r][c] = color
        if color != EMPTY:
            self._hash ^= ZOBRIST[r][c][color]
        # Note: callers handle removing the old value's hash before placing
        # a new one. For our usage we always go EMPTY <-> color, so a single
        # XOR is enough per cell when used symmetrically.

    def play(self, r: int, c: int, color: int, check_superko: bool = True) -> MoveResult:
        """Play a stone, enforcing all rules. Raises IllegalMove if invalid.

        check_superko=False skips the positional-superko test (and doesn't
        record the position hash). Used in MCTS rollouts for speed; harmless
        there because rollouts are throwaway random games.
        """
        if not self.in_bounds(r, c):
            raise IllegalMove(f"out of bounds: ({r},{c})")
        if self.grid[r][c] != EMPTY:
            raise IllegalMove(f"point occupied: ({r},{c})")

        # Tentatively place the stone (update hash too).
        self.grid[r][c] = color
        new_hash = self._hash ^ ZOBRIST[r][c][color]

        # Capture opponent groups with zero liberties.
        opp = opponent(color)
        captured_positions: list[tuple[int, int]] = []
        for nr, nc in self.neighbors(r, c):
            if self.grid[nr][nc] == opp:
                grp, libs = self._group_and_liberties(nr, nc)
                if libs == 0:
                    for gr, gc in grp:
                        self.grid[gr][gc] = EMPTY
                        new_hash ^= ZOBRIST[gr][gc][opp]
                    captured_positions.extend(grp)

        # Suicide check: own group must still have at least one liberty.
        _, my_libs = self._group_and_liberties(r, c)
        if my_libs == 0:
            # Roll back: restore captured stones and the placed stone.
            for gr, gc in captured_positions:
                self.grid[gr][gc] = opp
            self.grid[r][c] = EMPTY
            raise IllegalMove("suicide")

        # Positional superko check.
        if check_superko and new_hash in self._history_hashes:
            # Roll back.
            for gr, gc in captured_positions:
                self.grid[gr][gc] = opp
            self.grid[r][c] = EMPTY
            raise IllegalMove("ko (position would repeat)")

        # Commit.
        self._hash = new_hash
        if check_superko:
            self._history_hashes.add(new_hash)
        return MoveResult(captured=len(captured_positions),
                          captured_positions=captured_positions)

    def is_legal(self, r: int, c: int, color: int) -> bool:
        """Non-throwing legality check. Does not mutate the board."""
        if not self.in_bounds(r, c) or self.grid[r][c] != EMPTY:
            return False
        probe = self.copy()
        try:
            probe.play(r, c, color)
            return True
        except IllegalMove:
            return False

    def fast_legal(self, r: int, c: int, color: int) -> bool:
        """Fast legality check WITHOUT copying the board or checking superko.

        Used in MCTS rollouts where speed matters and the rare superko
        violation is harmless. Real moves still go through play()/is_legal().

        A move is legal (ignoring ko) iff, after placing it, the played stone's
        group has at least one liberty. We detect this cheaply:
          - if any orthogonal neighbor is empty → immediate liberty, legal
          - else if any neighboring enemy group has exactly 1 liberty → we
            capture it, freeing space → legal
          - else if any neighboring friendly group has > 1 liberty → we share
            its remaining liberties → legal
          - otherwise it's suicide → illegal
        """
        if self.grid[r][c] != EMPTY:
            return False
        opp = opponent(color)
        for nr, nc in self.neighbors(r, c):
            v = self.grid[nr][nc]
            if v == EMPTY:
                return True  # has a direct liberty
            if v == opp:
                _, libs = self._group_and_liberties(nr, nc)
                if libs == 1:
                    return True  # captures an enemy group
            elif v == color:
                _, libs = self._group_and_liberties(nr, nc)
                if libs > 1:
                    return True  # connects to a friendly group with spare liberties
        return False

    def legal_moves(self, color: int) -> list[tuple[int, int]]:
        """Return all legal move coordinates for the given color.
        Uses full (copy-based) checking including superko — correct but slower.
        For rollouts, prefer fast_legal_moves()."""
        moves = []
        for r in range(self.size):
            for c in range(self.size):
                if self.grid[r][c] != EMPTY:
                    continue
                # Use a copy to avoid corrupting history hashes.
                probe = self.copy()
                try:
                    probe.play(r, c, color)
                    moves.append((r, c))
                except IllegalMove:
                    pass
        return moves

    def fast_legal_moves(self, color: int) -> list[tuple[int, int]]:
        """All legal moves using the fast (no-copy, no-superko) check."""
        moves = []
        for r in range(self.size):
            row = self.grid[r]
            for c in range(self.size):
                if row[c] != EMPTY:
                    continue
                if self.fast_legal(r, c, color):
                    moves.append((r, c))
        return moves

    # ---- debug ------------------------------------------------------------

    def __str__(self) -> str:
        symbols = {EMPTY: ".", BLACK: "X", WHITE: "O"}
        rows = []
        for r in range(self.size):
            rows.append(" ".join(symbols[self.grid[r][c]] for c in range(self.size)))
        return "\n".join(rows)
