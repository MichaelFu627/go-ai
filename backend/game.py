"""
Game state: wraps Board with move history, turn tracking, captures count,
and two-consecutive-pass game end detection.

A Game holds:
- A Board (which itself enforces rules).
- Move history (for undo and replay).
- Capture counters per color.
- Pass state (for game-end detection).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Literal
from board import Board, BLACK, WHITE, EMPTY, opponent, IllegalMove


MoveType = Literal["play", "pass", "resign"]


@dataclass
class Move:
    color: int
    kind: MoveType
    row: Optional[int] = None
    col: Optional[int] = None
    captured: int = 0


@dataclass
class GameSnapshot:
    """Lightweight state for undo. We snapshot the whole board to keep undo
    correct in the face of captures and ko (which is much simpler than trying
    to reverse-engineer a single move)."""
    grid: list[list[int]]
    history_hashes: set[int]
    board_hash: int
    turn: int
    captures: dict[int, int]
    consecutive_passes: int


class Game:
    """Top-level game state used by the API layer."""

    def __init__(self, size: int = 9, komi: float = 7.5,
                 ai_color: Optional[int] = WHITE):
        self.board = Board(size)
        self.size = size
        self.komi = komi
        self.ai_color = ai_color  # None means human vs human

        self.turn: int = BLACK  # Black always plays first
        self.captures: dict[int, int] = {BLACK: 0, WHITE: 0}
        self.consecutive_passes: int = 0
        self.history: list[Move] = []
        self._snapshots: list[GameSnapshot] = []
        self.resigned_by: Optional[int] = None

    # ---- state queries ----------------------------------------------------

    @property
    def is_over(self) -> bool:
        return self.consecutive_passes >= 2 or self.resigned_by is not None

    @property
    def human_color(self) -> Optional[int]:
        if self.ai_color is None:
            return None
        return opponent(self.ai_color)

    def coord_label(self, r: int, c: int) -> str:
        """Convert (row, col) to a Go-style label like 'D4'.
        Columns A-J skipping I (Go convention). Rows 1 at the bottom."""
        cols = "ABCDEFGHJKLMNOPQRSTUVWXYZ"
        return f"{cols[c]}{self.size - r}"

    # ---- actions ----------------------------------------------------------

    def _snapshot(self) -> None:
        self._snapshots.append(GameSnapshot(
            grid=[row[:] for row in self.board.grid],
            history_hashes=set(self.board._history_hashes),
            board_hash=self.board._hash,
            turn=self.turn,
            captures=dict(self.captures),
            consecutive_passes=self.consecutive_passes,
        ))

    def play(self, r: int, c: int, color: Optional[int] = None) -> Move:
        if self.is_over:
            raise IllegalMove("game is over")
        if color is None:
            color = self.turn
        if color != self.turn:
            raise IllegalMove(f"not {color}'s turn")

        self._snapshot()
        try:
            result = self.board.play(r, c, color)
        except IllegalMove:
            self._snapshots.pop()  # discard snapshot on failure
            raise

        self.captures[color] += result.captured
        move = Move(color=color, kind="play", row=r, col=c,
                    captured=result.captured)
        self.history.append(move)
        self.consecutive_passes = 0
        self.turn = opponent(color)
        return move

    def play_pass(self, color: Optional[int] = None) -> Move:
        if self.is_over:
            raise IllegalMove("game is over")
        if color is None:
            color = self.turn
        if color != self.turn:
            raise IllegalMove(f"not {color}'s turn")
        self._snapshot()
        move = Move(color=color, kind="pass")
        self.history.append(move)
        self.consecutive_passes += 1
        self.turn = opponent(color)
        return move

    def resign(self, color: Optional[int] = None) -> Move:
        if self.is_over:
            raise IllegalMove("game is already over")
        if color is None:
            color = self.turn
        self._snapshot()
        move = Move(color=color, kind="resign")
        self.history.append(move)
        self.resigned_by = color
        return move

    def undo(self) -> Optional[Move]:
        """Undo the last move. Returns the move that was undone, or None."""
        if not self.history or not self._snapshots:
            return None
        snap = self._snapshots.pop()
        last = self.history.pop()
        self.board.grid = snap.grid
        self.board._history_hashes = snap.history_hashes
        self.board._hash = snap.board_hash
        self.turn = snap.turn
        self.captures = snap.captures
        self.consecutive_passes = snap.consecutive_passes
        self.resigned_by = None
        return last

    # ---- scoring ----------------------------------------------------------

    def area_score(self) -> dict:
        """Chinese-rules area scoring with per-point ownership.

        Returns stones + surrounded territory for each color, plus komi for
        white. Also returns an `ownership` grid (same shape as the board) where
        each cell is 'black', 'white', or 'neutral' (dame / contested).

        IMPORTANT: this does NOT do dead-stone removal. It scores the board
        exactly as it stands. While a game is in progress (or if dead stones
        remain on the board at the end), the count reflects the literal
        position, not the "agreed" result. We surface this via `is_estimate`
        and `dame` so the UI can be honest about it.

        Dead-stone detection is genuinely hard (it needs life/death reading);
        we'll revisit it once the MCTS + value network can judge groups.
        """
        size = self.size
        grid = self.board.grid
        visited = [[False] * size for _ in range(size)]

        # ownership grid: 0 neutral, 1 black, 2 white
        ownership = [[EMPTY] * size for _ in range(size)]

        black_stones = 0
        white_stones = 0
        black_territory = 0
        white_territory = 0
        dame = 0  # neutral empty points (border between both colors)

        # Count stones and mark their ownership.
        for r in range(size):
            for c in range(size):
                if grid[r][c] == BLACK:
                    black_stones += 1
                    ownership[r][c] = BLACK
                elif grid[r][c] == WHITE:
                    white_stones += 1
                    ownership[r][c] = WHITE

        # Flood-fill each empty region; assign to whichever single color
        # borders it, else it's dame (neutral).
        for r in range(size):
            for c in range(size):
                if grid[r][c] != EMPTY or visited[r][c]:
                    continue
                region: list[tuple[int, int]] = []
                touched: set[int] = set()
                stack = [(r, c)]
                while stack:
                    cr, cc = stack.pop()
                    if visited[cr][cc]:
                        continue
                    visited[cr][cc] = True
                    region.append((cr, cc))
                    for nr, nc in self.board.neighbors(cr, cc):
                        v = grid[nr][nc]
                        if v == EMPTY and not visited[nr][nc]:
                            stack.append((nr, nc))
                        elif v != EMPTY:
                            touched.add(v)
                if touched == {BLACK}:
                    black_territory += len(region)
                    for (rr, cc) in region:
                        ownership[rr][cc] = BLACK
                elif touched == {WHITE}:
                    white_territory += len(region)
                    for (rr, cc) in region:
                        ownership[rr][cc] = WHITE
                else:
                    dame += len(region)
                    # ownership stays EMPTY (neutral) for these points

        black_total = float(black_stones + black_territory)
        white_total = float(white_stones + white_territory) + self.komi

        # We treat it as an "estimate" (rather than a final result) whenever
        # the game isn't formally over, or when dame remain (a sign the
        # boundaries aren't settled / dead stones may still be on the board).
        is_estimate = (not self.is_over) or dame > 0

        ownership_str = [
            ["black" if v == BLACK else "white" if v == WHITE else "neutral"
             for v in row]
            for row in ownership
        ]

        return {
            "black": black_total,
            "white": white_total,
            "black_stones": black_stones,
            "white_stones": white_stones,
            "black_territory": black_territory,
            "white_territory": white_territory,
            "komi": self.komi,
            "dame": dame,
            "winner": "black" if black_total > white_total else "white",
            "margin": abs(black_total - white_total),
            "ownership": ownership_str,
            "is_estimate": is_estimate,
            "is_over": self.is_over,
        }
