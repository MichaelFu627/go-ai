"""
Pure Monte-Carlo Tree Search (UCT) with random rollouts.

This is the classical MCTS that strong pre-AlphaGo Go programs used (Pachi,
Fuego, etc. — see the paper's references). It needs no training and no neural
network: it estimates a move's value by playing many fast random games to the
end and averaging the outcomes.

The four phases per simulation (paper Figure 3):
  1. Selection   — descend the tree by UCB1 until reaching a not-fully-expanded
                   node or a terminal position.
  2. Expansion   — add one new child for an untried move.
  3. Rollout     — play random (eye-avoiding) moves to the end of the game.
  4. Backup      — propagate the win/loss up the path.

When we later add a policy/value network, this file's `Node` and search loop
stay almost the same — we'll swap UCB1 for PUCT (adding a prior P) and replace
the random rollout with a value-network evaluation. The AIDecision interface
to the rest of the app does not change at all.
"""

from __future__ import annotations
import math
import random
from typing import Optional
from board import Board, BLACK, WHITE, EMPTY, opponent, IllegalMove
from game import Game
from .base import AIDecision


# A move is either a board point (r, c) or the sentinel PASS.
PASS = None


def _is_obvious_eye(board: Board, r: int, c: int, color: int) -> bool:
    """Same 'don't fill your own eye' heuristic as the random AI, but operating
    directly on a Board (not a Game) so rollouts stay cheap."""
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


def _sensible_moves(board: Board, color: int) -> list[tuple[int, int]]:
    """Legal moves that aren't obvious self-eye-fills. Uses the fast (no-copy)
    legality check since this runs inside the search hot loop."""
    out = []
    for (r, c) in board.fast_legal_moves(color):
        if not _is_obvious_eye(board, r, c, color):
            out.append((r, c))
    return out


class _Position:
    """Lightweight game position for use inside the search — just a board,
    the side to move, and a pass counter. Much cheaper than a full Game.

    Coordinates here are board (row, col). A move of PASS increments the pass
    counter; two passes end the position.
    """
    __slots__ = ("board", "to_move", "passes", "komi")

    def __init__(self, board: Board, to_move: int, passes: int, komi: float):
        self.board = board
        self.to_move = to_move
        self.passes = passes
        self.komi = komi

    def copy(self) -> "_Position":
        return _Position(self.board.copy(), self.to_move, self.passes, self.komi)

    @property
    def is_terminal(self) -> bool:
        return self.passes >= 2

    def play(self, move) -> None:
        """Apply a move in place. move is (r, c) or PASS.
        Superko is skipped — these are throwaway search positions."""
        if move is PASS:
            self.passes += 1
        else:
            self.board.play(move[0], move[1], self.to_move, check_superko=False)
            self.passes = 0
        self.to_move = opponent(self.to_move)

    def winner(self) -> int:
        """Score the terminal position by Chinese-rules area scoring.
        Returns BLACK or WHITE. (No dead-stone removal — random rollouts
        play everything out, so most groups are resolved naturally.)"""
        size = self.board.size
        grid = self.board.grid
        visited = [[False] * size for _ in range(size)]
        black = 0
        white = 0
        for r in range(size):
            for c in range(size):
                if grid[r][c] == BLACK:
                    black += 1
                elif grid[r][c] == WHITE:
                    white += 1
        for r in range(size):
            for c in range(size):
                if grid[r][c] != EMPTY or visited[r][c]:
                    continue
                region = 0
                touched = set()
                stack = [(r, c)]
                while stack:
                    cr, cc = stack.pop()
                    if visited[cr][cc]:
                        continue
                    visited[cr][cc] = True
                    region += 1
                    for nr, nc in self.board.neighbors(cr, cc):
                        v = grid[nr][nc]
                        if v == EMPTY and not visited[nr][nc]:
                            stack.append((nr, nc))
                        elif v != EMPTY:
                            touched.add(v)
                if touched == {BLACK}:
                    black += region
                elif touched == {WHITE}:
                    white += region
        return BLACK if black > white + self.komi else WHITE


class Node:
    """A node in the search tree. Stores statistics from the perspective of
    the player who is ABOUT TO MOVE at this node's parent — i.e. Q is the
    value of reaching this node for the parent's mover.

    We keep it simpler: each node stores wins/visits from the perspective of
    the player to move at THAT node, and flip during backup.
    """
    __slots__ = ("move", "parent", "children", "untried", "to_move",
                 "wins", "visits")

    def __init__(self, position: _Position, move=None, parent: "Node" = None):
        self.move = move              # the move that led to this node
        self.parent = parent
        self.children: list["Node"] = []
        self.to_move = position.to_move
        # Moves we can still expand from here. Include PASS so the engine can
        # choose to pass (important near the end of the game).
        self.untried = _sensible_moves(position.board, position.to_move)
        self.untried.append(PASS)
        random.shuffle(self.untried)
        self.wins = 0.0   # wins from the perspective of the player who just moved
        self.visits = 0

    def is_fully_expanded(self) -> bool:
        return len(self.untried) == 0

    def ucb1(self, c_param: float) -> float:
        """UCB1 score for selecting this child from its parent."""
        if self.visits == 0:
            return float("inf")
        exploit = self.wins / self.visits
        explore = c_param * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploit + explore

    def best_child(self, c_param: float) -> "Node":
        return max(self.children, key=lambda ch: ch.ucb1(c_param))


class MCTSAI:
    """UCT search with random rollouts. Drop-in for RandomAI."""

    def __init__(self, simulations: int = 800, c_param: float = 1.4,
                 max_rollout: Optional[int] = None, seed: Optional[int] = None):
        self.simulations = simulations
        self.c_param = c_param
        # Cap rollout length so a pathological random game can't loop forever.
        self.max_rollout = max_rollout
        self.rng = random.Random(seed)
        random.seed(seed)  # _sensible_moves shuffles via the global rng

    # ---- public API ----
    def select_move(self, game: Game) -> AIDecision:
        root_pos = _Position(
            board=game.board.copy(),
            to_move=game.turn,
            passes=game.consecutive_passes,
            komi=game.komi,
        )
        if root_pos.is_terminal:
            return AIDecision(kind="pass", simulations=0)

        root = Node(root_pos)

        for _ in range(self.simulations):
            node = root
            pos = root_pos.copy()

            # 1. Selection
            while node.is_fully_expanded() and node.children:
                node = node.best_child(self.c_param)
                pos.play(node.move)

            # 2. Expansion
            if not pos.is_terminal and node.untried:
                move = node.untried.pop()
                # Defensive: a move could be illegal due to superko captured
                # in this line; skip if so.
                try:
                    pos.play(move)
                except IllegalMove:
                    continue
                child = Node(pos, move=move, parent=node)
                node.children.append(child)
                node = child

            # 3. Rollout
            winner = self._rollout(pos)

            # 4. Backup
            self._backup(node, winner)

        return self._decide(root, game)

    # ---- internals ----
    def _rollout(self, pos: _Position) -> int:
        """Play random eye-avoiding moves to the end; return the winner.

        Optimization: instead of enumerating ALL legal moves every step
        (expensive — O(board) liberty BFS per step), we keep a shuffled list
        of empty points and try them until one is playable. This is much
        faster in the rollout hot loop.
        """
        sim = pos.copy()
        board = sim.board
        size = board.size
        cap = self.max_rollout or (size * size * 2)
        steps = 0

        while not sim.is_terminal and steps < cap:
            color = sim.to_move
            # Collect current empty points (cheap) and shuffle.
            empties = [(r, c) for r in range(size) for c in range(size)
                       if board.grid[r][c] == EMPTY]
            self.rng.shuffle(empties)
            played = False
            for (r, c) in empties:
                if _is_obvious_eye(board, r, c, color):
                    continue
                if not board.fast_legal(r, c, color):
                    continue
                sim.play((r, c))
                played = True
                break
            if not played:
                sim.play(PASS)
            steps += 1
        return sim.winner()

    def _backup(self, node: Node, winner: int) -> None:
        """Propagate the result up to the root, flipping perspective each level.

        A node's `wins` counts wins for the player who just moved to reach it,
        which is the OPPONENT of node.to_move."""
        while node is not None:
            node.visits += 1
            mover = opponent(node.to_move)  # who moved into this node
            if mover == winner:
                node.wins += 1.0
            node = node.parent

    def _decide(self, root: Node, game: Game) -> AIDecision:
        if not root.children:
            return AIDecision(kind="pass", simulations=self.simulations)

        # Most-visited child is the robust choice (paper: less sensitive to
        # outliers than max value).
        best = max(root.children, key=lambda ch: ch.visits)

        # Winrate from the root mover's perspective.
        # best.wins is from the perspective of the root mover (who moved into
        # best), so winrate is directly best.wins / best.visits.
        winrate = best.wins / best.visits if best.visits else 0.0

        # Build top-move introspection for the UI.
        ranked = sorted(root.children, key=lambda ch: ch.visits, reverse=True)
        top_moves = []
        for ch in ranked[:6]:
            if ch.move is PASS:
                top_moves.append({"row": None, "col": None, "pass": True,
                                  "visits": ch.visits,
                                  "q": round(ch.wins / ch.visits, 3) if ch.visits else 0})
            else:
                top_moves.append({"row": ch.move[0], "col": ch.move[1],
                                  "visits": ch.visits,
                                  "q": round(ch.wins / ch.visits, 3) if ch.visits else 0})

        # Principal variation: follow most-visited children a few plies deep.
        pv = []
        node = root
        for _ in range(8):
            if not node.children:
                break
            node = max(node.children, key=lambda ch: ch.visits)
            if node.move is PASS:
                pv.append({"pass": True})
            else:
                pv.append({"row": node.move[0], "col": node.move[1]})

        # Resign if it looks hopeless (paper resigns below ~10% win prob).
        if winrate < 0.10 and best.visits > 20:
            return AIDecision(kind="resign", winrate=winrate,
                              simulations=self.simulations,
                              top_moves=top_moves, principal_variation=pv)

        if best.move is PASS:
            return AIDecision(kind="pass", winrate=winrate,
                              simulations=self.simulations,
                              top_moves=top_moves, principal_variation=pv)

        return AIDecision(kind="play", row=best.move[0], col=best.move[1],
                          winrate=winrate, simulations=self.simulations,
                          top_moves=top_moves, principal_variation=pv)
