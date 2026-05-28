"""
PUCT Monte-Carlo Tree Search, guided by the neural network.

This is the AlphaGo / AlphaGo Zero search. It differs from the pure-MCTS in
mcts_ai.py in three ways:

  1. Selection uses PUCT (paper equation 5):
         a* = argmax[ Q(s,a) + c_puct * P(s,a) * sqrt(ΣN) / (1 + N(s,a)) ]
     The prior P comes from the policy network — it steers the search toward
     moves the network thinks are good, instead of exploring blindly.

  2. Leaf evaluation uses the VALUE NETWORK, not a random rollout. One forward
     pass gives a position's win estimate directly — fast and far less noisy
     than playing random games to the end.

  3. Expansion asks the network once for the whole policy distribution and
     stores it as the prior P on every child edge.

The node statistics and the four-phase loop are otherwise the same shape as
classical MCTS, so this file reads as a close cousin of mcts_ai.py.

Self-play (used for training) will reuse this search with two extra knobs:
Dirichlet root noise (for exploration) and a temperature on the final move
choice. Both are supported here via flags.
"""

from __future__ import annotations
import math
import numpy as np
from typing import Optional
from board import Board, BLACK, WHITE, EMPTY, opponent, IllegalMove
from game import Game
from features import encode, move_to_index, index_to_move, policy_size
from .base import AIDecision

PASS = None


def _is_obvious_eye(board: Board, r: int, c: int, color: int) -> bool:
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


def _legal_move_set(board: Board, color: int) -> list:
    """Sensible legal moves plus PASS, as a list of moves (None = pass)."""
    moves = [(r, c) for (r, c) in board.fast_legal_moves(color)
             if not _is_obvious_eye(board, r, c, color)]
    moves.append(PASS)
    return moves


class Edge:
    """One action edge from a node. Stores the PUCT statistics:
    P (prior from policy net), N (visit count), W (total value), Q (mean value).
    Q and W are from the perspective of the player to move at the PARENT node."""
    __slots__ = ("move", "P", "N", "W", "child")

    def __init__(self, move, prior: float):
        self.move = move
        self.P = prior
        self.N = 0
        self.W = 0.0
        self.child: Optional["PUCTNode"] = None

    @property
    def Q(self) -> float:
        return self.W / self.N if self.N > 0 else 0.0


class PUCTNode:
    """A node = a position. Holds one Edge per legal move."""
    __slots__ = ("to_move", "edges", "last_move", "is_expanded")

    def __init__(self, to_move: int, last_move):
        self.to_move = to_move
        self.last_move = last_move  # the move that led here (for feature encoding)
        self.edges: list[Edge] = []
        self.is_expanded = False


class PUCTPosition:
    """Lightweight position for search: board + side to move + pass count.
    Tracks last move and opponent's last move for feature encoding."""
    __slots__ = ("board", "to_move", "passes", "komi", "last_move", "opp_last_move")

    def __init__(self, board, to_move, passes, komi, last_move=None, opp_last_move=None):
        self.board = board
        self.to_move = to_move
        self.passes = passes
        self.komi = komi
        self.last_move = last_move
        self.opp_last_move = opp_last_move

    def copy(self):
        return PUCTPosition(self.board.copy(), self.to_move, self.passes,
                            self.komi, self.last_move, self.opp_last_move)

    @property
    def is_terminal(self) -> bool:
        return self.passes >= 2

    def play(self, move):
        if move is PASS:
            self.passes += 1
        else:
            self.board.play(move[0], move[1], self.to_move, check_superko=False)
            self.passes = 0
        # rotate last-move tracking
        self.opp_last_move = self.last_move
        self.last_move = move if move is not PASS else None
        self.to_move = opponent(self.to_move)

    def score_winner(self) -> int:
        """Chinese-rules area scoring for terminal positions."""
        size = self.board.size
        grid = self.board.grid
        visited = [[False] * size for _ in range(size)]
        black = white = 0
        for r in range(size):
            for c in range(size):
                if grid[r][c] == BLACK: black += 1
                elif grid[r][c] == WHITE: white += 1
        for r in range(size):
            for c in range(size):
                if grid[r][c] != EMPTY or visited[r][c]:
                    continue
                region = 0
                touched = set()
                stack = [(r, c)]
                while stack:
                    cr, cc = stack.pop()
                    if visited[cr][cc]: continue
                    visited[cr][cc] = True
                    region += 1
                    for nr, nc in self.board.neighbors(cr, cc):
                        v = grid[nr][nc]
                        if v == EMPTY and not visited[nr][nc]:
                            stack.append((nr, nc))
                        elif v != EMPTY:
                            touched.add(v)
                if touched == {BLACK}: black += region
                elif touched == {WHITE}: white += region
        return BLACK if black > white + self.komi else WHITE


class PUCTPlayer:
    """Neural-network-guided PUCT search. Drop-in for the AI interface."""

    def __init__(self, net, simulations: int = 400, c_puct: float = 1.5,
                 device: str = "cpu", add_dirichlet: bool = False,
                 dirichlet_alpha: float = 0.5, dirichlet_eps: float = 0.25,
                 temperature: float = 0.0, seed: Optional[int] = None):
        self.net = net
        self.simulations = simulations
        self.c_puct = c_puct
        self.device = device
        self.add_dirichlet = add_dirichlet
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.temperature = temperature
        self.rng = np.random.default_rng(seed)

    # ---- network evaluation of a position ----
    def _evaluate(self, pos: PUCTPosition):
        """Run the net on a position. Returns (priors_dict, value).
        priors_dict maps each legal move -> prior probability (renormalized
        over legal moves only). value is from pos.to_move's perspective."""
        planes = encode(pos.board, pos.to_move,
                        last_move=pos.last_move, opp_last_move=pos.opp_last_move)
        probs, value = self.net.predict(planes, device=self.device)
        size = pos.board.size

        legal = _legal_move_set(pos.board, pos.to_move)
        priors = {}
        total = 0.0
        for mv in legal:
            idx = move_to_index(mv, size)
            p = float(probs[idx])
            priors[mv] = p
            total += p
        # renormalize over legal moves (the net's mass on illegal moves is dropped)
        if total > 1e-8:
            for mv in priors:
                priors[mv] /= total
        else:
            # network gave ~0 to all legal moves; fall back to uniform
            u = 1.0 / len(legal)
            for mv in priors:
                priors[mv] = u
        return priors, value

    def _expand(self, node: PUCTNode, pos: PUCTPosition):
        """Create edges for all legal moves using network priors."""
        priors, value = self._evaluate(pos)
        for mv, p in priors.items():
            node.edges.append(Edge(mv, p))
        node.is_expanded = True
        return value

    def _add_root_noise(self, root: PUCTNode):
        """Mix Dirichlet noise into root priors (AlphaGo Zero exploration)."""
        n = len(root.edges)
        if n == 0:
            return
        noise = self.rng.dirichlet([self.dirichlet_alpha] * n)
        for edge, eta in zip(root.edges, noise):
            edge.P = (1 - self.dirichlet_eps) * edge.P + self.dirichlet_eps * eta

    def _select_edge(self, node: PUCTNode) -> Edge:
        """PUCT selection (paper equation 5)."""
        total_n = sum(e.N for e in node.edges)
        sqrt_total = math.sqrt(total_n) if total_n > 0 else 1.0
        best_edge = None
        best_score = -float("inf")
        for e in node.edges:
            u = self.c_puct * e.P * sqrt_total / (1 + e.N)
            score = e.Q + u
            if score > best_score:
                best_score = score
                best_edge = e
        return best_edge

    # ---- public API ----
    def select_move(self, game: Game) -> AIDecision:
        # find the human/previous last move for feature encoding
        last_mv = None
        opp_last = None
        plays = [m for m in game.history if m.kind == "play"]
        if plays:
            last_mv = (plays[-1].row, plays[-1].col)
        if len(plays) >= 2:
            opp_last = (plays[-2].row, plays[-2].col)

        root_pos = PUCTPosition(
            board=game.board.copy(), to_move=game.turn,
            passes=game.consecutive_passes, komi=game.komi,
            last_move=last_mv, opp_last_move=opp_last,
        )
        if root_pos.is_terminal:
            return AIDecision(kind="pass", simulations=0)

        root = PUCTNode(root_pos.to_move, last_move=last_mv)
        self._expand(root, root_pos)
        if self.add_dirichlet:
            self._add_root_noise(root)

        for _ in range(self.simulations):
            self._simulate(root, root_pos.copy())

        return self._decide(root, game)

    def _simulate(self, root: PUCTNode, pos: PUCTPosition):
        """One PUCT simulation: select → expand+evaluate → backup."""
        path = []  # list of (node, edge)
        node = root

        # 1. Selection: descend via PUCT until we reach an unexpanded node
        #    or a terminal position.
        while node.is_expanded and not pos.is_terminal:
            edge = self._select_edge(node)
            path.append((node, edge))
            pos.play(edge.move)
            if edge.child is None:
                # create the child node shell (expanded lazily below)
                edge.child = PUCTNode(pos.to_move, last_move=edge.move)
                node = edge.child
                break
            node = edge.child

        # 2. Evaluate the leaf.
        if pos.is_terminal:
            winner = pos.score_winner()
            # value from the perspective of the player to move at the leaf
            leaf_value = 1.0 if winner == pos.to_move else -1.0
        else:
            if not node.is_expanded:
                leaf_value = self._expand(node, pos)  # value from node.to_move POV
            else:
                # already expanded (transposition) — just evaluate
                _, leaf_value = self._evaluate(pos)

        # 3. Backup. leaf_value is from the perspective of pos.to_move (the
        #    player to move at the leaf). As we walk back up, the perspective
        #    flips every ply.
        value = leaf_value
        for node_i, edge_i in reversed(path):
            # edge_i.move was played by node_i.to_move. The value stored on the
            # edge should be from node_i.to_move's perspective. The leaf value
            # is from the leaf mover's POV; flip per ply distance.
            value = -value
            edge_i.N += 1
            edge_i.W += value

    def _decide(self, root: PUCTNode, game: Game) -> AIDecision:
        if not root.edges:
            return AIDecision(kind="pass", simulations=self.simulations)

        visits = np.array([e.N for e in root.edges], dtype=np.float64)
        if visits.sum() == 0:
            # no simulations resolved (shouldn't happen) — pick max prior
            best = max(root.edges, key=lambda e: e.P)
        elif self.temperature and self.temperature > 1e-3:
            # sample proportional to N^(1/temperature) — used in self-play
            probs = visits ** (1.0 / self.temperature)
            probs = probs / probs.sum()
            idx = self.rng.choice(len(root.edges), p=probs)
            best = root.edges[idx]
        else:
            best = max(root.edges, key=lambda e: e.N)

        # winrate from root mover's perspective: best.Q is already from the
        # root mover's POV (it's the parent's perspective).
        winrate = (best.Q + 1.0) / 2.0  # map [-1,1] → [0,1]

        ranked = sorted(root.edges, key=lambda e: e.N, reverse=True)
        top_moves = []
        for e in ranked[:6]:
            if e.move is PASS:
                top_moves.append({"row": None, "col": None, "pass": True,
                                  "visits": e.N, "q": round((e.Q + 1) / 2, 3)})
            else:
                top_moves.append({"row": e.move[0], "col": e.move[1],
                                  "visits": e.N, "q": round((e.Q + 1) / 2, 3)})

        # principal variation: follow most-visited edges a few plies
        pv = []
        node = root
        for _ in range(8):
            if not node.edges:
                break
            e = max(node.edges, key=lambda x: x.N)
            if e.N == 0:
                break
            if e.move is PASS:
                pv.append({"pass": True})
            else:
                pv.append({"row": e.move[0], "col": e.move[1]})
            node = e.child
            if node is None:
                break

        if winrate < 0.10 and best.N > 30:
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
