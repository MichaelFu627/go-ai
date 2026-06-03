"""
Self-play: one game where the current network plays itself, using PUCT with
Dirichlet noise + temperature sampling. Returns the training data the game
produced.

For every move we record:
  - state:  the raw position info needed to re-encode features later
  - pi:     the MCTS visit-count distribution at the root (policy target)
  - player: who was to move at this position
After the game ends, the result z is filled in for each record:
  - z = +1 if the player at that record won
  - z = -1 if they lost
  - z =  0 on draws (rare with komi 7.5)

This z-from-mover's-perspective convention matches what the value head outputs
(value is from current-player's POV), so training is straightforward.

We store the RAW board + side_to_move + last moves, not pre-computed feature
planes — features can be regenerated cheaply when training (and storing raw
is ~50x smaller, which matters when we accumulate thousands of positions).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from board import Board, BLACK, WHITE, EMPTY, opponent, IllegalMove
from game import Game
from features import encode, policy_size, NUM_PLANES


@dataclass
class TrainingSample:
    """One position from a self-play game.

    We keep the raw board state and re-encode features at training time. This
    keeps the saved data small (a 9x9 board is just 81 small ints)."""
    board_grid: list      # 2D list of ints (0/1/2)
    to_move: int          # BLACK or WHITE
    last_move: Optional[tuple]
    opp_last_move: Optional[tuple]
    pi: np.ndarray        # policy target, shape (policy_size,)
    z: float = 0.0        # filled in after the game ends

    def encode_features(self) -> np.ndarray:
        """Reconstruct the network input planes from the stored raw state."""
        b = Board(size=len(self.board_grid))
        b.grid = [row[:] for row in self.board_grid]
        return encode(b, self.to_move, self.last_move, self.opp_last_move)


def play_one_game(net, board_size: int = 9, komi: float = 7.5,
                  simulations: int = 100,
                  temperature: float = 1.0,
                  temperature_drop_move: int = 16,
                  c_puct: float = 1.5,
                  add_dirichlet: bool = True,
                  dirichlet_alpha: float = 0.5,
                  device: str = "cpu",
                  max_moves: int = 200,
                  seed: Optional[int] = None,
                  verbose: bool = False,
                  save_kifu_to=None,
                  kifu_iteration: Optional[int] = None,
                  kifu_game_idx: Optional[int] = None,
                  ) -> tuple[list[TrainingSample], int, dict]:
    """Play one self-play game and return its training samples.

    Returns:
      samples: list of TrainingSample, z already filled
      winner:  BLACK / WHITE
      info:    dict with 'moves', 'reason' (terminal cause), 'duration_s'

    If save_kifu_to is set (a path-like to a directory), the completed game
    is also written there as a JSON kifu for later review in the UI.
    """
    import time
    from ai.puct_ai import PUCTPlayer

    t0 = time.time()
    g = Game(size=board_size, komi=komi, ai_color=None)
    samples: list[TrainingSample] = []

    # Two PUCT players sharing the same network (this is just self-play).
    # We use a single player object and reset temperature mid-game (paper:
    # exploration via temperature=1 for the first ~30 moves on 19x19, then
    # ~0 — we use a shorter window on 9x9 since games are shorter).
    player = PUCTPlayer(
        net=net,
        simulations=simulations,
        c_puct=c_puct,
        device=device,
        add_dirichlet=add_dirichlet,
        dirichlet_alpha=dirichlet_alpha,
        temperature=temperature,
        seed=seed,
    )

    move_num = 0
    reason = "two_passes"
    while not g.is_over and move_num < max_moves:
        # Drop temperature after the early exploration phase: greedy from now on.
        if move_num == temperature_drop_move:
            player.temperature = 0.0

        # Build sample BEFORE playing — record the position as it was when the
        # search ran.
        last_mv = None
        opp_last = None
        plays = [m for m in g.history if m.kind == "play"]
        if plays:
            last_mv = (plays[-1].row, plays[-1].col)
        if len(plays) >= 2:
            opp_last = (plays[-2].row, plays[-2].col)

        pi, move, wr = player.search_distribution(g)

        samples.append(TrainingSample(
            board_grid=[row[:] for row in g.board.grid],
            to_move=g.turn,
            last_move=last_mv,
            opp_last_move=opp_last,
            pi=pi.astype(np.float32),
        ))

        # Play the chosen move (or pass).
        try:
            if move is None:
                g.play_pass()
            else:
                g.play(move[0], move[1])
        except IllegalMove as e:
            # Should not happen in normal play (PUCT only suggests legal moves),
            # but be safe: pass and let the game end.
            if verbose:
                print(f"  illegal at move {move_num}: {e} -> passing")
            try: g.play_pass()
            except IllegalMove: break

        if verbose and move_num < 5:
            label = "pass" if move is None else g.coord_label(move[0], move[1])
            print(f"  move {move_num+1}: {('B' if g.turn==WHITE else 'W')} chose {label}  wr={wr:.2f}")

        move_num += 1

    if move_num >= max_moves:
        reason = "move_cap"

    # Score the final position.
    score = g.area_score()
    winner = BLACK if score["winner"] == "black" else WHITE

    # Fill in z for each sample from the sample's own player POV.
    for s in samples:
        if winner == s.to_move:
            s.z = 1.0
        else:
            s.z = -1.0

    info = {
        "moves": move_num,
        "reason": reason,
        "duration_s": time.time() - t0,
        "winner": "black" if winner == BLACK else "white",
        "score_black": score["black"],
        "score_white": score["white"],
    }

    if save_kifu_to is not None:
        from kifu import save_game
        try:
            path = save_game(
                g, save_kifu_to,
                iteration=kifu_iteration,
                game_idx=kifu_game_idx,
                simulations=simulations,
                source="selfplay",
                extra={"duration_s": info["duration_s"]},
            )
            info["kifu_path"] = str(path)
        except Exception as e:
            if verbose:
                print(f"  failed to save kifu: {e}")

    return samples, winner, info
