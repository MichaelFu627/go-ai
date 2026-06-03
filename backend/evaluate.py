"""
Evaluation: play games between two networks (e.g. the freshly-trained one vs
the previous best) and report the win rate of the first.

Uses PUCT with no Dirichlet noise and temperature 0 — i.e. play deterministically
strong, since the goal is to judge actual strength, not generate diverse data.
"""

from __future__ import annotations
from typing import Optional
from game import Game
from board import BLACK, WHITE, opponent, IllegalMove
from ai.puct_ai import PUCTPlayer


def play_match(net_a, net_b, num_games: int = 20, board_size: int = 9,
               komi: float = 7.5, simulations: int = 100,
               device: str = "cpu", max_moves: int = 200,
               seed: Optional[int] = None, verbose: bool = False) -> dict:
    """Play num_games games, alternating colors. Returns win/loss/draw counts.

    Result is from net_a's perspective: a_wins / b_wins / draws.
    """
    import random
    rng = random.Random(seed)

    a_wins = 0
    b_wins = 0
    draws = 0

    for i in range(num_games):
        # Alternate who plays black to neutralize first-move advantage.
        a_is_black = (i % 2 == 0)
        black_net = net_a if a_is_black else net_b
        white_net = net_b if a_is_black else net_a

        # Fresh PUCT players (no Dirichlet, greedy — true strength).
        black_player = PUCTPlayer(black_net, simulations=simulations,
                                  device=device, add_dirichlet=False,
                                  temperature=0.0, seed=rng.randrange(10**9))
        white_player = PUCTPlayer(white_net, simulations=simulations,
                                  device=device, add_dirichlet=False,
                                  temperature=0.0, seed=rng.randrange(10**9))

        g = Game(size=board_size, komi=komi, ai_color=None)
        moves = 0
        while not g.is_over and moves < max_moves:
            player = black_player if g.turn == BLACK else white_player
            d = player.select_move(g)
            try:
                if d.kind == "play":
                    g.play(d.row, d.col)
                elif d.kind == "pass":
                    g.play_pass()
                elif d.kind == "resign":
                    g.resign()
                    break
            except IllegalMove:
                g.play_pass()
            moves += 1

        # Determine winner.
        if g.resigned_by is not None:
            winner = opponent(g.resigned_by)
        else:
            sc = g.area_score()
            winner = BLACK if sc["winner"] == "black" else WHITE

        a_won = (winner == BLACK) == a_is_black
        if a_won:
            a_wins += 1
        else:
            b_wins += 1

        if verbose:
            print(f"  game {i+1}: A as {'B' if a_is_black else 'W'}, "
                  f"{moves} moves, {'A' if a_won else 'B'} wins")

    return {
        "a_wins": a_wins,
        "b_wins": b_wins,
        "draws": draws,
        "games": num_games,
        "win_rate": a_wins / num_games if num_games > 0 else 0.0,
    }
