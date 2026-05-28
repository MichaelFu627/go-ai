"""Tests for PUCT search (ai/puct_ai.py).

We use small fake 'oracle' networks instead of the real net so the tests are
fast and deterministic, and so they check the SEARCH logic independently of
whether the real network is trained.

Run: python -m pytest test_puct.py -v
"""
import numpy as np
from game import Game
from board import BLACK, WHITE
from ai.puct_ai import PUCTPlayer
from features import move_to_index


class FavorNet:
    """Assigns almost all policy mass to one favored move; neutral value."""
    def __init__(self, size, favored):
        self.size = size
        self.favored = favored

    def predict(self, planes, device="cpu"):
        n = self.size * self.size + 1
        probs = np.ones(n) * (0.01 / n)
        probs[move_to_index(self.favored, self.size)] = 0.99
        probs = probs / probs.sum()
        return probs, 0.0


class CountNet:
    """Value = +1 if side-to-move has more stones, else -1. Uniform policy."""
    def __init__(self, size):
        self.size = size

    def predict(self, planes, device="cpu"):
        n = self.size * self.size + 1
        probs = np.ones(n) / n
        my = planes[0].sum()
        opp = planes[1].sum()
        value = 1.0 if my > opp else (-1.0 if my < opp else 0.0)
        return probs, value


def test_puct_runs_and_returns_valid_move():
    from net import GoNet
    net = GoNet(board_size=9)
    g = Game(size=9, komi=7.5)
    g.play(4, 4)
    d = PUCTPlayer(net, simulations=20, seed=0).select_move(g)
    assert d.kind in ("play", "pass", "resign")
    assert d.simulations == 20
    if d.kind == "play":
        assert 0 <= d.row < 9 and 0 <= d.col < 9


def test_puct_concentrates_on_high_prior():
    """With a strong policy prior on one move, PUCT should pick that move."""
    g = Game(size=9, komi=7.5)
    net = FavorNet(9, (4, 4))
    d = PUCTPlayer(net, simulations=100, c_puct=1.5, seed=0).select_move(g)
    assert (d.row, d.col) == (4, 4)
    # nearly all visits should land on the favored move
    top = d.top_moves[0]
    assert top["row"] == 4 and top["col"] == 4
    assert top["visits"] >= 90


def test_puct_backup_sign():
    """The dominant side should get a high winrate (backup signs correct)."""
    g = Game(size=9, komi=7.5, ai_color=None)
    for c in range(9):
        for r in range(6):
            try:
                g.play(r, c, BLACK)
            except Exception:
                pass
            try:
                g.play_pass(WHITE)
            except Exception:
                pass
    g.turn = BLACK
    d = PUCTPlayer(CountNet(9), simulations=60, seed=1).select_move(g)
    assert d.winrate > 0.5


def test_puct_winrate_in_range():
    from net import GoNet
    net = GoNet(board_size=9)
    g = Game(size=9, komi=7.5)
    d = PUCTPlayer(net, simulations=20, seed=2).select_move(g)
    assert 0.0 <= d.winrate <= 1.0


def test_dirichlet_noise_runs():
    """Self-play mode: root noise + temperature should not crash."""
    from net import GoNet
    net = GoNet(board_size=9)
    g = Game(size=9, komi=7.5)
    player = PUCTPlayer(net, simulations=30, add_dirichlet=True,
                        temperature=1.0, seed=3)
    d = player.select_move(g)
    assert d.kind in ("play", "pass", "resign")


if __name__ == "__main__":
    import sys, pytest
    pytest.main([__file__, "-v"] + sys.argv[1:])
