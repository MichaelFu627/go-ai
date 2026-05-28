"""Tests for features.py and net.py. Run: python -m pytest test_net.py -v"""
import numpy as np
import torch
from board import Board, BLACK, WHITE
from features import (encode, NUM_PLANES, move_to_index, index_to_move,
                      policy_size)
from net import GoNet, count_parameters


def test_encode_shape():
    b = Board(9)
    planes = encode(b, BLACK)
    assert planes.shape == (NUM_PLANES, 9, 9)
    assert planes.dtype == np.float32


def test_encode_empty_board():
    b = Board(9)
    planes = encode(b, BLACK)
    # On an empty board every point is "empty" (plane 2 all ones).
    assert planes[2].sum() == 81
    # No stones.
    assert planes[0].sum() == 0
    assert planes[1].sum() == 0
    # Constant ones plane.
    assert planes[14].sum() == 81
    # Black to move → plane 15 all ones.
    assert planes[15].sum() == 81


def test_encode_relative_to_mover():
    b = Board(9)
    b.play(4, 4, BLACK)
    # From black's perspective, the stone is "mine".
    pb = encode(b, BLACK)
    assert pb[0, 4, 4] == 1.0   # my stone
    assert pb[1, 4, 4] == 0.0
    # From white's perspective, the same stone is "opponent's".
    pw = encode(b, WHITE)
    assert pw[0, 4, 4] == 0.0
    assert pw[1, 4, 4] == 1.0   # opponent stone
    # White to move → colour plane is 0.
    assert pw[15].sum() == 0


def test_encode_liberties():
    b = Board(9)
    b.play(4, 4, BLACK)  # a lone stone in the center has 4 liberties
    planes = encode(b, BLACK)
    # liberty bucket for >=4 is plane index 3+3 = 6
    assert planes[6, 4, 4] == 1.0
    # Now reduce its liberties.
    b.play(3, 4, WHITE)
    b.play(5, 4, WHITE)
    planes = encode(b, BLACK)
    # black stone now has 2 liberties → bucket index 1 → plane 3+1 = 4
    assert planes[4, 4, 4] == 1.0


def test_move_index_roundtrip():
    size = 9
    for r in range(size):
        for c in range(size):
            idx = move_to_index((r, c), size)
            assert index_to_move(idx, size) == (r, c)
    # pass
    pass_idx = move_to_index(None, size)
    assert pass_idx == 81
    assert index_to_move(pass_idx, size) is None


def test_network_forward():
    net = GoNet(board_size=9, channels=64, num_res_blocks=5)
    x = torch.randn(4, NUM_PLANES, 9, 9)  # batch of 4
    logits, value = net(x)
    assert logits.shape == (4, policy_size(9))  # (4, 82)
    assert value.shape == (4, 1)
    # value is tanh → within [-1, 1]
    assert value.min() >= -1.0 and value.max() <= 1.0


def test_network_predict_single():
    net = GoNet(board_size=9)
    b = Board(9)
    b.play(4, 4, BLACK)
    planes = encode(b, WHITE)
    probs, value = net.predict(planes)
    assert probs.shape == (policy_size(9),)
    # softmax → sums to ~1
    assert abs(probs.sum() - 1.0) < 1e-5
    assert -1.0 <= value <= 1.0


def test_parameter_count_reasonable():
    net = GoNet(board_size=9, channels=64, num_res_blocks=5)
    n = count_parameters(net)
    # Sanity: a small 9x9 net should be in the hundreds of thousands to low
    # millions of parameters, not tiny and not huge.
    assert 100_000 < n < 5_000_000
    print(f"\nGoNet parameters: {n:,}")


if __name__ == "__main__":
    import sys
    import pytest
    pytest.main([__file__, "-v", "-s"] + sys.argv[1:])
