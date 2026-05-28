"""Tests for board rules. Run with: python -m pytest test_board.py -v"""
import pytest
from board import Board, BLACK, WHITE, EMPTY, IllegalMove


def test_basic_placement():
    b = Board(9)
    b.play(4, 4, BLACK)
    assert b[(4, 4)] == BLACK
    b.play(4, 5, WHITE)
    assert b[(4, 5)] == WHITE


def test_cannot_play_on_occupied():
    b = Board(9)
    b.play(4, 4, BLACK)
    with pytest.raises(IllegalMove, match="occupied"):
        b.play(4, 4, WHITE)


def test_simple_capture():
    """Black surrounds a single white stone — white should be removed."""
    b = Board(9)
    # White at center, surrounded by black on 3 sides
    b.play(4, 4, WHITE)
    b.play(3, 4, BLACK)
    b.play(5, 4, BLACK)
    b.play(4, 3, BLACK)
    # White still has one liberty (4,5)
    assert b[(4, 4)] == WHITE
    # Closing the last liberty captures
    result = b.play(4, 5, BLACK)
    assert result.captured == 1
    assert b[(4, 4)] == EMPTY


def test_group_capture():
    """Capture a whole connected group at once."""
    b = Board(9)
    # White group of 2 stones
    b.play(4, 4, WHITE)
    b.play(4, 5, WHITE)
    # Surround it
    b.play(3, 4, BLACK)
    b.play(3, 5, BLACK)
    b.play(5, 4, BLACK)
    b.play(5, 5, BLACK)
    b.play(4, 3, BLACK)
    # Now (4,6) is the last liberty
    result = b.play(4, 6, BLACK)
    assert result.captured == 2
    assert b[(4, 4)] == EMPTY
    assert b[(4, 5)] == EMPTY


def test_suicide_forbidden():
    """Playing into a point with no liberty (and no capture) is illegal."""
    b = Board(9)
    # Build a black ring around (4,4)
    b.play(3, 4, BLACK)
    b.play(5, 4, BLACK)
    b.play(4, 3, BLACK)
    b.play(4, 5, BLACK)
    # White plays into the trap — suicide
    with pytest.raises(IllegalMove, match="suicide"):
        b.play(4, 4, WHITE)
    # Verify board unchanged
    assert b[(4, 4)] == EMPTY


def test_capture_is_not_suicide():
    """Playing a 'suicide-looking' move that actually captures is legal."""
    b = Board(9)
    # White stones at (4,4) and (4,5), with one shared liberty at (4,6)
    b.play(4, 4, WHITE)
    b.play(4, 5, WHITE)
    # Black surrounds nearly completely
    b.play(3, 4, BLACK)
    b.play(3, 5, BLACK)
    b.play(5, 4, BLACK)
    b.play(5, 5, BLACK)
    b.play(4, 3, BLACK)
    # Now if black plays at (4,6), white group at (4,4)(4,5) is captured.
    # Black at (4,6) would have its own liberty appear at (4,4) after capture.
    result = b.play(4, 6, BLACK)
    assert result.captured == 2
    assert b[(4, 6)] == BLACK


def test_corner_capture():
    """Capture in a corner — common edge case."""
    b = Board(9)
    b.play(0, 0, WHITE)
    b.play(0, 1, BLACK)
    result = b.play(1, 0, BLACK)
    assert result.captured == 1
    assert b[(0, 0)] == EMPTY


def test_ko_rule():
    """A standard ko shape — playing the same capture twice in a row is forbidden."""
    b = Board(9)
    # Set up a ko: black captures one white stone, white can't recapture
    # immediately because it would repeat the previous position.
    #
    #   . X O .
    #   X . X O
    #   . X O .
    #
    # Place stones
    b.play(0, 1, BLACK)
    b.play(0, 2, WHITE)
    b.play(1, 0, BLACK)
    b.play(1, 2, BLACK)
    b.play(1, 3, WHITE)
    b.play(2, 1, BLACK)
    b.play(2, 2, WHITE)
    # White at (1,1) — black would capture white at (1,2) ... wait, let me redo.
    # Simpler ko: black captures, white tries to immediately recapture.
    b2 = Board(9)
    # Position:
    #   . B W .
    #   B W . W
    #   . B W .
    b2.play(0, 1, BLACK)
    b2.play(1, 0, BLACK)
    b2.play(2, 1, BLACK)
    b2.play(0, 2, WHITE)
    b2.play(1, 3, WHITE)
    b2.play(2, 2, WHITE)
    b2.play(1, 1, WHITE)
    # Black captures white at (1,1) by playing (1,2)
    result = b2.play(1, 2, BLACK)
    assert result.captured == 1
    assert b2[(1, 1)] == EMPTY
    # White tries to recapture by playing (1,1) — this would restore the
    # exact previous position, which is forbidden by superko.
    with pytest.raises(IllegalMove, match="ko"):
        b2.play(1, 1, WHITE)


def test_legal_moves_excludes_suicide():
    b = Board(9)
    b.play(3, 4, BLACK)
    b.play(5, 4, BLACK)
    b.play(4, 3, BLACK)
    b.play(4, 5, BLACK)
    legal = b.legal_moves(WHITE)
    assert (4, 4) not in legal  # surrounded — suicide


def test_legal_moves_includes_capturing_move():
    b = Board(9)
    # White stone with one liberty
    b.play(0, 0, WHITE)
    b.play(0, 1, BLACK)
    legal = b.legal_moves(BLACK)
    # (1, 0) captures (0, 0) — legal
    assert (1, 0) in legal


def test_board_copy_independence():
    b = Board(9)
    b.play(4, 4, BLACK)
    b2 = b.copy()
    b2.play(4, 5, WHITE)
    assert b[(4, 5)] == EMPTY
    assert b2[(4, 5)] == WHITE


def test_is_legal_does_not_mutate():
    b = Board(9)
    b.play(4, 4, BLACK)
    before = [row[:] for row in b.grid]
    before_hash = b._hash
    before_history = set(b._history_hashes)
    assert b.is_legal(3, 3, WHITE) is True
    assert b.grid == before
    assert b._hash == before_hash
    assert b._history_hashes == before_history


if __name__ == "__main__":
    import sys
    pytest.main([__file__, "-v"] + sys.argv[1:])
