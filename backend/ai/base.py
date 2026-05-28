"""AI base interface. Future MCTS AI will implement the same protocol."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable
from game import Game


@dataclass
class AIDecision:
    """What the AI decided to do, plus optional introspection data.

    The extra fields (winrate, simulations, top_moves, principal_variation)
    are placeholders for MCTS — the frontend will display them when available
    and ignore them when they're None / empty.
    """
    kind: str  # "play", "pass", or "resign"
    row: Optional[int] = None
    col: Optional[int] = None

    # Optional introspection — filled by MCTS later.
    winrate: Optional[float] = None        # value-network estimate, 0-1
    simulations: int = 0
    top_moves: list[dict] = field(default_factory=list)  # [{row, col, visits, q}]
    principal_variation: list[dict] = field(default_factory=list)


@runtime_checkable
class AI(Protocol):
    """All AI strategies implement this single method."""
    def select_move(self, game: Game) -> AIDecision: ...
