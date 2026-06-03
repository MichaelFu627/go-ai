"""
Replay buffer — a fixed-size rolling window of training samples.

The training loop pulls random mini-batches from here. Keeping only the most
recent N samples (paper does this too) means:
  - we don't grow unbounded
  - early, weak-network data gets washed out as the network improves
"""

from __future__ import annotations
import random
from collections import deque
from typing import Iterable
from selfplay import TrainingSample


class ReplayBuffer:
    def __init__(self, capacity: int = 20000):
        self.capacity = capacity
        self.data: deque[TrainingSample] = deque(maxlen=capacity)

    def add(self, samples: Iterable[TrainingSample]) -> None:
        for s in samples:
            self.data.append(s)

    def __len__(self) -> int:
        return len(self.data)

    def sample(self, batch_size: int) -> list[TrainingSample]:
        if batch_size >= len(self.data):
            return list(self.data)
        return random.sample(self.data, batch_size)

    def clear(self) -> None:
        self.data.clear()
