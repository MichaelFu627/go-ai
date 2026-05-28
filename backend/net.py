"""
The neural network — a dual-head residual network (AlphaGo Zero style).

A shared convolutional trunk reads the feature planes and builds an internal
representation of the position. Two heads branch off it:

  - policy head : outputs a probability distribution over all moves
                  (every board point + pass). This is the "where to look"
                  prior P that guides MCTS.
  - value head  : outputs a single scalar in [-1, 1] estimating the game
                  result from the current player's perspective
                  (+1 = current player winning, -1 = losing).

This differs from the 2016 AlphaGo paper, which used two SEPARATE networks.
Sharing a trunk (the 2017 AlphaGo Zero design) is simpler to train and works
better for a from-scratch project — both heads benefit from the same learned
understanding of the board.

Sizes are deliberately small for 9x9 on a home machine: a handful of residual
blocks with 64 channels. Scale these up for bigger boards or more compute.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from features import NUM_PLANES, policy_size


class ResidualBlock(nn.Module):
    """Standard ResNet block: two 3x3 convs with a skip connection.
    The skip connection lets gradients flow through deep stacks without
    vanishing — this is why we can train more layers reliably."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual          # skip connection
        return F.relu(out)


class GoNet(nn.Module):
    """Dual-head network for a board of a given size."""

    def __init__(self, board_size: int = 9, channels: int = 64,
                 num_res_blocks: int = 5):
        super().__init__()
        self.board_size = board_size
        self.n_policy = policy_size(board_size)  # points + pass

        # ---- shared trunk ----
        self.conv_in = nn.Conv2d(NUM_PLANES, channels, 3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(channels)
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(channels) for _ in range(num_res_blocks)]
        )

        # ---- policy head ----
        # 1x1 conv down to 2 channels, then a linear layer to the move logits.
        self.policy_conv = nn.Conv2d(channels, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, self.n_policy)

        # ---- value head ----
        # 1x1 conv down to 1 channel, then two linear layers to a single tanh.
        self.value_conv = nn.Conv2d(channels, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        """x: (batch, NUM_PLANES, size, size).
        Returns (policy_logits, value):
          policy_logits: (batch, n_policy) — raw scores, apply softmax outside
          value: (batch, 1) — in [-1, 1]
        """
        # trunk
        out = F.relu(self.bn_in(self.conv_in(x)))
        for block in self.res_blocks:
            out = block(out)

        # policy head
        p = F.relu(self.policy_bn(self.policy_conv(out)))
        p = p.view(p.size(0), -1)
        policy_logits = self.policy_fc(p)

        # value head
        v = F.relu(self.value_bn(self.value_conv(out)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value

    @torch.no_grad()
    def predict(self, planes, device="cpu"):
        """Convenience for inference on a single position.
        planes: numpy array (NUM_PLANES, size, size).
        Returns (policy_probs, value_scalar):
          policy_probs: numpy array (n_policy,) summing to 1
          value_scalar: python float in [-1, 1]
        """
        self.eval()
        x = torch.from_numpy(planes).unsqueeze(0).to(device)  # add batch dim
        logits, value = self.forward(x)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        return probs, float(value.item())


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
