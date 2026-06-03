"""
One training step on a batch of self-play samples.

Loss = policy_loss + value_loss + L2_regularization
  policy_loss = cross-entropy between network policy and MCTS pi
  value_loss  = MSE between network value and game outcome z

This matches the AlphaGo Zero loss (a single combined loss for the dual-head
network). We use PyTorch's optimizer to handle L2 regularization via the
optimizer's weight_decay parameter, which is cleaner than adding it to the
loss explicitly.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F


def train_step(net, optimizer, samples, device: str = "cpu") -> dict:
    """Run one optimization step on a batch of TrainingSamples.

    Returns a dict with the loss components (as floats) for logging.
    """
    net.train()

    # Stack samples into tensors.
    feats = np.stack([s.encode_features() for s in samples])  # (B, C, H, W)
    pis   = np.stack([s.pi for s in samples])                 # (B, n_policy)
    zs    = np.array([s.z for s in samples], dtype=np.float32) # (B,)

    x  = torch.from_numpy(feats).to(device)
    pi_target = torch.from_numpy(pis).to(device)
    z_target  = torch.from_numpy(zs).to(device).unsqueeze(1)  # (B, 1)

    # Forward.
    policy_logits, value = net(x)

    # Policy loss: cross-entropy between network log-softmax and MCTS pi.
    #   -sum(pi * log_softmax(logits)) per row, mean over batch.
    log_probs = F.log_softmax(policy_logits, dim=1)
    policy_loss = -(pi_target * log_probs).sum(dim=1).mean()

    # Value loss: MSE between predicted value and outcome z.
    value_loss = F.mse_loss(value, z_target)

    loss = policy_loss + value_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {
        "loss": float(loss.item()),
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "batch_size": len(samples),
    }
