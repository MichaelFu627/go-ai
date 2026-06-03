"""
Main training loop.

One ITERATION = self-play games + training steps + evaluation vs previous best.

Run:   python train.py
Args:  see --help. All defaults are small/quick so you can verify the pipeline
       end to end in a few minutes on a laptop.

What you should see:
  - loss decreasing iteration over iteration
  - win rate against the previous best generally >= 50%
  - over many iterations: win rate vs an early checkpoint increasing
"""

from __future__ import annotations
import argparse
import copy
import json
import os
import time
from pathlib import Path
import torch

from net import GoNet, count_parameters
from selfplay import play_one_game
from replay_buffer import ReplayBuffer
from train_step import train_step
from evaluate import play_match


def pick_device() -> str:
    """Use Apple MPS if available, else CUDA, else CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def save_checkpoint(net, path: Path, meta: dict | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_state": net.state_dict()}
    if meta:
        payload["meta"] = meta
    torch.save(payload, path)


def load_checkpoint(net, path: Path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net.load_state_dict(ck["model_state"])
    return ck.get("meta", {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=3,
                    help="Number of training iterations")
    ap.add_argument("--games-per-iter", type=int, default=4,
                    help="Self-play games per iteration")
    ap.add_argument("--sims", type=int, default=40,
                    help="MCTS simulations per move during self-play")
    ap.add_argument("--eval-sims", type=int, default=80,
                    help="MCTS simulations per move during evaluation")
    ap.add_argument("--eval-games", type=int, default=10,
                    help="Number of evaluation games vs previous best")
    ap.add_argument("--train-steps", type=int, default=40,
                    help="Training steps per iteration")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--buffer-size", type=int, default=10000)
    ap.add_argument("--board-size", type=int, default=9)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--res-blocks", type=int, default=5)
    ap.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    ap.add_argument("--resume", type=str, default=None,
                    help="Path to a checkpoint to resume from")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto",
                    choices=["auto", "cpu", "mps", "cuda"])
    args = ap.parse_args()

    device = pick_device() if args.device == "auto" else args.device
    torch.manual_seed(args.seed)

    # ---- setup ----
    print(f"[setup] device={device}, board_size={args.board_size}")
    net = GoNet(board_size=args.board_size,
                channels=args.channels,
                num_res_blocks=args.res_blocks).to(device)
    print(f"[setup] network parameters: {count_parameters(net):,}")

    if args.resume:
        meta = load_checkpoint(net, Path(args.resume))
        print(f"[setup] resumed from {args.resume} (meta={meta})")

    # The 'best' network is the one we'll keep playing against in evaluation.
    # Start it as a copy of the initial network.
    best_net = GoNet(board_size=args.board_size,
                     channels=args.channels,
                     num_res_blocks=args.res_blocks).to(device)
    best_net.load_state_dict(net.state_dict())

    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr,
                                momentum=0.9, weight_decay=args.weight_decay)
    buf = ReplayBuffer(capacity=args.buffer_size)

    ckdir = Path(args.checkpoint_dir)
    ckdir.mkdir(parents=True, exist_ok=True)
    log_path = ckdir / "log.jsonl"

    # ---- iterations ----
    total_t0 = time.time()
    for it in range(1, args.iterations + 1):
        print(f"\n=== iteration {it}/{args.iterations} ===")

        # 1. SELF-PLAY
        t_sp = time.time()
        games_info = []
        kifu_dir = ckdir / "games"
        for g in range(args.games_per_iter):
            samples, winner, info = play_one_game(
                net, board_size=args.board_size, simulations=args.sims,
                device=device, seed=args.seed * 1000 + it * 100 + g,
                save_kifu_to=kifu_dir,
                kifu_iteration=it, kifu_game_idx=g,
            )
            buf.add(samples)
            games_info.append(info)
            print(f"  selfplay {g+1}/{args.games_per_iter}: "
                  f"{info['moves']} moves, {info['winner']} won "
                  f"({info['duration_s']:.1f}s, buffer={len(buf)})")
        sp_time = time.time() - t_sp

        # 2. TRAIN
        t_tr = time.time()
        losses = []
        for step in range(args.train_steps):
            if len(buf) < args.batch_size:
                break
            batch = buf.sample(args.batch_size)
            info = train_step(net, optimizer, batch, device=device)
            losses.append(info)
        if losses:
            avg = lambda k: sum(l[k] for l in losses) / len(losses)
            print(f"  training: {len(losses)} steps, "
                  f"avg loss={avg('loss'):.3f} "
                  f"(policy={avg('policy_loss'):.3f}, value={avg('value_loss'):.3f})")
        tr_time = time.time() - t_tr

        # 3. EVALUATE vs current best
        t_ev = time.time()
        result = play_match(net, best_net,
                            num_games=args.eval_games,
                            board_size=args.board_size,
                            simulations=args.eval_sims,
                            device=device,
                            seed=args.seed * 7 + it)
        ev_time = time.time() - t_ev
        wr = result["win_rate"]
        print(f"  evaluation: {result['a_wins']}-{result['b_wins']} "
              f"vs previous best (win_rate={wr:.0%}, {ev_time:.1f}s)")

        # 4. PROMOTE if new net is better.
        # AlphaGo Zero threshold is 55%; we use 55% here too.
        promoted = wr >= 0.55
        if promoted:
            best_net.load_state_dict(net.state_dict())
            save_checkpoint(best_net, ckdir / "best.pt",
                            meta={"iteration": it, "win_rate": wr})
            print(f"  -> PROMOTED to new best (win_rate {wr:.0%} >= 55%)")
        save_checkpoint(net, ckdir / f"iter_{it:03d}.pt",
                        meta={"iteration": it, "win_rate": wr})

        # 5. log this iteration to a JSONL file (one JSON object per line).
        row = {
            "iteration": it,
            "games": args.games_per_iter,
            "sp_time_s": round(sp_time, 1),
            "train_steps": len(losses),
            "train_time_s": round(tr_time, 1),
            "avg_loss": round(sum(l["loss"] for l in losses) / max(1, len(losses)), 4),
            "avg_policy_loss": round(sum(l["policy_loss"] for l in losses) / max(1, len(losses)), 4),
            "avg_value_loss": round(sum(l["value_loss"] for l in losses) / max(1, len(losses)), 4),
            "eval_wins": result["a_wins"],
            "eval_losses": result["b_wins"],
            "eval_win_rate": round(wr, 3),
            "promoted": promoted,
            "buffer_size": len(buf),
            "total_elapsed_s": round(time.time() - total_t0, 1),
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")

    print(f"\n[done] total {time.time()-total_t0:.1f}s. "
          f"Log: {log_path}  Checkpoints: {ckdir}/")


if __name__ == "__main__":
    main()
