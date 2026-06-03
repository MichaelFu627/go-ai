"""
Kifu (game record) storage: save self-play games to disk for later review.

Each saved game is a JSON file with:
  - meta: iteration, timestamp, who won, score, etc.
  - moves: list of {color, kind, row, col, label} per move
  - settings: board size, komi, simulations used

The format is intentionally simple — no SGF complexity. The frontend reads
these directly. To export to SGF for use in other tools, write a converter
later.
"""

from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Optional

from board import BLACK, WHITE
from game import Game


def _color_name(c: int) -> str:
    return "black" if c == BLACK else "white"


def _coord_label(r: int, c: int, size: int) -> str:
    cols = "ABCDEFGHJKLMNOPQRSTUVWXYZ"
    return f"{cols[c]}{size - r}"


def save_game(game: Game, out_dir: Path, *,
              iteration: Optional[int] = None,
              game_idx: Optional[int] = None,
              simulations: Optional[int] = None,
              source: str = "selfplay",
              extra: Optional[dict] = None) -> Path:
    """Save a finished (or stopped) game to a JSON file.

    Returns the path written. Filenames are sortable by time + iteration so the
    library view shows newest training games first.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = int(time.time() * 1000)  # ms since epoch — sortable
    iter_part = f"iter{iteration:03d}_" if iteration is not None else ""
    idx_part = f"g{game_idx:02d}_" if game_idx is not None else ""
    fname = f"{ts}_{iter_part}{idx_part}kifu.json"
    path = out_dir / fname

    score = game.area_score() if game.is_over else None
    winner = None
    if game.resigned_by is not None:
        winner = _color_name(WHITE if game.resigned_by == BLACK else BLACK)
    elif score:
        winner = score["winner"]

    moves = []
    for m in game.history:
        row = {"color": _color_name(m.color), "kind": m.kind}
        if m.kind == "play":
            row["row"] = m.row
            row["col"] = m.col
            row["label"] = _coord_label(m.row, m.col, game.size)
            row["captured"] = m.captured
        moves.append(row)

    data = {
        "version": 1,
        "id": path.stem,
        "saved_at_ms": ts,
        "source": source,                   # 'selfplay', 'human_vs_ai', etc.
        "meta": {
            "iteration": iteration,
            "game_idx": game_idx,
            "simulations": simulations,
            "winner": winner,
            "resigned_by": _color_name(game.resigned_by) if game.resigned_by is not None else None,
            "score": score,
            "move_count": len(moves),
        },
        "settings": {
            "size": game.size,
            "komi": game.komi,
        },
        "moves": moves,
        "extra": extra or {},
    }
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    return path


def list_games(dir_path: Path, limit: int = 200) -> list[dict]:
    """List saved games, newest first. Returns lightweight metadata only
    (not the move list) so the library view stays fast."""
    dir_path = Path(dir_path)
    if not dir_path.exists():
        return []
    files = sorted(dir_path.glob("*.json"), key=lambda p: p.name, reverse=True)
    out = []
    for p in files[:limit]:
        try:
            with open(p) as f:
                d = json.load(f)
            out.append({
                "id": d.get("id", p.stem),
                "saved_at_ms": d.get("saved_at_ms"),
                "source": d.get("source"),
                "meta": d.get("meta", {}),
                "settings": d.get("settings", {}),
            })
        except Exception:
            continue
    return out


def load_game(dir_path: Path, game_id: str) -> Optional[dict]:
    """Load a single game by id. Returns the full game dict, or None."""
    dir_path = Path(dir_path)
    candidate = dir_path / f"{game_id}.json"
    if not candidate.exists():
        # Allow lookup by id substring (defensive — ids can have prefixes)
        matches = list(dir_path.glob(f"*{game_id}*.json"))
        if not matches:
            return None
        candidate = matches[0]
    with open(candidate) as f:
        return json.load(f)
