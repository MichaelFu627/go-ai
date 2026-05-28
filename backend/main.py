"""FastAPI app exposing the Go game over HTTP.

Endpoints:
    POST /api/game/new                Create a new game.
    GET  /api/game/{id}/state         Get current state.
    POST /api/game/{id}/move          Play a stone.
    POST /api/game/{id}/pass          Pass.
    POST /api/game/{id}/resign        Resign.
    POST /api/game/{id}/undo          Undo last move.
    POST /api/game/{id}/ai-move       Request AI to play.
    GET  /api/game/{id}/score         Get area score (only meaningful at end).

State is held in memory — a single-process server is fine for our use.
"""

from __future__ import annotations
import uuid
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from board import IllegalMove, BLACK, WHITE, EMPTY
from game import Game
from ai.random_ai import RandomAI
from ai.mcts_ai import MCTSAI


app = FastAPI(title="Go AI", version="0.1")

# Permissive CORS for local development — the frontend runs on its own dev server.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- in-memory store -----------------------------------------------------

GAMES: dict[str, Game] = {}
AIS: dict[str, object] = {}  # one AI instance per game (RandomAI or MCTSAI)


def _get_game(game_id: str) -> Game:
    if game_id not in GAMES:
        raise HTTPException(status_code=404, detail="game not found")
    return GAMES[game_id]


# ---- request / response models -------------------------------------------

class NewGameRequest(BaseModel):
    size: int = Field(9, ge=2, le=19)
    komi: Optional[float] = None  # None → pick a size-appropriate default
    ai_color: Optional[str] = "white"  # "black" | "white" | None
    seed: Optional[int] = None
    ai_type: str = "mcts"          # "mcts" | "random"
    ai_strength: Optional[int] = None  # MCTS simulations; None → size default


# Conventional komi values per board size (Chinese rules, area scoring).
DEFAULT_KOMI = {9: 7.5, 13: 7.5, 19: 7.5}

# Default MCTS simulation budget per board size (bigger boards need more,
# but pure-Python rollouts are slow, so keep these modest).
DEFAULT_SIMS = {9: 300, 13: 200, 19: 120}


class MoveRequest(BaseModel):
    row: int
    col: int


def _color_from_str(s: Optional[str]) -> Optional[int]:
    if s is None: return None
    s = s.lower()
    if s == "black": return BLACK
    if s == "white": return WHITE
    return None


def _color_to_str(c: Optional[int]) -> Optional[str]:
    if c == BLACK: return "black"
    if c == WHITE: return "white"
    return None


def _serialize_state(game_id: str, game: Game, ai_info: Optional[dict] = None) -> dict:
    """Single source of truth for what the frontend receives."""
    return {
        "game_id": game_id,
        "size": game.size,
        "komi": game.komi,
        "board": game.board.grid,
        "turn": _color_to_str(game.turn),
        "ai_color": _color_to_str(game.ai_color),
        "human_color": _color_to_str(game.human_color),
        "captures": {
            "black": game.captures[BLACK],
            "white": game.captures[WHITE],
        },
        "history": [
            {
                "color": _color_to_str(m.color),
                "kind": m.kind,
                "row": m.row,
                "col": m.col,
                "captured": m.captured,
                "label": (game.coord_label(m.row, m.col)
                          if m.kind == "play" and m.row is not None else None),
            }
            for m in game.history
        ],
        "consecutive_passes": game.consecutive_passes,
        "is_over": game.is_over,
        "resigned_by": _color_to_str(game.resigned_by),
        "ai_info": ai_info or {},
    }


# ---- endpoints -----------------------------------------------------------

@app.post("/api/game/new")
def new_game(req: NewGameRequest):
    game_id = uuid.uuid4().hex[:12]
    komi = req.komi if req.komi is not None else DEFAULT_KOMI.get(req.size, 7.5)
    game = Game(size=req.size, komi=komi, ai_color=_color_from_str(req.ai_color))
    GAMES[game_id] = game

    if req.ai_type == "random":
        AIS[game_id] = RandomAI(seed=req.seed)
    else:
        sims = req.ai_strength or DEFAULT_SIMS.get(req.size, 200)
        AIS[game_id] = MCTSAI(simulations=sims, seed=req.seed)

    return _serialize_state(game_id, game)


@app.get("/api/game/{game_id}/state")
def get_state(game_id: str):
    return _serialize_state(game_id, _get_game(game_id))


@app.post("/api/game/{game_id}/move")
def play_move(game_id: str, req: MoveRequest):
    game = _get_game(game_id)
    try:
        game.play(req.row, req.col)
    except IllegalMove as e:
        raise HTTPException(status_code=400, detail=f"illegal move: {e}")
    return _serialize_state(game_id, game)


@app.post("/api/game/{game_id}/pass")
def play_pass(game_id: str):
    game = _get_game(game_id)
    try:
        game.play_pass()
    except IllegalMove as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _serialize_state(game_id, game)


@app.post("/api/game/{game_id}/resign")
def resign(game_id: str):
    game = _get_game(game_id)
    try:
        game.resign()
    except IllegalMove as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _serialize_state(game_id, game)


@app.post("/api/game/{game_id}/undo")
def undo(game_id: str):
    game = _get_game(game_id)
    # Undo twice if the last move was AI's — so the human gets back to their turn.
    last = game.undo()
    if last and game.ai_color is not None and last.color == game.ai_color:
        game.undo()
    return _serialize_state(game_id, game)


@app.post("/api/game/{game_id}/ai-move")
def ai_move(game_id: str):
    game = _get_game(game_id)
    ai = AIS[game_id]
    if game.is_over:
        raise HTTPException(status_code=400, detail="game is over")
    if game.ai_color is None or game.turn != game.ai_color:
        raise HTTPException(status_code=400, detail="not AI's turn")

    decision = ai.select_move(game)
    if decision.kind == "play":
        try:
            game.play(decision.row, decision.col)
        except IllegalMove as e:
            raise HTTPException(status_code=500, detail=f"AI produced illegal move: {e}")
    elif decision.kind == "pass":
        game.play_pass()
    elif decision.kind == "resign":
        game.resign()

    ai_info = {
        "kind": decision.kind,
        "winrate": decision.winrate,
        "simulations": decision.simulations,
        "top_moves": decision.top_moves,
        "principal_variation": decision.principal_variation,
    }
    return _serialize_state(game_id, game, ai_info=ai_info)


@app.get("/api/game/{game_id}/score")
def score(game_id: str):
    return _get_game(game_id).area_score()


@app.get("/api/health")
def health():
    return {"ok": True, "games": len(GAMES)}
