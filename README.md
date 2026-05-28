# Go AI · Week 1

A 9×9 Go web app — Python (FastAPI) backend with full rule enforcement, Vite + vanilla JS frontend with an editorial / Edo-kifu aesthetic.

The AI in this phase is a **random walker** (with light eye-avoidance). The backend and API are designed so that swapping in an MCTS + neural network AI later is a single-class change.

```
go-ai/
├── backend/
│   ├── board.py          # Go rules: capture, suicide, positional superko
│   ├── game.py           # Game state, history, area scoring
│   ├── main.py           # FastAPI app
│   ├── test_board.py     # Rule tests (12 cases)
│   ├── ai/
│   │   ├── base.py       # AI Protocol + AIDecision dataclass
│   │   └── random_ai.py  # First-week stub
│   └── requirements.txt
└── frontend/
    ├── index.html
    ├── src/
    │   ├── style.css     # The aesthetic
    │   ├── board.js      # SVG board renderer
    │   ├── api.js        # Backend client
    │   └── main.js       # Glue
    ├── package.json
    └── vite.config.js    # Proxies /api → :8000
```

## Run

### Backend
```bash
cd backend
python -m venv .venv && source .venv/bin/activate     # or your usual env
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
Visit http://localhost:8000/docs for the auto-generated API docs.

### Frontend
```bash
cd frontend
npm install
npm run dev
```
Visit http://localhost:5173

### Tests
```bash
cd backend && python -m pytest test_board.py -v
```

## How the pieces fit together

The whole game is plain HTTP state. The frontend never makes assumptions about how the AI works:

```
POST /api/game/new          → { game_id, board, turn, ... }
POST /api/game/{id}/move    → updated state
POST /api/game/{id}/ai-move → updated state + ai_info
```

`ai_info` already has fields for `winrate`, `simulations`, `top_moves`, and `principal_variation` — the random AI leaves them empty, MCTS will fill them in next week.

## Design notes

- **Positional superko** (not just basic ko) via Zobrist hashing — handles every legal Ko situation including triple-ko.
- **Suicide is forbidden**, but a move that would otherwise be suicide is legal if it captures opponent stones first.
- **Two consecutive passes** end the game; Chinese-rules area scoring is implemented but dead-stone detection is not (this matters when we evaluate real games, not Week-1 random walks).
- **Random AI avoids filling its own eyes** — without this, random Go is unwatchable.
- **Undo unwinds both human + AI moves** so the human ends up on their turn.

## Next week
Swap `RandomAI` for an `MCTSAI` that implements the same `select_move(game) -> AIDecision` interface. No frontend changes needed.
