# Diplomacy Web App

Human vs AI Diplomacy gameplay with training data collection.

## Quick Start

### 1. Install Dependencies

```bash
# Backend (from repo root)
pip install fastapi uvicorn[standard] python-multipart

# Frontend
cd web/frontend
npm install
```

### 2. Run Development Servers

**Terminal 1 - Backend:**
```bash
# From repo root
INFERENCE_MODE=mock PYTHONPATH=. uvicorn web.backend.server:app --reload --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd web/frontend
npm run dev
```

Then open http://localhost:3000

## Inference Modes

Set `INFERENCE_MODE` environment variable:

- `mock` (default): Random valid moves, no GPU needed. Great for UI development.
- `modal`: Use Modal's deployed InferenceEngine. Requires Modal CLI and deployed app.
- `local`: Use local vLLM. Requires CUDA GPU.

### Using Modal Inference

```bash
INFERENCE_MODE=modal PYTHONPATH=. uvicorn web.backend.server:app --reload --port 8000
```

### Using Local vLLM

```bash
# Requires CUDA GPU
INFERENCE_MODE=local MODEL_ID=Qwen/Qwen2.5-7B-Instruct PYTHONPATH=. uvicorn web.backend.server:app --port 8000
```

## Project Structure

```
web/
├── frontend/              # Next.js app
│   ├── app/               # Pages
│   │   ├── page.tsx       # Home - game creation
│   │   └── game/[id]/     # Game page
│   ├── components/        # React components
│   │   ├── GameBoard.tsx
│   │   ├── OrderInput.tsx
│   │   ├── PhaseDisplay.tsx
│   │   └── PowerStatus.tsx
│   └── lib/               # Utilities
│       ├── api.ts         # API client with Effect
│       ├── store.ts       # Zustand state management
│       └── types.ts       # TypeScript types with Effect Schema
│
├── backend/               # FastAPI server
│   ├── server.py          # Main app
│   ├── routes/
│   │   ├── game.py        # Game CRUD + orders
│   │   └── training_data.py
│   ├── services/
│   │   ├── game_session.py    # Game state management
│   │   ├── inference_service.py # Mock/Modal/Local vLLM
│   │   └── persistence.py     # SQLite storage
│   └── models/
│       └── schemas.py     # Pydantic models
│
└── data/                  # Created at runtime
    └── diplomacy.db       # SQLite database
```

## API Endpoints

### Game

- `POST /api/game/new` - Create new game
- `GET /api/game/{id}` - Get game state
- `POST /api/game/{id}/orders` - Submit orders
- `GET /api/game/config/powers` - List powers
- `GET /api/game/config/adapters` - List AI difficulties

### Training Data

- `GET /api/training/stats` - Collection statistics
- `GET /api/training/export` - Export trajectories
- `POST /api/training/mark-exported/{batch_id}` - Mark as exported

## Training Data Collection

The app collects training trajectories in the same format as the rollout system:

```python
{
    "prompt": "...",
    "completion": "...",
    "reward": 0.5,
    "group_id": "game123_FRANCE_1901",
    "prompt_token_ids": [...],
    "completion_token_ids": [...],
    "completion_logprobs": [...]
}
```

Export via API:
```bash
curl http://localhost:8000/api/training/export
```

## Development Status

### Completed
- [x] Backend infrastructure (FastAPI)
- [x] Game session management
- [x] Inference service (mock/modal/local)
- [x] SQLite persistence
- [x] Training data collection
- [x] Frontend home page
- [x] Game page with order input
- [x] Power/adapter selection

### TODO
- [ ] SVG game board visualization
- [ ] Turn history display
- [ ] Replay viewer
- [ ] Auth for leaderboards
- [ ] Multiplayer support
