Comprehensive Implementation Plan: Diplomacy Web App
Executive Summary
Build a Next.js web app at /web with a FastAPI backend that reuses existing Python code. The app lets humans play Diplomacy against LLM bots while collecting training data in the same format as your existing rollout system.
Project Structure
diplomacy-v2/
├── src/                           # Existing Python ML code
│   ├── agents/
│   ├── apps/
│   ├── engine/
│   ├── inference/
│   └── ...
├── web/                           # NEW: Next.js + FastAPI
│   ├── frontend/                  # Next.js app
│   │   ├── app/
│   │   │   ├── page.tsx           # Home/lobby
│   │   │   ├── game/[id]/page.tsx # Game board
│   │   │   ├── api/               # API routes (proxy to backend)
│   │   │   └── layout.tsx
│   │   ├── components/
│   │   │   ├── GameBoard.tsx
│   │   │   ├── OrderInput.tsx
│   │   │   ├── PhaseDisplay.tsx
│   │   │   └── ...
│   │   ├── lib/
│   │   │   ├── api.ts             # Backend client
│   │   │   └── types.ts           # Shared TypeScript types
│   │   ├── package.json
│   │   └── next.config.js
│   │
│   ├── backend/                   # FastAPI server
│   │   ├── __init__.py
│   │   ├── server.py              # Main FastAPI app
│   │   ├── routes/
│   │   │   ├── game.py            # Game endpoints
│   │   │   ├── inference.py       # LLM endpoints
│   │   │   └── training_data.py   # Export endpoints
│   │   ├── services/
│   │   │   ├── game_session.py    # Game state management
│   │   │   ├── inference_service.py # Local/Modal inference
│   │   │   └── persistence.py     # Storage layer
│   │   ├── models/
│   │   │   └── schemas.py         # Pydantic models
│   │   └── requirements.txt
│   │
│   ├── docker-compose.yml         # Local dev orchestration
│   └── README.md
│
├── pyproject.toml                 # Add web backend deps
└── ...
Phase 1: Infrastructure Setup
1.1 Create Web Directory Structure
mkdir -p web/{frontend,backend}
mkdir -p web/backend/{routes,services,models}
mkdir -p web/frontend/{app,components,lib}
1.2 Backend Dependencies
Add to pyproject.toml:
[project.optional-dependencies]
web = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.32.0",
    "python-multipart>=0.0.9",
]
Or create web/backend/requirements.txt:
fastapi>=0.115.0
uvicorn[standard]>=0.32.0
python-multipart>=0.0.9
1.3 Frontend Setup
cd web/frontend
npx create-next-app@latest . --typescript --tailwind --eslint --app --no-src-dir
npm install zustand @tanstack/react-query
Phase 2: Backend API (FastAPI)
2.1 Core Server (web/backend/server.py)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from web.backend.routes import game, inference, training_data

app = FastAPI(title="Diplomacy Web API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(game.router, prefix="/api/game", tags=["game"])
app.include_router(inference.router, prefix="/api/inference", tags=["inference"])
app.include_router(training_data.router, prefix="/api/training", tags=["training"])
2.2 Game Session Service (web/backend/services/game_session.py)
This is the key abstraction that wraps DiplomacyWrapper for web use:
from dataclasses import dataclass, field
from typing import Any
import uuid
import time

from src.engine.wrapper import DiplomacyWrapper
from src.agents.llm_agent import LLMAgent, PromptConfig
from src.utils.parsing import extract_orders
from src.utils.scoring import calculate_final_scores


@dataclass
class GameSession:
    """Manages a single human-vs-AI Diplomacy game."""
    
    id: str
    game: DiplomacyWrapper
    agent: LLMAgent
    human_power: str
    adapter_name: str | None
    created_at: float
    
    # Training data collection
    trajectories: list[dict] = field(default_factory=list)
    turn_history: list[dict] = field(default_factory=list)
    
    @classmethod
    def create(
        cls,
        human_power: str = "FRANCE",
        adapter_name: str | None = None,
        horizon: int = 10,
    ) -> "GameSession":
        session_id = str(uuid.uuid4())[:8]
        game = DiplomacyWrapper(game_id=session_id, horizon=horizon)
        agent = LLMAgent(config=PromptConfig(compact_mode=True))
        
        return cls(
            id=session_id,
            game=game,
            agent=agent,
            human_power=human_power,
            adapter_name=adapter_name,
            created_at=time.time(),
        )
    
    def get_state(self) -> dict[str, Any]:
        """Get current game state for frontend."""
        return {
            "id": self.id,
            "phase": self.game.get_current_phase(),
            "year": self.game.get_year(),
            "is_done": self.game.is_done(),
            "human_power": self.human_power,
            "board_context": self.game.get_board_context(self.human_power),
            "valid_moves": self.game.get_valid_moves(self.human_power),
            "all_units": self._get_all_units(),
            "all_centers": self._get_all_centers(),
        }
    
    def _get_all_units(self) -> dict[str, list[str]]:
        return {
            power: list(obj.units) 
            for power, obj in self.game.game.powers.items()
        }
    
    def _get_all_centers(self) -> dict[str, list[str]]:
        return {
            power: list(obj.centers) 
            for power, obj in self.game.game.powers.items()
        }
    
    def collect_trajectory(
        self,
        power: str,
        prompt: str,
        completion: str,
        response_data: dict,
    ) -> None:
        """Collect training data for a turn."""
        self.trajectories.append({
            "prompt": prompt,
            "completion": completion,
            "prompt_token_ids": response_data.get("prompt_token_ids", []),
            "completion_token_ids": response_data.get("token_ids", []),
            "completion_logprobs": response_data.get("completion_logprobs", []),
            "group_id": f"{self.id}_{power}_{self.game.get_year()}",
            "power": power,
            "phase": self.game.get_current_phase(),
            # Reward computed at game end
        })
    
    def finalize_trajectories(self) -> list[dict]:
        """Add rewards to trajectories based on final scores."""
        if not self.trajectories:
            return []
        
        final_scores = calculate_final_scores(self.game)
        
        for traj in self.trajectories:
            power = traj.pop("power")  # Remove internal field
            traj["reward"] = final_scores.get(power, 0.0)
        
        return self.trajectories
2.3 Inference Service (web/backend/services/inference_service.py)
Supports both local vLLM and Modal:
from abc import ABC, abstractmethod
from typing import Any
import os


class InferenceService(ABC):
    """Abstract inference service interface."""
    
    @abstractmethod
    async def generate(
        self,
        prompts: list[str],
        valid_moves: list[dict],
        lora_name: str | None = None,
        temperature: float = 0.8,
    ) -> list[dict]:
        pass


class LocalVLLMService(InferenceService):
    """Local vLLM inference for debugging."""
    
    def __init__(self, model_id: str = "Qwen/Qwen2.5-7B-Instruct"):
        from vllm import LLM, SamplingParams
        from src.inference.logits import DiplomacyLogitsProcessor
        
        self.model_id = model_id
        self.llm = LLM(
            model=model_id,
            enable_lora=True,
            max_lora_rank=16,
        )
        self.logits_processor_cls = DiplomacyLogitsProcessor
    
    async def generate(
        self,
        prompts: list[str],
        valid_moves: list[dict],
        lora_name: str | None = None,
        temperature: float = 0.8,
    ) -> list[dict]:
        # Implementation using local vLLM
        # ... (similar to InferenceEngine but synchronous)
        pass


class ModalInferenceService(InferenceService):
    """Modal-based inference for production."""
    
    def __init__(self, model_id: str = "Qwen/Qwen2.5-7B-Instruct"):
        import modal
        self.engine_cls = modal.Cls.from_name("diplomacy-grpo", "InferenceEngine")
        self.model_id = model_id
    
    async def generate(
        self,
        prompts: list[str],
        valid_moves: list[dict],
        lora_name: str | None = None,
        temperature: float = 0.8,
    ) -> list[dict]:
        engine = self.engine_cls(model_id=self.model_id)
        return await engine.generate.remote.aio(
            prompts=prompts,
            valid_moves=valid_moves,
            lora_name=lora_name,
            temperature=temperature,
        )


class MockInferenceService(InferenceService):
    """Mock service for UI development (random valid moves)."""
    
    async def generate(
        self,
        prompts: list[str],
        valid_moves: list[dict],
        lora_name: str | None = None,
        temperature: float = 0.8,
    ) -> list[dict]:
        import random
        results = []
        for moves_dict in valid_moves:
            orders = []
            for unit, options in moves_dict.items():
                orders.append(random.choice(options))
            results.append({
                "text": "\n".join(orders),
                "token_ids": [],
                "prompt_token_ids": [],
                "completion_logprobs": [],
            })
        return results


def get_inference_service() -> InferenceService:
    """Factory to get appropriate inference service."""
    mode = os.environ.get("INFERENCE_MODE", "mock")
    
    if mode == "local":
        return LocalVLLMService()
    elif mode == "modal":
        return ModalInferenceService()
    else:
        return MockInferenceService()
2.4 Persistence Layer (web/backend/services/persistence.py)
SQLite for MVP with stubs for training data export:
from abc import ABC, abstractmethod
from typing import Any
import json
import sqlite3
from pathlib import Path
from dataclasses import asdict
import time


class PersistenceLayer(ABC):
    """Abstract persistence interface."""
    
    @abstractmethod
    def save_game(self, session_id: str, state: dict) -> None:
        pass
    
    @abstractmethod
    def load_game(self, session_id: str) -> dict | None:
        pass
    
    @abstractmethod
    def save_trajectories(self, trajectories: list[dict]) -> str:
        pass
    
    @abstractmethod
    def export_training_data(self, batch_id: str | None = None) -> list[dict]:
        pass


class SQLitePersistence(PersistenceLayer):
    """SQLite-based persistence for MVP."""
    
    def __init__(self, db_path: str = "web/data/diplomacy.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS games (
                    id TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    created_at REAL,
                    updated_at REAL,
                    user_id TEXT  -- For future auth
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trajectories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    batch_id TEXT NOT NULL,
                    data TEXT NOT NULL,
                    created_at REAL,
                    exported BOOLEAN DEFAULT FALSE
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trajectories_batch 
                ON trajectories(batch_id)
            """)
    
    def save_game(self, session_id: str, state: dict) -> None:
        now = time.time()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO games (id, state, created_at, updated_at)
                VALUES (?, ?, COALESCE(
                    (SELECT created_at FROM games WHERE id = ?), ?
                ), ?)
            """, (session_id, json.dumps(state), session_id, now, now))
    
    def load_game(self, session_id: str) -> dict | None:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT state FROM games WHERE id = ?", (session_id,)
            ).fetchone()
            return json.loads(row[0]) if row else None
    
    def save_trajectories(self, trajectories: list[dict]) -> str:
        """Save trajectories and return batch_id."""
        import uuid
        batch_id = f"web_{uuid.uuid4().hex[:8]}_{int(time.time())}"
        now = time.time()
        
        with sqlite3.connect(self.db_path) as conn:
            for traj in trajectories:
                conn.execute("""
                    INSERT INTO trajectories (batch_id, data, created_at)
                    VALUES (?, ?, ?)
                """, (batch_id, json.dumps(traj), now))
        
        return batch_id
    
    def export_training_data(self, batch_id: str | None = None) -> list[dict]:
        """Export trajectories in training-compatible format."""
        with sqlite3.connect(self.db_path) as conn:
            if batch_id:
                rows = conn.execute(
                    "SELECT data FROM trajectories WHERE batch_id = ?",
                    (batch_id,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT data FROM trajectories WHERE exported = FALSE"
                ).fetchall()
            
            return [json.loads(row[0]) for row in rows]


# Singleton instance
_persistence: PersistenceLayer | None = None

def get_persistence() -> PersistenceLayer:
    global _persistence
    if _persistence is None:
        _persistence = SQLitePersistence()
    return _persistence
2.5 Game Routes (web/backend/routes/game.py)
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Any

from web.backend.services.game_session import GameSession
from web.backend.services.inference_service import get_inference_service
from web.backend.services.persistence import get_persistence
from src.utils.parsing import extract_orders

router = APIRouter()

# In-memory session store (for MVP)
_sessions: dict[str, GameSession] = {}


class NewGameRequest(BaseModel):
    human_power: str = "FRANCE"
    adapter_name: str | None = None
    horizon: int = 10


class SubmitOrdersRequest(BaseModel):
    orders: list[str]


@router.post("/new")
async def create_game(req: NewGameRequest) -> dict[str, Any]:
    """Start a new game."""
    session = GameSession.create(
        human_power=req.human_power,
        adapter_name=req.adapter_name,
        horizon=req.horizon,
    )
    _sessions[session.id] = session
    
    # Persist initial state
    persistence = get_persistence()
    persistence.save_game(session.id, session.game.get_state_json())
    
    return session.get_state()


@router.get("/{game_id}")
async def get_game(game_id: str) -> dict[str, Any]:
    """Get current game state."""
    session = _sessions.get(game_id)
    if not session:
        raise HTTPException(404, "Game not found")
    return session.get_state()


@router.get("/{game_id}/valid-moves")
async def get_valid_moves(game_id: str) -> dict[str, list[str]]:
    """Get valid moves for human player."""
    session = _sessions.get(game_id)
    if not session:
        raise HTTPException(404, "Game not found")
    return session.game.get_valid_moves(session.human_power)


@router.post("/{game_id}/orders")
async def submit_orders(game_id: str, req: SubmitOrdersRequest) -> dict[str, Any]:
    """Submit human orders and process AI moves."""
    session = _sessions.get(game_id)
    if not session:
        raise HTTPException(404, "Game not found")
    
    if session.game.is_done():
        raise HTTPException(400, "Game is already finished")
    
    inference = get_inference_service()
    
    # Get inputs for all AI powers
    inputs = session.game.get_all_inputs(agent=session.agent)
    
    # Separate human and AI powers
    ai_indices = []
    ai_prompts = []
    ai_valid_moves = []
    
    for idx, power in enumerate(inputs["power_names"]):
        if power != session.human_power:
            ai_indices.append(idx)
            ai_prompts.append(inputs["prompts"][idx])
            ai_valid_moves.append(inputs["valid_moves"][idx])
    
    # Get AI responses
    ai_responses = await inference.generate(
        prompts=ai_prompts,
        valid_moves=ai_valid_moves,
        lora_name=session.adapter_name,
    )
    
    # Collect all orders
    all_orders = list(req.orders)  # Human orders
    
    for resp_idx, orig_idx in enumerate(ai_indices):
        power = inputs["power_names"][orig_idx]
        response_data = ai_responses[resp_idx]
        orders = extract_orders(response_data["text"])
        all_orders.extend(orders)
        
        # Collect training data for AI moves
        session.collect_trajectory(
            power=power,
            prompt=inputs["prompts"][orig_idx],
            completion=response_data["text"],
            response_data=response_data,
        )
    
    # Record turn history
    session.turn_history.append({
        "phase": session.game.get_current_phase(),
        "human_orders": req.orders,
        "all_orders": all_orders,
    })
    
    # Execute turn
    session.game.step(all_orders)
    
    # Persist state
    persistence = get_persistence()
    persistence.save_game(session.id, session.game.get_state_json())
    
    # If game is done, finalize and save trajectories
    result = session.get_state()
    if session.game.is_done():
        trajectories = session.finalize_trajectories()
        if trajectories:
            batch_id = persistence.save_trajectories(trajectories)
            result["training_batch_id"] = batch_id
            result["trajectories_collected"] = len(trajectories)
    
    return result


@router.get("/{game_id}/available-adapters")
async def get_available_adapters() -> list[dict[str, str]]:
    """List available bot difficulty levels."""
    return [
        {"id": None, "name": "Base Model", "description": "Untrained Qwen2.5-7B"},
        {"id": "adapter_v18", "name": "Intermediate", "description": "Step 18 checkpoint"},
        {"id": "adapter_v100", "name": "Advanced", "description": "Step 100 checkpoint"},
    ]
2.6 Training Data Export Routes (web/backend/routes/training_data.py)
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from web.backend.services.persistence import get_persistence

router = APIRouter()


@router.get("/export")
async def export_training_data(batch_id: str | None = None):
    """Export trajectories in training-compatible format."""
    persistence = get_persistence()
    trajectories = persistence.export_training_data(batch_id)
    
    return JSONResponse({
        "count": len(trajectories),
        "trajectories": trajectories,
    })


@router.get("/stats")
async def get_training_stats():
    """Get stats about collected training data."""
    # TODO: Implement stats query
    return {"total_trajectories": 0, "total_games": 0}
Phase 3: Frontend (Next.js)
3.1 TypeScript Types (web/frontend/lib/types.ts)
export interface GameState {
  id: string;
  phase: string;
  year: number;
  is_done: boolean;
  human_power: string;
  board_context: BoardContext;
  valid_moves: Record<string, string[]>;
  all_units: Record<string, string[]>;
  all_centers: Record<string, string[]>;
  training_batch_id?: string;
  trajectories_collected?: number;
}

export interface BoardContext {
  my_units: string[];
  my_centers: string[];
  opponent_units: Record<string, string[]>;
  opponent_centers: Record<string, string[]>;
  unowned_centers: string[];
  power_rankings: [string, number][];
  compact_map_view: string;
}

export interface Adapter {
  id: string | null;
  name: string;
  description: string;
}

export type Power = 
  | "AUSTRIA" 
  | "ENGLAND" 
  | "FRANCE" 
  | "GERMANY" 
  | "ITALY" 
  | "RUSSIA" 
  | "TURKEY";
3.2 API Client (web/frontend/lib/api.ts)
const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export const api = {
  async createGame(humanPower: string, adapterName: string | null) {
    const res = await fetch(`${API_BASE}/api/game/new`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ 
        human_power: humanPower, 
        adapter_name: adapterName 
      }),
    });
    return res.json();
  },

  async getGame(gameId: string) {
    const res = await fetch(`${API_BASE}/api/game/${gameId}`);
    return res.json();
  },

  async submitOrders(gameId: string, orders: string[]) {
    const res = await fetch(`${API_BASE}/api/game/${gameId}/orders`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ orders }),
    });
    return res.json();
  },

  async getAvailableAdapters() {
    const res = await fetch(`${API_BASE}/api/game/available-adapters`);
    return res.json();
  },
};
3.3 Game Store (Zustand) (web/frontend/lib/store.ts)
import { create } from 'zustand';
import type { GameState } from './types';

interface GameStore {
  game: GameState | null;
  selectedOrders: Record<string, string>;  // unit -> order
  isSubmitting: boolean;
  
  setGame: (game: GameState) => void;
  selectOrder: (unit: string, order: string) => void;
  clearOrders: () => void;
  setSubmitting: (val: boolean) => void;
}

export const useGameStore = create<GameStore>((set) => ({
  game: null,
  selectedOrders: {},
  isSubmitting: false,
  
  setGame: (game) => set({ game }),
  selectOrder: (unit, order) => 
    set((state) => ({
      selectedOrders: { ...state.selectedOrders, [unit]: order }
    })),
  clearOrders: () => set({ selectedOrders: {} }),
  setSubmitting: (val) => set({ isSubmitting: val }),
}));
3.4 Main Game Page (web/frontend/app/game/[id]/page.tsx)
'use client';

import { useEffect } from 'react';
import { useParams } from 'next/navigation';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '@/lib/api';
import { useGameStore } from '@/lib/store';
import { GameBoard } from '@/components/GameBoard';
import { OrderInput } from '@/components/OrderInput';
import { PhaseDisplay } from '@/components/PhaseDisplay';
import { PowerStatus } from '@/components/PowerStatus';

export default function GamePage() {
  const { id } = useParams<{ id: string }>();
  const queryClient = useQueryClient();
  const { setGame, selectedOrders, clearOrders, setSubmitting } = useGameStore();
  
  const { data: game, isLoading } = useQuery({
    queryKey: ['game', id],
    queryFn: () => api.getGame(id),
    refetchInterval: false,
  });
  
  useEffect(() => {
    if (game) setGame(game);
  }, [game, setGame]);
  
  const submitMutation = useMutation({
    mutationFn: (orders: string[]) => api.submitOrders(id, orders),
    onMutate: () => setSubmitting(true),
    onSuccess: (newState) => {
      setGame(newState);
      clearOrders();
      queryClient.setQueryData(['game', id], newState);
    },
    onSettled: () => setSubmitting(false),
  });
  
  const handleSubmit = () => {
    const orders = Object.values(selectedOrders);
    submitMutation.mutate(orders);
  };
  
  if (isLoading || !game) {
    return <div className="p-8">Loading...</div>;
  }
  
  return (
    <div className="min-h-screen bg-gray-900 text-white">
      <header className="bg-gray-800 p-4">
        <PhaseDisplay phase={game.phase} year={game.year} />
      </header>
      
      <main className="flex gap-4 p-4">
        <div className="flex-1">
          <GameBoard 
            units={game.all_units}
            centers={game.all_centers}
            humanPower={game.human_power}
          />
        </div>
        
        <aside className="w-80 space-y-4">
          <PowerStatus 
            rankings={game.board_context.power_rankings}
            humanPower={game.human_power}
          />
          
          {!game.is_done && (
            <OrderInput 
              validMoves={game.valid_moves}
              onSubmit={handleSubmit}
            />
          )}
          
          {game.is_done && (
            <div className="bg-green-800 p-4 rounded">
              <h3 className="font-bold">Game Complete!</h3>
              {game.trajectories_collected && (
                <p className="text-sm mt-2">
                  Collected {game.trajectories_collected} training samples
                </p>
              )}
            </div>
          )}
        </aside>
      </main>
    </div>
  );
}
3.5 Order Input Component (web/frontend/components/OrderInput.tsx)
'use client';

import { useGameStore } from '@/lib/store';

interface Props {
  validMoves: Record<string, string[]>;
  onSubmit: () => void;
}

export function OrderInput({ validMoves, onSubmit }: Props) {
  const { selectedOrders, selectOrder, isSubmitting } = useGameStore();
  
  const allUnitsHaveOrders = Object.keys(validMoves).every(
    unit => selectedOrders[unit]
  );
  
  return (
    <div className="bg-gray-800 p-4 rounded space-y-4">
      <h3 className="font-bold text-lg">Your Orders</h3>
      
      {Object.entries(validMoves).map(([unit, options]) => (
        <div key={unit} className="space-y-1">
          <label className="text-sm font-medium text-gray-300">{unit}</label>
          <select
            value={selectedOrders[unit] || ''}
            onChange={(e) => selectOrder(unit, e.target.value)}
            className="w-full bg-gray-700 rounded p-2 text-sm"
          >
            <option value="">Select order...</option>
            {options.map(order => (
              <option key={order} value={order}>
                {order}
              </option>
            ))}
          </select>
        </div>
      ))}
      
      <button
        onClick={onSubmit}
        disabled={!allUnitsHaveOrders || isSubmitting}
        className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 
                   py-2 rounded font-medium transition-colors"
      >
        {isSubmitting ? 'Processing...' : 'Submit Orders'}
      </button>
    </div>
  );
}
Phase 4: Local Development Setup
4.1 Docker Compose (web/docker-compose.yml)
version: '3.8'

services:
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile.dev
    ports:
      - "3000:3000"
    volumes:
      - ./frontend:/app
      - /app/node_modules
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:8000
    depends_on:
      - backend

  backend:
    build:
      context: ..
      dockerfile: web/backend/Dockerfile.dev
    ports:
      - "8000:8000"
    volumes:
      - ..:/app
    environment:
      - INFERENCE_MODE=mock  # or 'local' or 'modal'
      - PYTHONPATH=/app
    command: uvicorn web.backend.server:app --host 0.0.0.0 --port 8000 --reload
4.2 Run Scripts
Add to root package.json (create if doesn't exist):
{
  "scripts": {
    "web:dev": "cd web && docker-compose up",
    "web:frontend": "cd web/frontend && npm run dev",
    "web:backend": "INFERENCE_MODE=mock uvicorn web.backend.server:app --reload --port 8000"
  }
}
Or shell scripts:
# scripts/run_web_dev.sh
#!/bin/bash
# Start both frontend and backend for local development

# Terminal 1: Backend
echo "Starting backend..."
INFERENCE_MODE=mock PYTHONPATH=. uvicorn web.backend.server:app --reload --port 8000 &
BACKEND_PID=$!

# Terminal 2: Frontend  
echo "Starting frontend..."
cd web/frontend && npm run dev &
FRONTEND_PID=$!

# Cleanup on exit
trap "kill $BACKEND_PID $FRONTEND_PID" EXIT
wait
4.3 Local vLLM Setup
For running local inference (when INFERENCE_MODE=local):
# Install vLLM locally (requires CUDA)
pip install vllm

# Run with smaller model for testing
INFERENCE_MODE=local MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct python -m web.backend.server
Phase 5: Refactoring for Code Reuse
5.1 Extract Game Loop from Rollouts
Create src/engine/game_runner.py to share logic between rollouts and web:
"""Shared game execution logic for rollouts and web app."""

from dataclasses import dataclass
from typing import Protocol, Any

from src.engine.wrapper import DiplomacyWrapper
from src.agents.llm_agent import LLMAgent
from src.utils.parsing import extract_orders


class InferenceProvider(Protocol):
    """Protocol for inference providers (Modal, local vLLM, mock)."""
    
    async def generate(
        self,
        prompts: list[str],
        valid_moves: list[dict],
        lora_name: str | None = None,
        temperature: float = 0.8,
    ) -> list[dict]:
        ...


@dataclass
class TurnResult:
    """Result of processing a game turn."""
    all_orders: list[str]
    power_data: dict[str, dict]  # power -> {prompt, completion, response_data}


async def process_ai_turn(
    game: DiplomacyWrapper,
    agent: LLMAgent,
    inference: InferenceProvider,
    exclude_powers: set[str] | None = None,
    adapter_map: dict[str, str | None] | None = None,
    temperature: float = 0.8,
) -> TurnResult:
    """
    Process a turn for AI-controlled powers.
    
    Args:
        game: Current game state
        agent: LLM agent for prompt building
        inference: Inference provider
        exclude_powers: Powers to skip (e.g., human player)
        adapter_map: Optional per-power adapter mapping
        temperature: Sampling temperature
    
    Returns:
        TurnResult with orders and data for training collection
    """
    exclude_powers = exclude_powers or set()
    adapter_map = adapter_map or {}
    
    inputs = game.get_all_inputs(agent=agent)
    
    # Filter to AI powers only
    ai_data = []
    for idx, power in enumerate(inputs["power_names"]):
        if power not in exclude_powers:
            ai_data.append({
                "idx": idx,
                "power": power,
                "prompt": inputs["prompts"][idx],
                "valid_moves": inputs["valid_moves"][idx],
                "adapter": adapter_map.get(power),
            })
    
    # Group by adapter for efficient batching
    # (simplified - full implementation would batch like rollouts)
    all_orders = []
    power_data = {}
    
    if ai_data:
        responses = await inference.generate(
            prompts=[d["prompt"] for d in ai_data],
            valid_moves=[d["valid_moves"] for d in ai_data],
            lora_name=ai_data[0]["adapter"],  # Simplified
            temperature=temperature,
        )
        
        for data, response in zip(ai_data, responses):
            orders = extract_orders(response["text"])
            all_orders.extend(orders)
            power_data[data["power"]] = {
                "prompt": data["prompt"],
                "completion": response["text"],
                "response_data": response,
            }
    
    return TurnResult(all_orders=all_orders, power_data=power_data)
5.2 Update Rollouts to Use Shared Code
The existing rollouts can optionally be refactored to use process_ai_turn, but this is lower priority since they already work.
Summary: Implementation Order
Week 1: Backend Foundation
Create directory structure
Implement GameSession, InferenceService, PersistenceLayer
FastAPI routes for game CRUD
Mock inference for testing
Week 2: Frontend MVP
Next.js setup with Tailwind
Game state management (Zustand)
Basic order input UI
Connect to backend API
Week 3: Game Board Visualization
SVG-based map rendering (or leverage existing HTML visualizer)
Unit display and click interactions
Phase transitions
Week 4: Local vLLM Integration
Local inference service implementation
Test with real model
Performance optimization
Week 5: Training Data & Polish
Training data export endpoint
Integration test with trainer
UI polish and error handling

---

## Implementation Progress

### ✅ Phase 1: Infrastructure Setup - COMPLETE
- [x] Created `web/backend/` directory structure with routes, services, models
- [x] Added `web/backend/requirements.txt` with FastAPI dependencies
- [x] Next.js frontend already set up with shadcn/ui

### ✅ Phase 2: Backend API - COMPLETE
- [x] `web/backend/server.py` - FastAPI app with CORS, routers
- [x] `web/backend/services/game_session.py` - GameSession class wrapping DiplomacyWrapper
- [x] `web/backend/services/inference_service.py` - Mock/Local/Modal inference abstraction
- [x] `web/backend/services/persistence.py` - SQLite persistence layer with trajectory storage
- [x] `web/backend/routes/game.py` - Game CRUD and order submission endpoints
- [x] `web/backend/routes/training_data.py` - Training data export and stats endpoints
- [x] `web/backend/models/schemas.py` - Pydantic models for API

### ✅ Phase 3: Frontend - COMPLETE
- [x] `web/frontend/lib/types.ts` - TypeScript interfaces for game state
- [x] `web/frontend/lib/api.ts` - API client with error handling
- [x] `web/frontend/lib/store.ts` - Zustand stores for game state and game creation
- [x] `web/frontend/lib/providers.tsx` - React Query provider
- [x] `web/frontend/app/page.tsx` - Home page with power/adapter selection
- [x] `web/frontend/app/game/[id]/page.tsx` - Game page with order submission
- [x] `web/frontend/components/PhaseDisplay.tsx` - Phase/season display
- [x] `web/frontend/components/PowerStatus.tsx` - Power rankings sidebar
- [x] `web/frontend/components/OrderInput.tsx` - Order selection UI with unit dropdown
- [x] `web/frontend/components/GameBoard.tsx` - Text-based compact map view

### ✅ Phase 4: Local Development Setup - COMPLETE
- [x] `scripts/run_web_backend.sh` - Backend startup script
- [x] `scripts/run_web_frontend.sh` - Frontend startup script
- [x] `web/README.md` - Documentation with run instructions

### 🔄 Remaining Work
- [ ] Test end-to-end flow with backend running
- [ ] Test local vLLM inference (INFERENCE_MODE=local)
- [ ] SVG-based map visualization (currently using text-based compact view)
- [ ] Unit click interactions on map

### How to Run

**Backend (from project root):**
```bash
source .venv/bin/activate
./scripts/run_web_backend.sh
# Or with specific inference mode:
INFERENCE_MODE=mock PYTHONPATH=. uvicorn web.backend.server:app --reload --port 8000
```

**Frontend (from project root):**
```bash
./scripts/run_web_frontend.sh
# Or manually:
cd web/frontend && npm run dev
```

Then open http://localhost:3000 in your browser.