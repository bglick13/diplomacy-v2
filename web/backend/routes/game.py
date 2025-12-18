"""Game API routes."""

from typing import Any

from fastapi import APIRouter, HTTPException

from src.utils.parsing import extract_orders
from web.backend.models.schemas import AdapterInfo, NewGameRequest, SubmitOrdersRequest
from web.backend.services.game_session import POWERS, GameSession
from web.backend.services.inference_service import get_inference_service
from web.backend.services.persistence import get_persistence

router = APIRouter()

# In-memory session store (for MVP)
# In production, this would be backed by the persistence layer
_sessions: dict[str, GameSession] = {}


def _get_session(game_id: str) -> GameSession:
    """Get a game session or raise 404."""
    session = _sessions.get(game_id)
    if not session:
        raise HTTPException(status_code=404, detail="Game not found")
    return session


@router.post("/new")
async def create_game(req: NewGameRequest) -> dict[str, Any]:
    """Start a new game.

    Args:
        req: New game request with human_power, adapter_name, and horizon.

    Returns:
        Initial game state.
    """
    # Validate human_power
    if req.human_power not in POWERS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid power. Must be one of: {POWERS}",
        )

    session = GameSession.create(
        human_power=req.human_power,
        adapter_name=req.adapter_name,
        horizon=req.horizon,
    )
    _sessions[session.id] = session

    # Persist initial state
    persistence = get_persistence()
    state = session.get_state()
    state["adapter_name"] = session.adapter_name
    persistence.save_game(session.id, state)

    return state


@router.get("/{game_id}")
async def get_game(game_id: str) -> dict[str, Any]:
    """Get current game state."""
    session = _get_session(game_id)
    return session.get_state()


@router.get("/{game_id}/valid-moves")
async def get_valid_moves(game_id: str) -> dict[str, list[str]]:
    """Get valid moves for human player."""
    session = _get_session(game_id)
    return session.game.get_valid_moves(session.human_power)


@router.post("/{game_id}/orders")
async def submit_orders(game_id: str, req: SubmitOrdersRequest) -> dict[str, Any]:
    """Submit human orders and process AI moves.

    This endpoint:
    1. Validates human orders
    2. Gets AI moves from the inference service
    3. Executes all orders
    4. Collects training data
    5. Returns the new game state

    Args:
        game_id: The game session ID.
        req: The orders to submit.

    Returns:
        Updated game state, including training batch info if game is done.
    """
    session = _get_session(game_id)

    if session.game.is_done():
        raise HTTPException(status_code=400, detail="Game is already finished")

    # Get current phase for logging
    phase = session.game.get_current_phase()

    # Get inference service
    inference = get_inference_service()

    # Get inputs for all powers
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

    # Get AI responses (if there are AI powers with valid moves)
    ai_responses = []
    if ai_prompts:
        ai_responses = await inference.generate(
            prompts=ai_prompts,
            valid_moves=ai_valid_moves,
            lora_name=session.adapter_name,
        )

    # Collect all orders: human orders first
    all_orders = list(req.orders)

    # Process AI responses
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
    session.record_turn(
        phase=phase,
        human_orders=req.orders,
        all_orders=all_orders,
    )

    # Execute turn
    session.game.step(all_orders)

    # Persist state
    persistence = get_persistence()
    state = session.get_state()
    state["adapter_name"] = session.adapter_name
    persistence.save_game(session.id, state)

    # If game is done, finalize and save trajectories
    result = session.get_state()
    if session.game.is_done():
        trajectories = session.finalize_trajectories()
        if trajectories:
            batch_id = persistence.save_trajectories(trajectories, session.id)
            result["training_batch_id"] = batch_id
            result["trajectories_collected"] = len(trajectories)

    return result


@router.delete("/{game_id}")
async def delete_game(game_id: str) -> dict[str, str]:
    """Delete a game session."""
    if game_id in _sessions:
        del _sessions[game_id]

    persistence = get_persistence()
    deleted = persistence.delete_game(game_id)

    if deleted:
        return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="Game not found")


@router.get("/list/all")
async def list_games() -> list[dict]:
    """List all games."""
    persistence = get_persistence()
    return persistence.list_games()


@router.get("/config/powers")
async def get_powers() -> list[str]:
    """Get list of all playable powers."""
    return POWERS


@router.get("/config/adapters")
async def get_available_adapters() -> list[AdapterInfo]:
    """List available bot difficulty levels / adapters."""
    return [
        AdapterInfo(
            id=None,
            name="Base Model",
            description="Untrained Qwen2.5-7B - easiest difficulty",
        ),
        AdapterInfo(
            id="adapter_v18",
            name="Intermediate",
            description="Step 18 checkpoint - medium difficulty",
        ),
        AdapterInfo(
            id="adapter_v100",
            name="Advanced",
            description="Step 100 checkpoint - hardest difficulty",
        ),
    ]
