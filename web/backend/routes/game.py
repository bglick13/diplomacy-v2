"""Game API routes."""

from typing import Any

from fastapi import APIRouter, HTTPException

from src.utils.parsing import extract_orders
from web.backend.models.schemas import AdapterInfo, NewGameRequest, SubmitOrdersRequest
from web.backend.services.game_session import (
    POWERS,
    GameSession,
    get_baseline_bot,
    is_baseline_bot,
)
from web.backend.services.inference_service import get_inference_service
from web.backend.services.persistence import get_persistence

router = APIRouter()

# In-memory session store with persistence backup
# Sessions are cached in memory but can be restored from SQLite
_sessions: dict[str, GameSession] = {}


def _get_session(game_id: str) -> GameSession:
    """Get a game session, restoring from persistence if needed."""
    # Check in-memory cache first
    session = _sessions.get(game_id)
    if session:
        return session

    # Try to restore from persistence
    persistence = get_persistence()
    saved_data = persistence.load_game(game_id)
    if saved_data and "game_state" in saved_data:
        try:
            session = GameSession.from_dict(saved_data)
            _sessions[game_id] = session  # Cache for future requests
            return session
        except Exception as e:
            print(f"Failed to restore session {game_id}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to restore game session: {e}",
            ) from e

    raise HTTPException(status_code=404, detail="Game not found")


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

    # Persist full session state (for restoration after server restart)
    persistence = get_persistence()
    persistence.save_game(session.id, session.to_dict())

    return session.get_state()


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
    2. Gets AI moves from the inference service (LLM) or baseline bots (rule-based)
    3. Executes all orders
    4. Collects training data (LLM moves only)
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

    # Get inputs for all powers
    inputs = session.game.get_all_inputs(agent=session.agent)

    # Separate AI powers into LLM-based vs baseline bots
    llm_indices = []
    llm_prompts = []
    llm_valid_moves = []
    bot_powers: list[tuple[str, str, int]] = []  # (power, adapter, orig_idx)

    for idx, power in enumerate(inputs["power_names"]):
        if power == session.human_power:
            continue

        adapter = session.get_adapter_for_power(power)
        if is_baseline_bot(adapter):
            bot_powers.append((power, adapter, idx))  # type: ignore[arg-type]
        else:
            llm_indices.append(idx)
            llm_prompts.append(inputs["prompts"][idx])
            llm_valid_moves.append(inputs["valid_moves"][idx])

    # Collect all orders: human orders first
    all_orders = list(req.orders)

    # Get baseline bot orders directly (no inference needed)
    for power, adapter, _idx in bot_powers:
        bot = get_baseline_bot(adapter)
        orders = bot.get_orders(session.game, power)
        all_orders.extend(orders)

    # Get LLM responses (if there are LLM-based AI powers)
    if llm_prompts:
        inference = get_inference_service()
        llm_responses = await inference.generate(
            prompts=llm_prompts,
            valid_moves=llm_valid_moves,
            lora_name=session.adapter_name,
        )

        # Process LLM responses
        for resp_idx, orig_idx in enumerate(llm_indices):
            power = inputs["power_names"][orig_idx]
            response_data = llm_responses[resp_idx]
            orders = extract_orders(response_data["text"])
            all_orders.extend(orders)

            # Collect training data for LLM moves only
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

    # Persist full session state (for restoration after server restart)
    persistence = get_persistence()
    persistence.save_game(session.id, session.to_dict())

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
        # LLM-based opponents (require inference)
        AdapterInfo(
            id=None,
            name="Base Model",
            description="Untrained Qwen2.5-7B - easiest LLM difficulty",
        ),
        AdapterInfo(
            id="grpo-20251222-191408/adapter_v150",
            name="Best (v150)",
            description="Peak Elo checkpoint - 1068 Elo",
        ),
        AdapterInfo(
            id="grpo-20251222-191408/adapter_v240",
            name="Final",
            description="Final training checkpoint",
        ),
        # Rule-based baseline bots (no inference needed, instant response)
        AdapterInfo(
            id="bot:random",
            name="Random Bot",
            description="Picks random valid moves - very easy",
        ),
        AdapterInfo(
            id="bot:chaos",
            name="Chaos Bot",
            description="Aggressive - prioritizes movement over holds",
        ),
        AdapterInfo(
            id="bot:defensive",
            name="Defensive Bot",
            description="Cautious - prioritizes holds and supports",
        ),
        AdapterInfo(
            id="bot:territorial",
            name="Territorial Bot",
            description="Greedy - targets neutral supply centers",
        ),
        AdapterInfo(
            id="bot:coordinated",
            name="Coordinated Bot",
            description="Team play - coordinates supports between units",
        ),
    ]
