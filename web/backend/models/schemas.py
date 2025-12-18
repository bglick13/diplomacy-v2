"""Pydantic models for API requests and responses."""

from pydantic import BaseModel


# Request models
class NewGameRequest(BaseModel):
    """Request to create a new game."""

    human_power: str = "FRANCE"
    adapter_name: str | None = None
    horizon: int = 10


class SubmitOrdersRequest(BaseModel):
    """Request to submit orders for the current phase."""

    orders: list[str]


# Response models
class AdapterInfo(BaseModel):
    """Information about an available adapter/bot difficulty."""

    id: str | None
    name: str
    description: str


class BoardContext(BaseModel):
    """Board context information for strategic decision making."""

    my_units: list[str]
    my_centers: list[str]
    opponent_units: dict[str, list[str]]
    opponent_centers: dict[str, list[str]]
    unowned_centers: list[str]
    power_rankings: list[tuple[str, int]]
    compact_map_view: str


class GameStateResponse(BaseModel):
    """Full game state response."""

    id: str
    phase: str
    year: int
    is_done: bool
    human_power: str
    board_context: BoardContext
    valid_moves: dict[str, list[str]]
    all_units: dict[str, list[str]]
    all_centers: dict[str, list[str]]
    training_batch_id: str | None = None
    trajectories_collected: int | None = None

    class Config:
        from_attributes = True
