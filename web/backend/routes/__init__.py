"""API routes for the Diplomacy web backend."""

from web.backend.routes.game import router as game_router
from web.backend.routes.training_data import router as training_router

__all__ = ["game_router", "training_router"]
