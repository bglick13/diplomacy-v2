"""Game session management for human vs AI gameplay."""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from src.agents.base import DiplomacyAgent
from src.agents.baselines import (
    ChaosBot,
    CoordinatedBot,
    DefensiveBot,
    RandomBot,
    TerritorialBot,
)
from src.agents.llm_agent import LLMAgent, PromptConfig
from src.engine.wrapper import DiplomacyWrapper
from src.utils.scoring import calculate_final_scores

POWERS = ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]

# Mapping from bot: prefixed adapter names to bot classes
BOT_MAP: dict[str, type[DiplomacyAgent]] = {
    "bot:random": RandomBot,
    "bot:chaos": ChaosBot,
    "bot:defensive": DefensiveBot,
    "bot:territorial": TerritorialBot,
    "bot:coordinated": CoordinatedBot,
}


def is_baseline_bot(adapter_name: str | None) -> bool:
    """Check if adapter name refers to a rule-based baseline bot."""
    return adapter_name is not None and adapter_name.startswith("bot:")


def get_baseline_bot(adapter_name: str) -> DiplomacyAgent:
    """Instantiate a baseline bot from its adapter name."""
    if adapter_name not in BOT_MAP:
        raise ValueError(f"Unknown baseline bot: {adapter_name}")
    return BOT_MAP[adapter_name]()


@dataclass
class GameSession:
    """Manages a single human-vs-AI Diplomacy game."""

    id: str
    game: DiplomacyWrapper
    agent: LLMAgent
    human_power: str
    adapter_name: str | None
    created_at: float

    # Per-power adapter mapping for league-style games
    power_adapters: dict[str, str | None] = field(default_factory=dict)

    # Training data collection
    trajectories: list[dict] = field(default_factory=list)
    turn_history: list[dict] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        human_power: str = "FRANCE",
        adapter_name: str | None = None,
        horizon: int = 10,
        power_adapters: dict[str, str | None] | None = None,
    ) -> "GameSession":
        """Create a new game session.

        Args:
            human_power: The power controlled by the human player.
            adapter_name: Default adapter for all AI powers (can be overridden by power_adapters).
            horizon: Maximum number of years to play.
            power_adapters: Optional per-power adapter mapping for league-style games.
        """
        session_id = str(uuid.uuid4())[:8]
        game = DiplomacyWrapper(game_id=session_id, horizon=horizon)
        # Match training config: show_valid_moves=False to include board context
        agent = LLMAgent(
            config=PromptConfig(
                compact_mode=True,
                show_valid_moves=False,
                show_board_context=True,
                show_map_windows=True,
            )
        )

        # Build power_adapters map
        if power_adapters is None:
            # All AI powers use the same adapter
            power_adapters = {p: adapter_name for p in POWERS if p != human_power}
        else:
            # Ensure human power isn't in the map
            power_adapters = {p: a for p, a in power_adapters.items() if p != human_power}

        return cls(
            id=session_id,
            game=game,
            agent=agent,
            human_power=human_power,
            adapter_name=adapter_name,
            power_adapters=power_adapters,
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
        """Get all units for all powers."""
        return {power: list(obj.units) for power, obj in self.game.game.powers.items()}

    def _get_all_centers(self) -> dict[str, list[str]]:
        """Get all supply centers for all powers."""
        return {power: list(obj.centers) for power, obj in self.game.game.powers.items()}

    def get_ai_powers(self) -> list[str]:
        """Get list of AI-controlled powers."""
        return [p for p in POWERS if p != self.human_power]

    def get_adapter_for_power(self, power: str) -> str | None:
        """Get the adapter name for a specific power."""
        return self.power_adapters.get(power, self.adapter_name)

    def collect_trajectory(
        self,
        power: str,
        prompt: str,
        completion: str,
        response_data: dict,
    ) -> None:
        """Collect training data for a turn.

        Args:
            power: The power this trajectory is for.
            prompt: The prompt sent to the model.
            completion: The model's response.
            response_data: Full response data including token IDs and logprobs.
        """
        self.trajectories.append(
            {
                "prompt": prompt,
                "completion": completion,
                "prompt_token_ids": response_data.get("prompt_token_ids", []),
                "completion_token_ids": response_data.get("token_ids", []),
                "completion_logprobs": response_data.get("completion_logprobs", []),
                "group_id": f"{self.id}_{power}_{self.game.get_year()}",
                "power": power,
                "phase": self.game.get_current_phase(),
                # Reward computed at game end
            }
        )

    def record_turn(
        self,
        phase: str,
        human_orders: list[str],
        all_orders: list[str],
    ) -> None:
        """Record turn history for debugging and replay."""
        self.turn_history.append(
            {
                "phase": phase,
                "human_orders": human_orders,
                "all_orders": all_orders,
                "timestamp": time.time(),
            }
        )

    def finalize_trajectories(self) -> list[dict]:
        """Add rewards to trajectories based on final scores.

        Should be called when the game is done.

        Returns:
            List of trajectories with rewards added.
        """
        if not self.trajectories:
            return []

        final_scores = calculate_final_scores(self.game)

        for traj in self.trajectories:
            power = traj.pop("power")  # Remove internal field
            traj["reward"] = final_scores.get(power, 0.0)

        return self.trajectories

    def to_dict(self) -> dict[str, Any]:
        """Serialize session for persistence."""
        return {
            "id": self.id,
            "game_state": self.game.get_state_json(),
            "human_power": self.human_power,
            "adapter_name": self.adapter_name,
            "power_adapters": self.power_adapters,
            "created_at": self.created_at,
            "turn_history": self.turn_history,
            # Note: trajectories are stored separately
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GameSession":
        """Restore a session from persisted data.

        Args:
            data: Dict from to_dict() or persistence layer

        Returns:
            Restored GameSession instance
        """
        # Restore the game engine from saved state
        game = DiplomacyWrapper.from_saved_state(data["game_state"], horizon=10)
        # Match training config: show_valid_moves=False to include board context
        agent = LLMAgent(
            config=PromptConfig(
                compact_mode=True,
                show_valid_moves=False,
                show_board_context=True,
                show_map_windows=True,
            )
        )

        return cls(
            id=data["id"],
            game=game,
            agent=agent,
            human_power=data["human_power"],
            adapter_name=data.get("adapter_name"),
            power_adapters=data.get("power_adapters", {}),
            created_at=data.get("created_at", time.time()),
            turn_history=data.get("turn_history", []),
        )
