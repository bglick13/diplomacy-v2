"""
Type definitions for the League Training system.

This module contains shared types used across league training and evaluation.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class OpponentType(str, Enum):
    """Types of opponents for training and evaluation."""

    # Baselines (no LLM inference)
    RANDOM = "random"
    CHAOS = "chaos"

    # Model-based opponents
    BASE_MODEL = "base_model"  # No LoRA adapter
    CHECKPOINT = "checkpoint"  # Historical checkpoint with LoRA
    SELF = "self"  # Current policy (pure self-play)


# Mapping from OpponentType to agent registry names
OPPONENT_TO_AGENT_NAME: dict[OpponentType, str] = {
    OpponentType.RANDOM: "random_bot",
    OpponentType.CHAOS: "chaos_bot",
    OpponentType.BASE_MODEL: "base_model",
}


class AgentType(str, Enum):
    """Classification of agents in the league."""

    BASELINE = "baseline"  # RandomBot, ChaosBot, base model
    CHECKPOINT = "checkpoint"  # Trained LoRA adapter


@dataclass
class AgentInfo:
    """
    Information about an agent in the league.

    Attributes:
        name: Unique identifier (e.g., "random_bot", "adapter_v50")
        agent_type: Whether this is a baseline or checkpoint
        elo: Current Elo rating
        matches: Number of matches played for Elo calculation
        path: Path to LoRA adapter (None for baselines)
        step: Training step when checkpoint was created (None for baselines)
        created_at: ISO timestamp of creation
        parent: Name of parent checkpoint (for lineage tracking)
    """

    name: str
    agent_type: AgentType
    elo: float = 1000.0
    matches: int = 0
    path: str | None = None
    step: int | None = None
    created_at: str | None = None
    parent: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return {
            "type": self.agent_type.value,
            "elo": self.elo,
            "matches": self.matches,
            "path": self.path,
            "step": self.step,
            "created_at": self.created_at,
            "parent": self.parent,
        }

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> "AgentInfo":
        """Deserialize from dictionary."""
        return cls(
            name=name,
            agent_type=AgentType(data["type"]),
            elo=data.get("elo", 1000.0),
            matches=data.get("matches", 0),
            path=data.get("path"),
            step=data.get("step"),
            created_at=data.get("created_at"),
            parent=data.get("parent"),
        )

    @classmethod
    def create_baseline(cls, name: str, elo: float = 1000.0) -> "AgentInfo":
        """Create a baseline agent (no LoRA)."""
        return cls(
            name=name,
            agent_type=AgentType.BASELINE,
            elo=elo,
            path=None,
        )

    @classmethod
    def create_checkpoint(
        cls,
        name: str,
        path: str,
        step: int,
        parent: str | None = None,
        elo: float = 1000.0,
    ) -> "AgentInfo":
        """Create a checkpoint agent (LoRA adapter)."""
        return cls(
            name=name,
            agent_type=AgentType.CHECKPOINT,
            elo=elo,
            path=path,
            step=step,
            created_at=datetime.now(UTC).isoformat(),
            parent=parent,
        )


@dataclass
class MatchResult:
    """
    Result of a single game/match between agents.

    Attributes:
        game_id: Unique game identifier
        step: Training step when match occurred
        power_agents: Mapping of power name to agent name
        scores: Final scores for each power
        rankings: Final ranking (1=best) for each power
        num_years: Number of years the game lasted
        winner: Agent name of the winner (or None if draw)
    """

    game_id: str
    step: int
    power_agents: dict[str, str]  # {"FRANCE": "adapter_v50", "ENGLAND": "random_bot"}
    scores: dict[str, float]  # {"FRANCE": 12.5, "ENGLAND": 3.0}
    rankings: dict[str, int]  # {"FRANCE": 1, "ENGLAND": 5}
    num_years: int
    winner: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return {
            "game_id": self.game_id,
            "step": self.step,
            "power_agents": self.power_agents,
            "scores": self.scores,
            "rankings": self.rankings,
            "num_years": self.num_years,
            "winner": self.winner,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MatchResult":
        """Deserialize from dictionary."""
        return cls(
            game_id=data["game_id"],
            step=data["step"],
            power_agents=data["power_agents"],
            scores=data["scores"],
            rankings=data["rankings"],
            num_years=data["num_years"],
            winner=data.get("winner"),
        )


@dataclass
class LeagueMetadata:
    """
    Metadata about the league state.

    Attributes:
        run_name: Name of the training run
        best_elo: Highest Elo achieved by any checkpoint
        best_agent: Name of the agent with highest Elo
        latest_step: Most recent training step with a checkpoint
        total_matches: Total number of matches recorded
    """

    run_name: str
    best_elo: float = 1000.0
    best_agent: str = "base_model"
    latest_step: int = 0
    total_matches: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "run_name": self.run_name,
            "best_elo": self.best_elo,
            "best_agent": self.best_agent,
            "latest_step": self.latest_step,
            "total_matches": self.total_matches,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LeagueMetadata":
        """Deserialize from dictionary."""
        return cls(
            run_name=data["run_name"],
            best_elo=data.get("best_elo", 1000.0),
            best_agent=data.get("best_agent", "base_model"),
            latest_step=data.get("latest_step", 0),
            total_matches=data.get("total_matches", 0),
        )


# Default baseline agents to initialize the league with
# Removed random_bot (too exploitable at Elo 800)
DEFAULT_BASELINES: list[AgentInfo] = [
    AgentInfo.create_baseline("base_model", elo=1000.0),
    AgentInfo.create_baseline("chaos_bot", elo=900.0),
    AgentInfo.create_baseline("defensive_bot", elo=950.0),
    AgentInfo.create_baseline("territorial_bot", elo=950.0),
    AgentInfo.create_baseline("coordinated_bot", elo=1000.0),
]


def opponent_type_to_agent_name(opponent_type: OpponentType) -> str:
    """Convert OpponentType enum to agent registry name."""
    if opponent_type in OPPONENT_TO_AGENT_NAME:
        return OPPONENT_TO_AGENT_NAME[opponent_type]
    raise ValueError(f"Cannot convert {opponent_type} to agent name")
