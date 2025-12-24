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


# TrueSkill default parameters (imported here for backward compat conversion)
MU_INIT = 25.0
SIGMA_INIT = 25.0 / 3  # ~8.333


@dataclass
class AgentInfo:
    """
    Information about an agent in the league.

    Attributes:
        name: Unique identifier (e.g., "random_bot", "adapter_v50")
        agent_type: Whether this is a baseline or checkpoint
        elo: Current Elo rating (DEPRECATED - kept for backward compat)
        mu: TrueSkill skill mean (higher = better)
        sigma: TrueSkill skill uncertainty (lower = more confident)
        matches: Number of matches played for rating calculation
        path: Path to LoRA adapter (None for baselines)
        step: Training step when checkpoint was created (None for baselines)
        created_at: ISO timestamp of creation
        parent: Name of parent checkpoint (for lineage tracking)
    """

    name: str
    agent_type: AgentType
    elo: float = 1000.0  # DEPRECATED - kept for backward compatibility
    mu: float = MU_INIT  # TrueSkill mean
    sigma: float = SIGMA_INIT  # TrueSkill uncertainty
    matches: int = 0
    path: str | None = None
    step: int | None = None
    created_at: str | None = None
    parent: str | None = None

    @property
    def display_rating(self) -> float:
        """Conservative rating estimate (mu - 3*sigma, 99.7% confidence lower bound)."""
        return self.mu - 3 * self.sigma

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return {
            "type": self.agent_type.value,
            "elo": self.elo,
            "mu": self.mu,
            "sigma": self.sigma,
            "matches": self.matches,
            "path": self.path,
            "step": self.step,
            "created_at": self.created_at,
            "parent": self.parent,
        }

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> "AgentInfo":
        """Deserialize from dictionary.

        Handles backward compatibility: if mu/sigma are missing, converts from Elo.
        """
        # Check if this is an old registry without TrueSkill fields
        if "mu" not in data:
            # Convert Elo to approximate TrueSkill
            elo = data.get("elo", 1000.0)
            mu = (elo - 1000) / 40 + MU_INIT
            sigma = SIGMA_INIT  # Full uncertainty for converted ratings
        else:
            mu = data["mu"]
            sigma = data["sigma"]

        return cls(
            name=name,
            agent_type=AgentType(data["type"]),
            elo=data.get("elo", 1000.0),
            mu=mu,
            sigma=sigma,
            matches=data.get("matches", 0),
            path=data.get("path"),
            step=data.get("step"),
            created_at=data.get("created_at"),
            parent=data.get("parent"),
        )

    @classmethod
    def create_baseline(
        cls,
        name: str,
        elo: float = 1000.0,
        mu: float | None = None,
        sigma: float = SIGMA_INIT,
    ) -> "AgentInfo":
        """Create a baseline agent (no LoRA)."""
        # Convert elo to mu if mu not specified
        if mu is None:
            mu = (elo - 1000) / 40 + MU_INIT
        return cls(
            name=name,
            agent_type=AgentType.BASELINE,
            elo=elo,
            mu=mu,
            sigma=sigma,
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
        mu: float | None = None,
        sigma: float | None = None,
    ) -> "AgentInfo":
        """Create a checkpoint agent (LoRA adapter)."""
        # Default mu from elo conversion
        if mu is None:
            mu = (elo - 1000) / 40 + MU_INIT
        # Default sigma - high for new checkpoints (uncertain)
        if sigma is None:
            sigma = SIGMA_INIT
        return cls(
            name=name,
            agent_type=AgentType.CHECKPOINT,
            elo=elo,
            mu=mu,
            sigma=sigma,
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
        best_elo: Highest Elo achieved by any checkpoint (DEPRECATED)
        best_mu: Highest TrueSkill mu achieved by any checkpoint
        best_display_rating: Highest display rating (mu - 3*sigma)
        best_agent: Name of the agent with highest rating
        latest_step: Most recent training step with a checkpoint
        total_matches: Total number of matches recorded
    """

    run_name: str
    best_elo: float = 1000.0  # DEPRECATED
    best_mu: float = MU_INIT
    best_display_rating: float = MU_INIT - 3 * SIGMA_INIT
    best_agent: str = "base_model"
    latest_step: int = 0
    total_matches: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "run_name": self.run_name,
            "best_elo": self.best_elo,
            "best_mu": self.best_mu,
            "best_display_rating": self.best_display_rating,
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
            best_mu=data.get("best_mu", MU_INIT),
            best_display_rating=data.get("best_display_rating", MU_INIT - 3 * SIGMA_INIT),
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
