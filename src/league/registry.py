"""
League Registry - Persistent storage for league agents and match history.

The registry is stored as a JSON file on the Modal Volume, enabling:
- Checkpoint versioning and promotion
- Elo rating tracking
- Match history for WandB visualization

File locking is used to prevent race conditions when multiple Modal containers
access the registry concurrently.
"""

import fcntl
import json
import logging
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from src.league.types import (
    DEFAULT_BASELINES,
    AgentInfo,
    AgentType,
    LeagueMetadata,
    MatchResult,
)

logger = logging.getLogger(__name__)

# Lock timeout in seconds
LOCK_TIMEOUT_S = 30.0


@contextmanager
def file_lock(lock_path: Path, timeout: float = LOCK_TIMEOUT_S) -> Generator[None, None, None]:
    """
    Context manager for exclusive file locking.

    Uses fcntl.flock() for blocking exclusive lock. The lock is released
    when the context exits.

    Args:
        lock_path: Path to the lock file (will be created if doesn't exist)
        timeout: Not currently used (flock blocks indefinitely), but kept for future
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = open(lock_path, "w")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()


class LeagueRegistry:
    """
    Manages the league of agents for training.

    The registry stores:
    - Agent information (baselines + checkpoints)
    - Match history for Elo computation
    - Metadata (best agent, latest step, etc.)

    Data is persisted as JSON to a Modal Volume.
    """

    def __init__(self, registry_path: Path | str, run_name: str | None = None):
        """
        Initialize the registry.

        Args:
            registry_path: Path to the league.json file
            run_name: Name of the training run (used for new registries)
        """
        self.path = Path(registry_path)
        self._lock_path = self.path.parent / ".league.lock"
        self._agents: dict[str, AgentInfo] = {}
        self._history: list[MatchResult] = []
        self._metadata: LeagueMetadata | None = None
        self._run_name = run_name

        # Load existing or create new (with lock to prevent races during init)
        with file_lock(self._lock_path):
            # Check if file exists AND has content (empty file = initialize)
            if self.path.exists() and self.path.stat().st_size > 0:
                self._load_unlocked()
            else:
                self._initialize_unlocked(run_name or "unknown-run")

    def _initialize_unlocked(self, run_name: str) -> None:
        """Initialize a new registry with default baselines (caller must hold lock)."""
        logger.info(f"Initializing new league registry for run: {run_name}")

        self._metadata = LeagueMetadata(run_name=run_name)

        # Add default baselines
        for baseline in DEFAULT_BASELINES:
            self._agents[baseline.name] = baseline

        self._history = []
        self._save_unlocked()

    def _load_unlocked(self) -> None:
        """Load registry from JSON file (caller must hold lock)."""
        logger.info(f"Loading league registry from {self.path}")

        with open(self.path) as f:
            data = json.load(f)

        # Load agents
        self._agents = {}
        for name, agent_data in data.get("agents", {}).items():
            self._agents[name] = AgentInfo.from_dict(name, agent_data)

        # Load history
        self._history = [MatchResult.from_dict(m) for m in data.get("history", [])]

        # Load metadata
        if "metadata" in data:
            self._metadata = LeagueMetadata.from_dict(data["metadata"])
        else:
            self._metadata = LeagueMetadata(run_name="unknown-run")

    def reload(self) -> None:
        """
        Reload registry from disk to get latest updates.

        This is useful when other processes (e.g., evaluate_league) have updated
        the registry file and we want to refresh our in-memory state.
        """
        with file_lock(self._lock_path):
            if self.path.exists() and self.path.stat().st_size > 0:
                self._load_unlocked()
                logger.debug(f"Reloaded league registry from {self.path}")
            else:
                logger.warning(
                    f"Registry file {self.path} does not exist or is empty, skipping reload"
                )

    def _save_unlocked(self) -> None:
        """Save registry to JSON file (caller must hold lock)."""
        # Ensure parent directory exists
        self.path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "agents": {name: agent.to_dict() for name, agent in self._agents.items()},
            "history": [m.to_dict() for m in self._history[-1000:]],  # Keep last 1000 matches
            "metadata": self._metadata.to_dict() if self._metadata else {},
        }

        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)

        logger.debug(f"Saved league registry to {self.path}")

    # -------------------------------------------------------------------------
    # Agent Management
    # -------------------------------------------------------------------------

    def get_agent(self, name: str) -> AgentInfo | None:
        """Get an agent by name."""
        return self._agents.get(name)

    def get_all_agents(self) -> list[AgentInfo]:
        """Get all agents in the league."""
        return list(self._agents.values())

    def get_checkpoints(self) -> list[AgentInfo]:
        """Get all checkpoint agents (excluding baselines)."""
        return [a for a in self._agents.values() if a.agent_type == AgentType.CHECKPOINT]

    def get_baselines(self) -> list[AgentInfo]:
        """Get all baseline agents."""
        return [a for a in self._agents.values() if a.agent_type == AgentType.BASELINE]

    def add_checkpoint(
        self,
        name: str,
        path: str,
        step: int,
        parent: str | None = None,
        initial_elo: float | None = None,
    ) -> AgentInfo:
        """
        Add a new checkpoint to the league.

        Args:
            name: Unique name for the checkpoint (e.g., "adapter_v50")
            path: Path to the LoRA adapter on the Volume
            step: Training step when checkpoint was created
            parent: Name of the parent checkpoint (for lineage)
            initial_elo: Starting Elo (defaults to parent's Elo or 1000)

        Returns:
            The created AgentInfo
        """
        with file_lock(self._lock_path):
            # Reload to get latest state before modifying
            if self.path.exists() and self.path.stat().st_size > 0:
                self._load_unlocked()

            if name in self._agents:
                logger.warning(f"Checkpoint {name} already exists, skipping")
                return self._agents[name]

            # Inherit Elo from parent if not specified
            if initial_elo is None:
                if parent and parent in self._agents:
                    initial_elo = self._agents[parent].elo
                elif parent:
                    # Parent specified but not found - warn about broken lineage
                    logger.warning(
                        f"Parent '{parent}' not found in registry for checkpoint '{name}'. "
                        f"Using default Elo 1000.0 instead of inheriting. "
                        "This may indicate a broken checkpoint lineage."
                    )
                    initial_elo = 1000.0
                else:
                    # No parent specified - normal case for first checkpoint
                    initial_elo = 1000.0

            agent = AgentInfo.create_checkpoint(
                name=name,
                path=path,
                step=step,
                parent=parent,
                elo=initial_elo,
            )
            self._agents[name] = agent

            # Update metadata
            if self._metadata:
                self._metadata.latest_step = max(self._metadata.latest_step, step)
                if agent.elo > self._metadata.best_elo:
                    self._metadata.best_elo = agent.elo
                    self._metadata.best_agent = name

            self._save_unlocked()
            logger.info(f"Added checkpoint {name} at step {step} with Elo {initial_elo:.0f}")

            return agent

    def update_elo(self, name: str, new_elo: float, matches_delta: int = 1) -> None:
        """
        Update an agent's Elo rating.

        Args:
            name: Agent name
            new_elo: New Elo rating
            matches_delta: Number of matches to add to count
        """
        with file_lock(self._lock_path):
            # Reload to get latest state before modifying
            if self.path.exists() and self.path.stat().st_size > 0:
                self._load_unlocked()

            if name not in self._agents:
                logger.warning(f"Agent {name} not found, cannot update Elo")
                return

            agent = self._agents[name]
            agent.elo = new_elo
            agent.matches += matches_delta

            # Update best agent tracking
            if self._metadata and new_elo > self._metadata.best_elo:
                self._metadata.best_elo = new_elo
                self._metadata.best_agent = name

            self._save_unlocked()

    def bulk_update_elos(self, elo_updates: dict[str, float], matches_delta: int = 1) -> None:
        """
        Update multiple agent Elo ratings at once.

        Args:
            elo_updates: Mapping of agent name to new Elo
            matches_delta: Number of matches to add to each agent's count.
                           Use this when the Elo update reflects multiple games.
        """
        with file_lock(self._lock_path):
            # Reload to get latest state before modifying
            if self.path.exists() and self.path.stat().st_size > 0:
                self._load_unlocked()

            for name, new_elo in elo_updates.items():
                if name in self._agents:
                    self._agents[name].elo = new_elo
                    self._agents[name].matches += matches_delta

            # Update best agent tracking
            if self._metadata:
                for name, new_elo in elo_updates.items():
                    if new_elo > self._metadata.best_elo:
                        self._metadata.best_elo = new_elo
                        self._metadata.best_agent = name

            self._save_unlocked()

    # -------------------------------------------------------------------------
    # Match History
    # -------------------------------------------------------------------------

    def add_match(self, match: MatchResult) -> None:
        """Add a match result to history."""
        self._history.append(match)

        if self._metadata:
            self._metadata.total_matches += 1

        # Don't save on every match - caller should call save() periodically
        # self._save()

    def get_recent_matches(self, limit: int = 100) -> list[MatchResult]:
        """Get the most recent matches."""
        return self._history[-limit:]

    def get_matches_for_agent(self, agent_name: str, limit: int = 50) -> list[MatchResult]:
        """Get matches where a specific agent participated."""
        matches = [m for m in self._history if agent_name in m.power_agents.values()]
        return matches[-limit:]

    def save_history(self) -> None:
        """Explicitly save (call after batch of add_match calls)."""
        with file_lock(self._lock_path):
            self._save_unlocked()

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def metadata(self) -> LeagueMetadata | None:
        """Get league metadata."""
        return self._metadata

    @property
    def best_elo(self) -> float:
        """Get the highest Elo in the league."""
        return self._metadata.best_elo if self._metadata else 1000.0

    @property
    def best_agent(self) -> str:
        """Get the name of the agent with highest Elo."""
        return self._metadata.best_agent if self._metadata else "base_model"

    @property
    def latest_step(self) -> int:
        """Get the most recent training step with a checkpoint."""
        return self._metadata.latest_step if self._metadata else 0

    @property
    def num_checkpoints(self) -> int:
        """Get the number of checkpoint agents."""
        return len(self.get_checkpoints())

    def to_dict(self) -> dict[str, Any]:
        """Export full registry as dictionary."""
        return {
            "agents": {name: agent.to_dict() for name, agent in self._agents.items()},
            "history": [m.to_dict() for m in self._history],
            "metadata": self._metadata.to_dict() if self._metadata else {},
        }


def should_add_to_league(
    step: int,
    registry: LeagueRegistry,
    current_elo: float | None = None,
) -> bool:
    """
    Determine if a checkpoint should be added to the league.

    Uses a geometric checkpoint schedule:
    1. Recent curriculum: every 10 steps for last 100 steps
    2. Historical anchors: every 100 steps forever
    3. Elite: new high score in Elo

    Args:
        step: Current training step
        registry: The league registry
        current_elo: Current estimated Elo (optional, for elite check)

    Returns:
        True if checkpoint should be added
    """
    # Always add first checkpoint
    if registry.num_checkpoints == 0:
        return True

    # Recent curriculum: every 10 steps for last 100
    if step % 10 == 0 and step > registry.latest_step - 100:
        return True

    # Historical anchors: every 100 steps
    if step % 100 == 0:
        return True

    # Elite: new high score
    if current_elo is not None and current_elo > registry.best_elo:
        return True

    return False


def get_checkpoint_name(run_name: str, step: int) -> str:
    """Generate a standard checkpoint name."""
    return f"{run_name}/adapter_v{step}"
