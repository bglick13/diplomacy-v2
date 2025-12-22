"""
PFSP (Prioritized Fictitious Self-Play) Matchmaker for League Training.

This module implements opponent sampling for training, balancing:
- Stability (self-play with current policy)
- Learning (playing against similar-strength opponents)
- Regression testing (playing against baselines)
"""

import random
from dataclasses import dataclass, field
from typing import Any

from src.league.registry import LeagueRegistry
from src.league.types import AgentInfo, AgentType


@dataclass
class PFSPConfig:
    """Configuration for PFSP opponent sampling."""

    # Sampling distribution weights (biased toward positive EV matchups)
    self_play_weight: float = 0.30  # Current policy (stability)
    peer_weight: float = 0.30  # Similar Elo agents (learning)
    exploitable_weight: float = 0.35  # Weaker agents we should beat (positive EV)
    baseline_weight: float = 0.05  # Baseline bots (regression testing)

    # Elo thresholds for peer matching
    peer_elo_range: int = 100  # +/- from hero Elo
    near_peer_elo_range: int = 300  # Extended range for near-peers

    # Baseline sampling (stronger baselines for better signal)
    baseline_agents: list[str] = field(
        default_factory=lambda: [
            "chaos_bot",  # Elo ~900 - aggressive random
            "defensive_bot",  # Elo ~950 - supports/holds
            "territorial_bot",  # Elo ~950 - greedy expansion
            "coordinated_bot",  # Elo ~1000 - team coordination
        ]
    )

    def validate(self) -> None:
        """Ensure weights sum to 1.0."""
        total = (
            self.self_play_weight
            + self.peer_weight
            + self.exploitable_weight
            + self.baseline_weight
        )
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"PFSP weights must sum to 1.0, got {total}")


@dataclass
class MatchmakingResult:
    """Result of opponent sampling for a single game."""

    hero_power: str
    hero_agent: str
    hero_elo: float  # Hero's Elo used for matchmaking
    power_adapters: dict[str, str | None]  # Power -> adapter path or bot name
    opponent_categories: dict[str, str]  # Power -> category (self, peer, baseline, etc.)

    def to_wandb_dict(self) -> dict[str, Any]:
        """Format for WandB logging."""
        category_counts = {}
        for category in self.opponent_categories.values():
            category_counts[category] = category_counts.get(category, 0) + 1

        return {
            "hero_power": self.hero_power,
            "hero_agent": self.hero_agent,
            "hero_elo": self.hero_elo,
            "opponent_categories": category_counts,
            "num_unique_adapters": len(set(self.power_adapters.values())),
        }


class PFSPMatchmaker:
    """
    Prioritized Fictitious Self-Play matchmaker.

    Implements opponent sampling based on information gain:
    - 30% self-play (current policy) for stability
    - 40% peers (similar Elo) for optimal learning
    - 20% exploitable (weaker agents we should beat)
    - 10% baselines (regression testing)
    """

    POWERS = ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]

    def __init__(self, registry: LeagueRegistry, config: PFSPConfig | None = None):
        """
        Initialize matchmaker.

        Args:
            registry: League registry with agent info
            config: PFSP configuration (uses defaults if None)
        """
        self.registry = registry
        self.config = config or PFSPConfig()
        self.config.validate()

    def sample_opponents(
        self,
        hero_agent: str,
        hero_power: str | None = None,
        num_opponents: int = 6,
        hero_elo_override: float | None = None,
        hero_adapter_path: str | None = None,
    ) -> MatchmakingResult:
        """
        Sample opponents for a training game.

        Args:
            hero_agent: Agent name for the hero (e.g., "adapter_v50")
            hero_power: Which power the hero plays (random if None)
            num_opponents: Number of opponent slots (usually 6)
            hero_elo_override: Explicit Elo for the hero (used when hero isn't
                               registered yet, e.g., during active training)
            hero_adapter_path: Explicit adapter path for the hero (used for self-play
                               when hero isn't in registry yet)

        Returns:
            MatchmakingResult with power_adapters mapping
        """
        # Pick hero power randomly if not specified
        if hero_power is None:
            hero_power = random.choice(self.POWERS)

        # Store hero info for self-play (may not be in registry yet)
        # Used by _agent_to_adapter for unregistered checkpoints
        self._current_hero_agent = hero_agent
        self._current_hero_adapter_path = hero_adapter_path

        # Determine hero Elo: explicit override > registry lookup > default
        if hero_elo_override is not None:
            hero_elo = hero_elo_override
        else:
            hero_info = self.registry.get_agent(hero_agent)
            hero_elo = hero_info.elo if hero_info else 1000.0

        # Get available agents for opponent sampling
        all_agents = self.registry.get_all_agents()
        checkpoints = [a for a in all_agents if a.agent_type == AgentType.CHECKPOINT]
        baselines = [a for a in all_agents if a.agent_type == AgentType.BASELINE]

        # Categorize opponents
        opponents: list[tuple[str, str]] = []  # (agent_name, category)

        # Sample opponents according to PFSP distribution
        for _ in range(num_opponents):
            category = self._sample_category()
            agent = self._sample_agent_for_category(
                category, hero_agent, hero_elo, checkpoints, baselines
            )
            opponents.append((agent, category))

        # Build power_adapters mapping
        opponent_powers = [p for p in self.POWERS if p != hero_power]
        random.shuffle(opponent_powers)

        power_adapters: dict[str, str | None] = {}
        opponent_categories: dict[str, str] = {}

        # Assign hero
        power_adapters[hero_power] = self._agent_to_adapter(hero_agent)
        opponent_categories[hero_power] = "hero"

        # Assign opponents
        for i, power in enumerate(opponent_powers):
            if i < len(opponents):
                agent_name, category = opponents[i]
                power_adapters[power] = self._agent_to_adapter(agent_name)
                opponent_categories[power] = category
            else:
                # Fallback: use self-play
                power_adapters[power] = self._agent_to_adapter(hero_agent)
                opponent_categories[power] = "self"

        return MatchmakingResult(
            hero_power=hero_power,
            hero_agent=hero_agent,
            hero_elo=hero_elo,
            power_adapters=power_adapters,
            opponent_categories=opponent_categories,
        )

    def _sample_category(self) -> str:
        """Sample an opponent category based on PFSP weights."""
        r = random.random()
        cumulative = 0.0

        cumulative += self.config.self_play_weight
        if r < cumulative:
            return "self"

        cumulative += self.config.peer_weight
        if r < cumulative:
            return "peer"

        cumulative += self.config.exploitable_weight
        if r < cumulative:
            return "exploitable"

        return "baseline"

    def _sample_agent_for_category(
        self,
        category: str,
        hero_agent: str,
        hero_elo: float,
        checkpoints: list[AgentInfo],
        baselines: list[AgentInfo],
    ) -> str:
        """Sample a specific agent based on category."""
        if category == "self":
            return hero_agent

        if category == "baseline":
            if baselines:
                # Prefer actual baseline bots over base_model
                baseline_bots = [b for b in baselines if b.name in self.config.baseline_agents]
                if baseline_bots:
                    return random.choice(baseline_bots).name
                return random.choice(baselines).name
            return "random_bot"  # Fallback

        if category == "peer":
            # Find agents within peer_elo_range
            peers = [
                a
                for a in checkpoints
                if abs(a.elo - hero_elo) <= self.config.peer_elo_range and a.name != hero_agent
            ]
            if peers:
                return random.choice(peers).name

            # Fall back to near-peers
            near_peers = [
                a
                for a in checkpoints
                if abs(a.elo - hero_elo) <= self.config.near_peer_elo_range and a.name != hero_agent
            ]
            if near_peers:
                return random.choice(near_peers).name

            # Fall back to self-play if no peers - warn as this may be unintentional
            # This typically happens at training start when few checkpoints exist
            if checkpoints:
                # Other checkpoints exist but none are close enough in Elo
                print(
                    f"⚠️ Peer sampling: No agents within Elo range for {hero_agent} "
                    f"(Elo {hero_elo:.0f}). Falling back to self-play. "
                    f"Consider widening peer_elo_range or near_peer_elo_range."
                )
            else:
                # No other checkpoints at all - expected at training start
                print(
                    f"ℹ️ Peer sampling: No other checkpoints yet for {hero_agent}. "
                    "Using self-play (expected during early training)."
                )
            return hero_agent

        if category == "exploitable":
            # Find agents with lower Elo that we should beat
            weaker = [
                a
                for a in checkpoints
                if a.elo < hero_elo - 50  # At least 50 Elo weaker
                and a.name != hero_agent
            ]
            if weaker:
                # Weight by how exploitable they should be
                weights = [max(1, hero_elo - a.elo) for a in weaker]
                return random.choices(weaker, weights=weights, k=1)[0].name

            # Fall back to baselines if no weaker checkpoints
            if baselines:
                return random.choice(baselines).name
            return "random_bot"

        # Unknown category - default to self-play
        return hero_agent

    def _agent_to_adapter(self, agent_name: str) -> str | None:
        """Convert agent name to adapter path or bot identifier."""
        # Baseline bots use their name directly
        baseline_bots = {
            "random_bot",
            "chaos_bot",
            "defensive_bot",
            "territorial_bot",
            "coordinated_bot",
        }
        if agent_name in baseline_bots:
            return agent_name

        # Base model uses None
        if agent_name == "base_model":
            return None

        # For self-play: if agent matches the current hero, use provided adapter path
        # This handles cases where the hero checkpoint isn't in registry yet
        # (e.g., adapter_v55 when only every 10th step is registered)
        if (
            hasattr(self, "_current_hero_agent")
            and agent_name == self._current_hero_agent
            and hasattr(self, "_current_hero_adapter_path")
            and self._current_hero_adapter_path
        ):
            return self._current_hero_adapter_path

        # Checkpoints use their path from registry
        agent = self.registry.get_agent(agent_name)
        if agent and agent.path:
            return agent.path

        # Unknown agent - treat as base model
        return None

    def get_cold_start_opponents(self, hero_power: str) -> MatchmakingResult:
        """
        Get opponents for cold start (before any checkpoints exist).

        Uses only baselines for the first training steps.
        """
        opponent_powers = [p for p in self.POWERS if p != hero_power]

        power_adapters: dict[str, str | None] = {}
        opponent_categories: dict[str, str] = {}

        # Hero uses base model initially
        power_adapters[hero_power] = None
        opponent_categories[hero_power] = "hero"

        # Opponents alternate between random_bot and chaos_bot
        baseline_cycle = [
            "random_bot",
            "chaos_bot",
            "random_bot",
            "chaos_bot",
            "random_bot",
            "chaos_bot",
        ]
        for i, power in enumerate(opponent_powers):
            power_adapters[power] = baseline_cycle[i % len(baseline_cycle)]
            opponent_categories[power] = "baseline"

        return MatchmakingResult(
            hero_power=hero_power,
            hero_agent="base_model",
            hero_elo=1000.0,  # Base model defaults to 1000 Elo
            power_adapters=power_adapters,
            opponent_categories=opponent_categories,
        )

    def get_sampling_stats(self, results: list[MatchmakingResult]) -> dict[str, Any]:
        """
        Compute statistics over multiple matchmaking results.

        Useful for WandB logging to verify PFSP distribution.
        """
        category_counts: dict[str, int] = {}
        hero_powers: dict[str, int] = {}

        for result in results:
            for category in result.opponent_categories.values():
                if category != "hero":
                    category_counts[category] = category_counts.get(category, 0) + 1
            hero_powers[result.hero_power] = hero_powers.get(result.hero_power, 0) + 1

        total_opponents = sum(category_counts.values())
        category_rates = {
            k: v / total_opponents if total_opponents > 0 else 0 for k, v in category_counts.items()
        }

        return {
            "category_counts": category_counts,
            "category_rates": category_rates,
            "hero_power_distribution": hero_powers,
            "total_games": len(results),
        }
