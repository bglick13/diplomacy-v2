import asyncio
import glob
import os
import random
import tempfile
import time
from collections import Counter, deque
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

if TYPE_CHECKING:
    from src.league.matchmaker import MatchmakingResult

import modal
import numpy as np
import torch
import wandb
from peft import LoraConfig, get_peft_model
from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.apps.common.images import gpu_image
from src.apps.common.volumes import MODELS_PATH, TRACE_PATH, VOLUME_PATH, trace_volume, volume
from src.training.loss import AdaptiveKLController, GRPOLoss, KLControllerConfig
from src.training.trainer import TrajectoryStats, process_trajectories
from src.utils.config import ExperimentConfig
from src.utils.observability import GPUStatsLogger, axiom, logger, stopwatch

app = modal.App("diplomacy-grpo-trainer")


# ============================================================================
# TYPE DEFINITIONS
# ============================================================================


class ExtractionStats(TypedDict):
    orders_expected: int
    orders_extracted: int
    empty_responses: int
    partial_responses: int
    extraction_rate: float


@dataclass
class RolloutResult:
    """Wrapped result from a single rollout."""

    adapter_used: str | None
    match_result: Any
    data: dict
    handle: Any


# ============================================================================
# CHECKPOINT MANAGEMENT
# ============================================================================


class CheckpointManager:
    """Manages training state checkpointing and resumption."""

    def __init__(self, run_name: str, optimizer: torch.optim.Optimizer, policy_model: Any):
        self.run_name = run_name
        self.optimizer = optimizer
        self.policy_model = policy_model
        self.run_path = MODELS_PATH / run_name

    def save(self, step: int, config: ExperimentConfig, wandb_run_id: str | None = None) -> None:
        """Save full training state with atomic writes."""
        self.run_path.mkdir(parents=True, exist_ok=True)
        state_path = self.run_path / f"training_state_v{step}.pt"

        state = {
            "step": step,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": config.model_dump(),
            "seed": config.seed,
        }

        if wandb_run_id:
            state["wandb_run_id"] = wandb_run_id

        # Atomic write: temp file then rename
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=str(self.run_path), delete=False, suffix=".tmp"
        ) as tmp_file:
            tmp_path = tmp_file.name
            torch.save(state, tmp_path)

        os.rename(tmp_path, str(state_path))
        volume.commit()
        logger.info(f"💾 Saved training state to {state_path}")

    def load(
        self, run_name: str | None = None, step: int | None = None, allow_fallback: bool = True
    ) -> tuple[int, str | None]:
        """
        Load training state and return (step, wandb_run_id).

        Args:
            run_name: Run to load from (defaults to self.run_name)
            step: Specific step to load (None = latest)
            allow_fallback: Try earlier checkpoints if latest corrupted
        """
        target_run = run_name or self.run_name
        target_path = MODELS_PATH / target_run

        # Find latest checkpoint if not specified
        if step is None:
            pattern = str(target_path / "training_state_v*.pt")
            state_files = glob.glob(pattern)
            if not state_files:
                raise FileNotFoundError(f"No training states found in {target_path}")

            steps = []
            for f in state_files:
                try:
                    s = int(f.split("_v")[-1].replace(".pt", ""))
                    steps.append(s)
                except ValueError:
                    pass

            if not steps:
                raise FileNotFoundError(f"No valid training states found in {target_path}")
            step = max(steps)

        assert step is not None

        # Build list of attempts (with fallbacks)
        attempts = [step]
        if allow_fallback:
            pattern = str(target_path / "training_state_v*.pt")
            all_files = glob.glob(pattern)
            all_steps = sorted(
                [
                    int(f.split("_v")[-1].replace(".pt", ""))
                    for f in all_files
                    if f.split("_v")[-1].replace(".pt", "").isdigit()
                ],
                reverse=True,
            )
            attempts.extend([s for s in all_steps if s < step][:2])

        # Try loading checkpoints
        last_error = None
        for attempt_step in attempts:
            state_path = target_path / f"training_state_v{attempt_step}.pt"
            if not state_path.exists():
                continue

            try:
                state = torch.load(str(state_path), weights_only=False)

                # Validate
                required_keys = ["step", "optimizer_state_dict", "config"]
                missing_keys = [k for k in required_keys if k not in state]
                if missing_keys:
                    raise ValueError(f"Checkpoint missing keys: {missing_keys}")

                if state["step"] != attempt_step:
                    logger.warning(
                        f"⚠️ Step mismatch: filename={attempt_step}, state={state['step']}"
                    )

                # Load optimizer state
                self.optimizer.load_state_dict(state["optimizer_state_dict"])

                # Load adapter
                adapter_path = target_path / f"adapter_v{attempt_step}"
                if adapter_path.exists():
                    self.policy_model.load_adapter(str(adapter_path), adapter_name="default")
                    logger.info(f"📂 Loaded adapter from {adapter_path}")
                else:
                    logger.warning(f"⚠️ Adapter not found at {adapter_path}")

                # Restore seed (including CUDA RNG for multi-GPU)
                saved_seed = state.get("seed")
                if saved_seed is not None:
                    random.seed(saved_seed)
                    np.random.seed(saved_seed)
                    torch.manual_seed(saved_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(saved_seed)
                    logger.info(f"🌱 Restored random seed: {saved_seed}")

                wandb_run_id = state.get("wandb_run_id")

                if attempt_step != step:
                    logger.warning(f"⚠️ Loaded step {attempt_step} (requested {step} unavailable)")

                logger.info(f"✅ Resumed from step {attempt_step} (run: {target_run})")
                return attempt_step, wandb_run_id

            except Exception as e:
                last_error = e
                logger.warning(f"⚠️ Failed to load checkpoint at step {attempt_step}: {e}")
                continue

        raise FileNotFoundError(
            f"Failed to load any valid checkpoint from {target_path}. Last error: {last_error}"
        )

    def save_adapter(self, step: int) -> Path:
        """Save adapter and return its path."""
        adapter_path = self.run_path / f"adapter_v{step}"
        self.policy_model.save_pretrained(str(adapter_path))
        volume.commit()
        logger.info(f"💾 Saved adapter to {adapter_path}")
        return adapter_path


# ============================================================================
# LEAGUE TRAINING HELPERS
# ============================================================================


@dataclass
class LeagueContext:
    """Context for league training."""

    registry: Any
    matchmaker: Any
    last_reload_step: int = -1


def initialize_league_training(cfg: ExperimentConfig) -> LeagueContext | None:
    """Initialize league training components if enabled."""
    if not cfg.league_training:
        return None

    from src.league import LeagueRegistry, PFSPConfig, PFSPMatchmaker

    logger.info("🏆 League training enabled - initializing registry and matchmaker")

    # Initialize registry
    if cfg.league_registry_path:
        registry_path = Path(cfg.league_registry_path)
    else:
        registry_path = Path(f"/data/league_{cfg.run_name}.json")

    # When disable_auto_resume is set, also reset the league registry
    # unless explicitly inheriting from another run
    if cfg.disable_auto_resume and not cfg.league_inherit_from and registry_path.exists():
        logger.warning(
            f"🗑️ Resetting league registry due to disable_auto_resume (was: {registry_path})"
        )
        registry_path.unlink()

    logger.info(f"📂 League registry path: {registry_path}")
    league_registry = LeagueRegistry(registry_path, run_name=cfg.run_name)

    # Inherit from previous run if specified
    if cfg.league_inherit_from:
        _inherit_league_opponents(cfg.league_inherit_from, league_registry)

    # Configure PFSP
    pfsp_config = PFSPConfig(
        self_play_weight=cfg.pfsp_self_play_weight,
        peer_weight=cfg.pfsp_peer_weight,
        exploitable_weight=cfg.pfsp_exploitable_weight,
        baseline_weight=cfg.pfsp_baseline_weight,
    )
    matchmaker = PFSPMatchmaker(league_registry, pfsp_config)

    logger.info(
        f"📊 League status: {league_registry.num_checkpoints} checkpoints, "
        f"best Elo: {league_registry.best_elo:.0f} ({league_registry.best_agent})"
    )

    # Log to WandB
    wandb.config.update(
        {
            "league_enabled": True,
            "pfsp_weights": {
                "self": cfg.pfsp_self_play_weight,
                "peer": cfg.pfsp_peer_weight,
                "exploitable": cfg.pfsp_exploitable_weight,
                "baseline": cfg.pfsp_baseline_weight,
            },
        }
    )

    return LeagueContext(registry=league_registry, matchmaker=matchmaker)


def _inherit_league_opponents(inherit_from: str, registry: Any) -> None:
    """Inherit opponents from a previous league run."""
    from src.league import LeagueRegistry

    inherit_path = Path(f"/data/league_{inherit_from}.json")
    if not inherit_path.exists():
        logger.warning(f"⚠️ Inherit league not found: {inherit_path}")
        return

    logger.info(f"📚 Inheriting opponents from {inherit_from}")
    parent_registry = LeagueRegistry(inherit_path, run_name=inherit_from)
    inherited_count = 0

    for agent in parent_registry.get_checkpoints():
        if agent.name not in [a.name for a in registry.get_all_agents()]:
            if agent.path and agent.step is not None:
                registry.add_checkpoint(
                    name=agent.name,
                    path=agent.path,
                    step=agent.step,
                    parent=agent.parent,
                    initial_elo=agent.elo,
                )
                inherited_count += 1

    logger.info(f"✅ Inherited {inherited_count} checkpoints from {inherit_from}")


def reload_league_registry(league_ctx: LeagueContext, step: int) -> None:
    """Reload registry periodically to get latest Elo updates."""
    if step - league_ctx.last_reload_step < 5:
        return

    try:
        old_best_elo = league_ctx.registry.best_elo
        volume.reload()
        league_ctx.registry.reload()
        league_ctx.last_reload_step = step
        new_best_elo = league_ctx.registry.best_elo

        if new_best_elo != old_best_elo:
            logger.info(
                f"🔄 Registry reloaded at step {step}: "
                f"best_elo {old_best_elo:.0f} → {new_best_elo:.0f} "
                f"(+{new_best_elo - old_best_elo:.0f})"
            )
        else:
            logger.debug(f"🔄 Registry reloaded at step {step}: best_elo unchanged")
    except Exception as e:
        logger.warning(f"⚠️ Failed to reload registry: {e}")


def update_elo_from_rollouts(
    collected_results: list[RolloutResult],
    league_ctx: LeagueContext,
    k_factor: float = 16.0,
) -> dict[str, float]:
    """
    Update Elo ratings from rollout game outcomes.

    Args:
        collected_results: List of RolloutResult from current step
        league_ctx: League context with registry
        k_factor: K-factor for Elo updates (lower = more stable)

    Returns:
        Dict of agent_name -> new_elo for agents that were updated
    """
    from collections import defaultdict

    from src.league.elo import update_elo_from_match

    all_elo_deltas: dict[str, list[float]] = defaultdict(list)

    for result in collected_results:
        # Get game outcomes from rollout data
        match_results_data = result.data.get("match_results", [])

        for match in match_results_data:
            power_agents = match.get("power_agents", {})
            power_scores = match.get("power_scores", {})

            if not power_agents or not power_scores:
                continue

            # Get current Elos for all agents in this match
            unique_agents = {a for a in power_agents.values() if a}
            agent_elos = {}
            for agent in unique_agents:
                # Handle bot agents (they're in registry as baselines)
                info = league_ctx.registry.get_agent(agent)
                agent_elos[agent] = info.elo if info else 1000.0

            # Compute new Elos using pairwise decomposition
            new_elos = update_elo_from_match(power_agents, power_scores, agent_elos, k=k_factor)

            # Accumulate deltas for averaging
            for agent, new_elo in new_elos.items():
                delta = new_elo - agent_elos[agent]
                all_elo_deltas[agent].append(delta)

    # Apply average delta for each agent
    final_updates = {}
    for agent, deltas in all_elo_deltas.items():
        current = league_ctx.registry.get_agent(agent)
        if current:
            avg_delta = sum(deltas) / len(deltas)
            final_updates[agent] = current.elo + avg_delta

    if final_updates:
        league_ctx.registry.bulk_update_elos(final_updates)
        logger.info(f"📊 Elo updated for {len(final_updates)} agents from rollouts (k={k_factor})")

    return final_updates


def update_trueskill_from_rollouts(
    collected_results: list[RolloutResult],
    league_ctx: LeagueContext,
) -> dict[str, tuple[float, float]]:
    """
    Update TrueSkill ratings from rollout game outcomes.

    Args:
        collected_results: List of RolloutResult from current step
        league_ctx: League context with registry

    Returns:
        Dict of agent_name -> (new_mu, new_sigma) for agents that were updated
    """
    from collections import defaultdict

    from src.league.trueskill_rating import update_trueskill_from_match

    # Track all rating updates per agent
    all_mu_deltas: dict[str, list[float]] = defaultdict(list)
    all_new_sigmas: dict[str, list[float]] = defaultdict(list)

    for result in collected_results:
        # Get game outcomes from rollout data
        match_results_data = result.data.get("match_results", [])

        for match in match_results_data:
            power_agents = match.get("power_agents", {})
            power_scores = match.get("power_scores", {})

            if not power_agents or not power_scores:
                continue

            # Get current TrueSkill ratings for all agents in this match
            unique_agents = {a for a in power_agents.values() if a}
            agent_ratings: dict[str, tuple[float, float]] = {}
            for agent in unique_agents:
                info = league_ctx.registry.get_agent(agent)
                if info:
                    agent_ratings[agent] = (info.mu, info.sigma)
                else:
                    # Default for unknown agents
                    from src.league.trueskill_rating import MU_INIT, SIGMA_INIT

                    agent_ratings[agent] = (MU_INIT, SIGMA_INIT)

            # Compute new ratings using TrueSkill
            new_ratings = update_trueskill_from_match(power_agents, power_scores, agent_ratings)

            # Accumulate updates for averaging
            for agent, (new_mu, new_sigma) in new_ratings.items():
                old_mu, _old_sigma = agent_ratings[agent]
                all_mu_deltas[agent].append(new_mu - old_mu)
                all_new_sigmas[agent].append(new_sigma)

    # Apply average delta for mu, minimum for sigma (most confident after seeing more games)
    final_updates: dict[str, tuple[float, float]] = {}
    for agent, mu_deltas in all_mu_deltas.items():
        current = league_ctx.registry.get_agent(agent)
        if current:
            avg_mu_delta = sum(mu_deltas) / len(mu_deltas)
            min_sigma = min(all_new_sigmas[agent])
            final_updates[agent] = (current.mu + avg_mu_delta, min_sigma)

    if final_updates:
        league_ctx.registry.bulk_update_trueskill(final_updates)
        # Log a sample of the updates
        sample_agent = next(iter(final_updates))
        sample_info = league_ctx.registry.get_agent(sample_agent)
        if sample_info:
            logger.info(
                f"📊 TrueSkill updated for {len(final_updates)} agents "
                f"(e.g., {sample_agent}: rating={sample_info.display_rating:.1f})"
            )

    return final_updates


def add_checkpoint_to_league(
    step: int,
    adapter_rel_path: str,
    cfg: ExperimentConfig,
    league_ctx: LeagueContext,
    evaluate_league_fn: Any,
) -> None:
    """Add checkpoint to league and spawn evaluation if needed."""
    from src.league import should_add_to_league

    checkpoint_key = adapter_rel_path

    if not should_add_to_league(step, league_ctx.registry):
        return

    # Find the most recent registered checkpoint as parent (not just step-1)
    # This ensures Elo inheritance works even when some steps are skipped
    if step > 0:
        # Get all registered checkpoints and find the one with highest step < current
        checkpoints = league_ctx.registry.get_checkpoints()
        earlier_checkpoints = [c for c in checkpoints if c.step < step]
        if earlier_checkpoints:
            # Use the most recent registered checkpoint
            parent_checkpoint = max(earlier_checkpoints, key=lambda c: c.step)
            parent_key = parent_checkpoint.name
        else:
            # No earlier checkpoints, use base_model
            parent_key = "base_model"
    else:
        parent_key = "base_model"

    league_ctx.registry.add_checkpoint(
        name=checkpoint_key,
        path=adapter_rel_path,
        step=step,
        parent=parent_key,
    )
    volume.commit()
    logger.info(f"🏆 Added checkpoint {checkpoint_key} to league")

    # Spawn league evaluation if enabled (for additional rating validation)
    if cfg.league_eval_every_n_steps > 0 and step % cfg.league_eval_every_n_steps == 0 and step > 1:
        logger.info(f"🎯 Spawning league evaluation for {checkpoint_key}")
        registry_path_str = str(
            f"/data/league_{cfg.run_name}.json"
            if not cfg.league_registry_path
            else cfg.league_registry_path
        )

        evaluate_league_fn.spawn(
            challenger_path=adapter_rel_path,
            league_registry_path=registry_path_str,
            games_per_opponent=cfg.league_eval_games_per_opponent,
            max_years=cfg.rollout_horizon_years,
            model_id=cfg.base_model_id,
            wandb_run_id=wandb.run.id if wandb.run else None,
            training_step=step,
            show_valid_moves=cfg.show_valid_moves,
            compact_prompts=cfg.compact_prompts,
            prefix_cache_optimized=cfg.prefix_cache_optimized,
            temperature=cfg.temperature,
            max_new_tokens=cfg.max_new_tokens,
        )


# ============================================================================
# ROLLOUT MANAGEMENT
# ============================================================================


class RolloutManager:
    """Manages rollout spawning, collection, and buffering.

    Enforces max_concurrent_containers to avoid hitting Modal's hard limits.
    """

    # Maximum number of times to retry a failed rollout before giving up
    MAX_ROLLOUT_RETRIES = 2

    def __init__(
        self,
        cfg: ExperimentConfig,
        run_rollout_fn: Any,
        league_ctx: LeagueContext | None = None,
    ):
        self.cfg = cfg
        self.run_rollout_fn = run_rollout_fn
        self.league_ctx = league_ctx
        self.buffer: deque[tuple[Any, str | None, Any]] = deque()
        self.max_containers = cfg.max_concurrent_containers
        # Track retry counts by handle id to limit retries
        self._retry_counts: dict[int, int] = {}

    @property
    def containers_in_flight(self) -> int:
        """Count of currently in-flight containers (buffer size)."""
        return len(self.buffer)

    @property
    def available_capacity(self) -> int:
        """How many more containers can be spawned before hitting the limit."""
        return max(0, self.max_containers - self.containers_in_flight)

    def spawn_batch(
        self, hero_adapter_path: str | None, current_step: int, max_to_spawn: int | None = None
    ) -> tuple[list[Any], list[Any]]:
        """Spawn a batch of rollouts with appropriate opponent sampling.

        Args:
            hero_adapter_path: Adapter path for hero power
            current_step: Current training step
            max_to_spawn: Maximum rollouts to spawn (respects container limit if None)
        """
        if self.league_ctx:
            reload_league_registry(self.league_ctx, current_step)

        handles = []
        match_results = []

        # Respect container limit
        batch_size = self.cfg.num_groups_per_step
        if max_to_spawn is not None:
            batch_size = min(batch_size, max_to_spawn)

        for _ in range(batch_size):
            if self.league_ctx:
                handle, match_result = self._spawn_league_rollout(hero_adapter_path)
            else:
                handle = self._spawn_legacy_rollout(hero_adapter_path)
                match_result = None

            handles.append(handle)
            match_results.append(match_result)

        return handles, match_results

    def _spawn_league_rollout(self, hero_adapter_path: str | None) -> tuple[Any, Any]:
        """Spawn a single league training rollout."""
        assert self.league_ctx is not None

        hero_power = random.choice(self.league_ctx.matchmaker.POWERS)
        hero_agent_name = hero_adapter_path or "base_model"

        # Check if this should be a dumbbot benchmark game
        is_dumbbot_game = (
            self.cfg.dumbbot_game_probability > 0
            and random.random() < self.cfg.dumbbot_game_probability
        )

        if is_dumbbot_game:
            # All-dumbbot game: hero vs 6 DumbBots
            match_result = self._create_dumbbot_match(hero_power, hero_agent_name)
        elif self.league_ctx.registry.num_checkpoints > 0:
            # Normal PFSP opponent sampling
            hero_info = self.league_ctx.registry.get_agent(hero_agent_name)
            estimated_hero_rating = (
                hero_info.display_rating
                if hero_info
                else self.league_ctx.registry.best_display_rating
            )

            match_result = self.league_ctx.matchmaker.sample_opponents(
                hero_agent=hero_agent_name,
                hero_power=hero_power,
                num_opponents=6,
                hero_rating_override=estimated_hero_rating,
                hero_adapter_path=hero_adapter_path,
            )
        else:
            match_result = self.league_ctx.matchmaker.get_cold_start_opponents(hero_power)

        # Override hero adapter (path) and agent name
        power_adapters = match_result.power_adapters.copy()
        power_adapters[hero_power] = hero_adapter_path

        power_agent_names = match_result.power_agent_names.copy()
        power_agent_names[hero_power] = hero_agent_name

        handle = self.run_rollout_fn.spawn(
            self.cfg.model_dump(),
            power_adapters=power_adapters,
            power_agent_names=power_agent_names,
            hero_power=hero_power,
        )

        return handle, match_result

    def _create_dumbbot_match(self, hero_power: str, hero_agent_name: str) -> "MatchmakingResult":
        """Create a match where all opponents are DumbBots."""
        from src.league.matchmaker import MatchmakingResult

        powers = ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]
        opponent_powers = [p for p in powers if p != hero_power]

        power_adapters: dict[str, str | None] = {hero_power: None}  # Will be overridden
        power_agent_names: dict[str, str] = {hero_power: hero_agent_name}
        opponent_categories: dict[str, str] = {}

        for power in opponent_powers:
            power_adapters[power] = "dumb_bot"
            power_agent_names[power] = "dumb_bot"
            opponent_categories[power] = "dumbbot_benchmark"

        return MatchmakingResult(
            hero_power=hero_power,
            hero_agent=hero_agent_name,
            hero_rating=0.0,  # Not used for dumbbot games
            power_adapters=power_adapters,
            power_agent_names=power_agent_names,
            opponent_categories=opponent_categories,
        )

    def _spawn_legacy_rollout(self, hero_adapter_path: str | None) -> Any:
        """Spawn a single legacy (non-league) rollout."""
        return self.run_rollout_fn.spawn(self.cfg.model_dump(), lora_name=hero_adapter_path)

    def prefill_buffer(self, adapter_path: str | None, buffer_depth: int) -> None:
        """Prefill rollout buffer with initial batches, respecting container limit."""
        target_rollouts = buffer_depth * self.cfg.num_groups_per_step
        actual_target = min(target_rollouts, self.max_containers)

        if actual_target < target_rollouts:
            logger.warning(
                f"⚠️ Container cap ({self.max_containers}) limits buffer prefill: "
                f"requested {target_rollouts}, spawning {actual_target}"
            )

        spawned = 0
        for _ in range(buffer_depth):
            if spawned >= actual_target:
                break
            remaining = actual_target - spawned
            handles, match_results = self.spawn_batch(
                adapter_path, current_step=0, max_to_spawn=remaining
            )
            for h, mr in zip(handles, match_results, strict=False):
                self.buffer.append((h, adapter_path, mr))
                spawned += 1

        logger.info(
            f"📦 Buffer initialized: {len(self.buffer)} rollouts "
            f"(cap: {self.max_containers}, depth: {buffer_depth})"
        )

    def collect_rollouts(
        self, target_count: int, current_adapter: str | None, step: int
    ) -> tuple[list[RolloutResult], dict[str, float | int]]:
        """
        Collect ready rollouts from buffer.

        Returns:
            Tuple of (results, timing_stats)
        """
        collected: list[RolloutResult] = []
        max_volume_reload_s = 0.0
        max_rollout_total_s = 0.0
        failed_count = 0

        rollout_start = time.time()
        max_wait_s = 300.0
        deadline = rollout_start + max_wait_s
        polls_without_progress = 0
        MAX_POLLS = 50

        while len(collected) < target_count:
            # Spawn emergency rollout if buffer empty (still respecting container limit)
            if not self.buffer:
                capacity = self.available_capacity
                if capacity > 0:
                    logger.warning(
                        f"Rollout buffer unexpectedly empty; spawning emergency rollout "
                        f"(capacity: {capacity})"
                    )
                    handles, match_results = self.spawn_batch(
                        current_adapter, step, max_to_spawn=capacity
                    )
                    for h, mr in zip(handles, match_results, strict=False):
                        self.buffer.append((h, current_adapter, mr))
                    polls_without_progress = 0
                else:
                    logger.error(
                        f"Buffer empty and at container cap ({self.max_containers}), cannot spawn"
                    )
                    break

            handle, adapter_used, match_result = self.buffer.popleft()

            # Adaptive timeout
            timeout = (
                0.1 if polls_without_progress < MAX_POLLS else min(30.0, deadline - time.time())
            )

            try:
                result = handle.get(timeout=timeout)
                polls_without_progress = 0

                # Check if rollout returned a failure result
                from src.utils.results import is_rollout_failure, parse_rollout_result

                if is_rollout_failure(result):
                    failure = parse_rollout_result(result)
                    failed_count += 1
                    logger.error(
                        f"❌ Rollout returned failure: {failure}\n"
                        f"  This is NOT a transient error - fix the root cause."
                    )
                    # Don't retry failures - they indicate config/setup issues
                    self._retry_counts.pop(id(handle), None)
                    continue

                # Track timing
                timing = result.get("timing", {})
                max_volume_reload_s = max(max_volume_reload_s, timing.get("volume_reload_s", 0.0))
                max_rollout_total_s = max(max_rollout_total_s, timing.get("total_s", 0.0))

                collected.append(
                    RolloutResult(
                        adapter_used=adapter_used,
                        match_result=match_result,
                        data=result,
                        handle=handle,
                    )
                )
                # Clean up retry tracking on success
                self._retry_counts.pop(id(handle), None)

            except TimeoutError:
                self.buffer.append((handle, adapter_used, match_result))
                polls_without_progress += 1
                if time.time() > deadline:
                    logger.error(
                        f"Rollout deadline exceeded ({max_wait_s}s), "
                        f"collected {len(collected)}/{target_count}"
                    )
                    break
            except Exception as e:
                if "timeout" in str(e).lower():
                    self.buffer.append((handle, adapter_used, match_result))
                    polls_without_progress += 1
                else:
                    # Check retry count for this handle
                    handle_id = id(handle)
                    retry_count = self._retry_counts.get(handle_id, 0)

                    if retry_count < self.MAX_ROLLOUT_RETRIES:
                        # Re-queue for retry
                        self._retry_counts[handle_id] = retry_count + 1
                        self.buffer.append((handle, adapter_used, match_result))
                        logger.warning(
                            f"⚠️ Rollout failed (retry {retry_count + 1}/{self.MAX_ROLLOUT_RETRIES}): "
                            f"{type(e).__name__}: {e}"
                        )
                    else:
                        # Max retries exceeded, permanently fail
                        failed_count += 1
                        # Clean up retry tracking
                        self._retry_counts.pop(handle_id, None)
                        logger.error(
                            f"❌ Rollout failed after {self.MAX_ROLLOUT_RETRIES} retries: "
                            f"{type(e).__name__}: {e}"
                        )
                    polls_without_progress = 0

        return collected, {
            "max_volume_reload_s": max_volume_reload_s,
            "max_rollout_total_s": max_rollout_total_s,
            "failed_count": failed_count,
            "polls_without_progress": polls_without_progress,
        }

    def spawn_replacement_rollouts(
        self, consumed_count: int, adapter_path: str | None, step: int
    ) -> int:
        """Spawn rollouts to replace consumed ones, respecting container limit.

        Returns count of rollouts actually spawned.
        """
        # Calculate how many we want vs how many we can spawn
        capacity = self.available_capacity
        to_spawn = min(consumed_count, capacity)

        if to_spawn < consumed_count:
            logger.info(
                f"📊 Container cap limits replacement: "
                f"wanted {consumed_count}, can spawn {to_spawn} "
                f"(in-flight: {self.containers_in_flight}, cap: {self.max_containers})"
            )

        if to_spawn <= 0:
            return 0

        batches_needed = max(
            1, (to_spawn + self.cfg.num_groups_per_step - 1) // self.cfg.num_groups_per_step
        )

        spawned = 0
        for _ in range(batches_needed):
            if spawned >= to_spawn:
                break
            remaining = to_spawn - spawned
            handles, match_results = self.spawn_batch(adapter_path, step, max_to_spawn=remaining)
            for h, mr in zip(handles, match_results, strict=False):
                self.buffer.append((h, adapter_path, mr))
                spawned += 1

        return spawned


# ============================================================================
# METRICS & LOGGING
# ============================================================================


def aggregate_extraction_stats(results: list[RolloutResult]) -> ExtractionStats:
    """Aggregate extraction statistics from multiple rollout results."""
    stats: ExtractionStats = {
        "orders_expected": 0,
        "orders_extracted": 0,
        "empty_responses": 0,
        "partial_responses": 0,
        "extraction_rate": 1.0,
    }

    for result in results:
        result_stats = result.data["extraction_stats"]
        stats["orders_expected"] += result_stats["orders_expected"]
        stats["orders_extracted"] += result_stats["orders_extracted"]
        stats["empty_responses"] += result_stats["empty_responses"]
        stats["partial_responses"] += result_stats["partial_responses"]

    if stats["orders_expected"] > 0:
        stats["extraction_rate"] = float(stats["orders_extracted"]) / float(
            stats["orders_expected"]
        )

    return stats


class GameStats(TypedDict):
    """Aggregated game statistics from rollouts."""

    sc_counts: list[int]
    win_bonus_awarded: int
    total_games: int
    avg_sc_count: float
    win_bonus_rate: float
    hero_placements: list[int]
    placement_counts: dict[int, int]  # rank -> count


def aggregate_game_stats(results: list[RolloutResult]) -> GameStats:
    """Aggregate game statistics from multiple rollout results."""
    all_sc_counts: list[int] = []
    all_hero_placements: list[int] = []
    total_win_bonus = 0
    total_games = 0

    for result in results:
        game_stats = result.data.get("game_stats", {})
        all_sc_counts.extend(game_stats.get("sc_counts", []))
        all_hero_placements.extend(game_stats.get("hero_placements", []))
        total_win_bonus += game_stats.get("win_bonus_awarded", 0)
        total_games += game_stats.get("total_games", 0)

    avg_sc = sum(all_sc_counts) / len(all_sc_counts) if all_sc_counts else 0.0
    win_rate = total_win_bonus / total_games if total_games > 0 else 0.0

    # Count placements by rank (1-7)
    placement_counts: dict[int, int] = dict.fromkeys(range(1, 8), 0)
    for rank in all_hero_placements:
        if 1 <= rank <= 7:
            placement_counts[rank] += 1

    return GameStats(
        sc_counts=all_sc_counts,
        win_bonus_awarded=total_win_bonus,
        total_games=total_games,
        avg_sc_count=avg_sc,
        win_bonus_rate=win_rate,
        hero_placements=all_hero_placements,
        placement_counts=placement_counts,
    )


def _build_placement_metrics(game_stats: GameStats) -> dict[str, float]:
    """Build placement rate metrics for WandB logging.

    Returns dict with keys like:
    - placement/1st_rate: fraction finishing 1st
    - placement/2nd_rate: fraction finishing 2nd
    - ...
    - placement/top3_rate: fraction finishing 1st, 2nd, or 3rd
    """
    placement_counts = game_stats.get("placement_counts", {})
    hero_placements = game_stats.get("hero_placements", [])
    total = len(hero_placements) if hero_placements else 0

    if total == 0:
        return {}

    ordinal_names = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 5: "5th", 6: "6th", 7: "7th"}

    metrics: dict[str, float] = {}
    for rank in range(1, 8):
        count = placement_counts.get(rank, 0)
        rate = count / total
        metrics[f"placement/{ordinal_names[rank]}_rate"] = rate

    # Add aggregate metrics
    top3_count = sum(placement_counts.get(r, 0) for r in [1, 2, 3])
    bottom3_count = sum(placement_counts.get(r, 0) for r in [5, 6, 7])
    metrics["placement/top3_rate"] = top3_count / total
    metrics["placement/bottom3_rate"] = bottom3_count / total

    # Mean placement (lower is better)
    if hero_placements:
        metrics["placement/mean_rank"] = sum(hero_placements) / len(hero_placements)

    return metrics


def build_wandb_metrics(
    step: int,
    step_metrics: dict[str, Any],
    extraction_stats: ExtractionStats,
    game_stats: GameStats,
    traj_stats: TrajectoryStats,
    cache_stats: dict | None,
    league_ctx: LeagueContext | None,
    match_results: list[Any],
    sim_years_per_step: float,
    collected_results: list["RolloutResult"] | None = None,
) -> dict[str, Any]:
    """Build comprehensive WandB metrics dict."""
    cumulative_sim_years = (step + 1) * sim_years_per_step

    metrics = {
        "benchmark/step": step,
        "benchmark/loss": step_metrics["loss"],
        "benchmark/kl": step_metrics["kl"],
        "benchmark/reward_mean": traj_stats.reward_mean,
        "benchmark/reward_std": traj_stats.reward_std,
        "benchmark/rollout_time_s": step_metrics["rollout_time_s"],
        "benchmark/training_time_s": step_metrics["training_time_s"],
        "benchmark/trajectories": step_metrics["processed_trajectories"],
        "benchmark/grad_norm": step_metrics["grad_norm"],
        "benchmark/pipeline_overlap_s": step_metrics["pipeline_overlap_s"],
        # True if ref logprobs came from rollouts, False if computed in trainer
        "benchmark/used_cached_ref_logprobs": step_metrics.get("used_cached_ref_logprobs", False),
        # Rollout timing
        "rollout/max_volume_reload_s": step_metrics["max_volume_reload_s"],
        "rollout/max_total_s": step_metrics["max_rollout_total_s"],
        "rollout/failed_count": step_metrics["failed_rollouts"],
        # Buffer health
        "buffer/depth": step_metrics["buffer_depth_actual"],
        "buffer/consumed": step_metrics["rollouts_consumed"],
        "buffer/spawned": step_metrics.get("rollouts_spawned", 0),
        "buffer/collection_polls": step_metrics["collection_polls"],
        # Container utilization
        "containers/in_flight": step_metrics.get("containers_in_flight", 0),
        "containers/cap": step_metrics.get("container_cap", 0),
        "containers/utilization": (
            step_metrics.get("containers_in_flight", 0) / step_metrics.get("container_cap", 1)
            if step_metrics.get("container_cap", 0) > 0
            else 0.0
        ),
        # Extraction
        "extraction/rate": extraction_stats["extraction_rate"],
        "extraction/orders_expected": extraction_stats["orders_expected"],
        "extraction/orders_extracted": extraction_stats["orders_extracted"],
        "extraction/empty_responses": extraction_stats["empty_responses"],
        "extraction/partial_responses": extraction_stats["partial_responses"],
        # Game outcome stats
        "game/avg_sc_count": game_stats["avg_sc_count"],
        "game/win_bonus_rate": game_stats["win_bonus_rate"],
        "game/total_games": game_stats["total_games"],
        # Placement rates (position-based scoring)
        **_build_placement_metrics(game_stats),
        # Power law
        "power_law/cumulative_simulated_years": cumulative_sim_years,
        "power_law/simulated_years_per_step": sim_years_per_step,
        "power_law/reward_at_compute": traj_stats.reward_mean,
        # KL diagnostics
        "kl/mean": step_metrics.get("kl", 0.0),
        "kl/max": step_metrics.get("kl_max", 0.0),
        "kl/min": step_metrics.get("kl_min", 0.0),
        "kl/std": step_metrics.get("kl_std", 0.0),
        "kl/beta": step_metrics.get("effective_beta", 0.04),
        "kl/warmup_progress": step_metrics.get("kl_ctrl_warmup_progress", 1.0),
        "kl/ema": step_metrics.get("kl_ctrl_kl_ema", step_metrics.get("kl", 0.0)),
        "kl/beta_adjusted": step_metrics.get("kl_ctrl_beta_adjusted", 0.0),
        # Advantage diagnostics
        "advantage/mean": traj_stats.advantage_mean,
        "advantage/std": traj_stats.advantage_std,
        "advantage/min": traj_stats.advantage_min,
        "advantage/max": traj_stats.advantage_max,
        "advantage/clipped_count": traj_stats.advantages_clipped,
        # Skip/processing diagnostics
        "processing/skip_rate": traj_stats.skip_rate,
        "processing/skipped_single_sample": traj_stats.skipped_single_sample_groups,
        "processing/skipped_zero_variance": traj_stats.skipped_zero_variance_groups,
        "processing/effective_batch_size": traj_stats.effective_batch_size,
        "processing/total_samples_skipped": traj_stats.total_samples_skipped,
        # PPO clipping diagnostics (DAPO-style)
        "ppo/ratio_mean": step_metrics.get("ratio_mean", 1.0),
        "ppo/ratio_std": step_metrics.get("ratio_std", 0.0),
        "ppo/clip_fraction": step_metrics.get("ratio_clipped_fraction", 0.0),
        "ppo/rollout_logprobs_rate": step_metrics.get("rollout_logprobs_rate", 0.0),
        # Policy entropy diagnostics (GTPO-style collapse detection)
        "policy/entropy_mean": step_metrics.get("entropy_mean", 0.0),
        "policy/entropy_std": step_metrics.get("entropy_std", 0.0),
        # Importance sampling diagnostics (vLLM-HuggingFace mismatch correction)
        "ppo/is_ratio_mean": step_metrics.get("is_ratio_mean", 1.0),
        "ppo/is_ratio_std": step_metrics.get("is_ratio_std", 0.0),
        "ppo/is_masked_fraction": step_metrics.get("is_masked_fraction", 0.0),
    }

    # NOTE: Cache metrics (cache/hit_rate, etc.) were removed because they never
    # worked correctly - vllm_queries and total_queries were never populated,
    # resulting in 0% hit rate regardless of actual cache performance.

    # Add league metrics
    if league_ctx:
        _add_league_metrics(metrics, league_ctx, match_results, collected_results)

    return metrics


class DumbbotGameStats(TypedDict):
    """Aggregated statistics for dumbbot benchmark games."""

    games_count: int
    win_count: int  # 1st place
    win_rate: float
    top3_count: int
    top3_rate: float
    avg_sc_count: float
    avg_placement: float
    sc_counts: list[int]
    placements: list[int]


def aggregate_dumbbot_game_stats(
    collected_results: list["RolloutResult"],
) -> DumbbotGameStats | None:
    """
    Aggregate statistics for dumbbot benchmark games.

    Dumbbot games are identified by having all 6 opponents with
    opponent_categories = "dumbbot_benchmark".

    Returns:
        DumbbotGameStats if any dumbbot games found, None otherwise
    """
    sc_counts: list[int] = []
    placements: list[int] = []

    for result in collected_results:
        match_result = result.match_result
        if not match_result:
            continue

        # Check if this is a dumbbot benchmark game
        opponent_categories = getattr(match_result, "opponent_categories", None)
        if not opponent_categories:
            continue

        # All 6 opponents should be "dumbbot_benchmark"
        non_hero_categories = [c for c in opponent_categories.values() if c != "hero"]
        if not all(c == "dumbbot_benchmark" for c in non_hero_categories):
            continue

        # This is a dumbbot game - extract hero stats
        game_stats = result.data.get("game_stats", {})
        hero_sc_counts = game_stats.get("sc_counts", [])
        hero_placements = game_stats.get("hero_placements", [])

        sc_counts.extend(hero_sc_counts)
        placements.extend(hero_placements)

    if not placements:
        return None

    games_count = len(placements)
    win_count = sum(1 for p in placements if p == 1)
    top3_count = sum(1 for p in placements if p <= 3)

    return DumbbotGameStats(
        games_count=games_count,
        win_count=win_count,
        win_rate=win_count / games_count if games_count > 0 else 0.0,
        top3_count=top3_count,
        top3_rate=top3_count / games_count if games_count > 0 else 0.0,
        avg_sc_count=sum(sc_counts) / len(sc_counts) if sc_counts else 0.0,
        avg_placement=sum(placements) / len(placements) if placements else 0.0,
        sc_counts=sc_counts,
        placements=placements,
    )


def aggregate_rewards_by_opponent_type(
    collected_results: list["RolloutResult"],
) -> dict[str, dict[str, float]]:
    """
    Aggregate rewards by dominant opponent category.

    For each game, we determine the "dominant" opponent category
    (most common among the 6 opponents) and attribute all rewards
    from that game to that category.

    Returns:
        Dict mapping category -> {"reward_mean", "reward_count", "game_count"}
    """
    from collections import defaultdict

    category_rewards: dict[str, list[float]] = defaultdict(list)
    category_games: dict[str, int] = defaultdict(int)

    for result in collected_results:
        match_result = result.match_result
        trajectories = result.data.get("trajectories", [])

        if not match_result or not trajectories:
            continue

        # Determine dominant opponent category for this game
        opponent_categories = getattr(match_result, "opponent_categories", None)
        if not opponent_categories:
            continue

        # Count categories (excluding "hero")
        cat_counts = Counter(c for c in opponent_categories.values() if c != "hero")
        if not cat_counts:
            dominant_cat = "self"  # Pure self-play fallback
        else:
            dominant_cat = cat_counts.most_common(1)[0][0]

        # Aggregate rewards for this category
        for traj in trajectories:
            reward = traj.get("reward", 0.0)
            category_rewards[dominant_cat].append(reward)

        category_games[dominant_cat] += 1

    # Compute means
    aggregated: dict[str, dict[str, float]] = {}
    for cat in ["self", "peer", "exploitable", "baseline"]:
        rewards = category_rewards.get(cat, [])
        if rewards:
            aggregated[cat] = {
                "reward_mean": float(np.mean(rewards)),
                "reward_count": len(rewards),
                "game_count": category_games.get(cat, 0),
            }

    return aggregated


def _add_league_metrics(
    metrics: dict,
    league_ctx: LeagueContext,
    match_results: list[Any],
    collected_results: list["RolloutResult"] | None = None,
) -> None:
    """Add league training metrics to dict."""
    metrics.update(
        {
            "league/num_checkpoints": league_ctx.registry.num_checkpoints,
            "league/best_elo": league_ctx.registry.best_elo,
            "league/latest_step": league_ctx.registry.latest_step,
        }
    )

    # Add PFSP distribution metrics
    if match_results:
        pfsp_stats = league_ctx.matchmaker.get_sampling_stats(match_results)
        category_rates = pfsp_stats.get("category_rates", {})
        hero_powers = pfsp_stats.get("hero_power_distribution", {})

        for category, rate in category_rates.items():
            metrics[f"pfsp/{category}_rate"] = rate

        total_games = sum(hero_powers.values()) or 1
        for power, count in hero_powers.items():
            metrics[f"pfsp/hero_{power.lower()}"] = count / total_games

    # Add per-opponent-type reward metrics
    if collected_results:
        opponent_metrics = aggregate_rewards_by_opponent_type(collected_results)
        for cat, cat_metrics in opponent_metrics.items():
            metrics[f"opponent/{cat}/reward_mean"] = cat_metrics["reward_mean"]
            metrics[f"opponent/{cat}/game_count"] = cat_metrics["game_count"]
            metrics[f"opponent/{cat}/trajectory_count"] = cat_metrics["reward_count"]

        # Add dumbbot benchmark metrics (DipNet paper: ~75% win rate is target)
        dumbbot_stats = aggregate_dumbbot_game_stats(collected_results)
        if dumbbot_stats:
            metrics["dumbbot/games_count"] = dumbbot_stats["games_count"]
            metrics["dumbbot/win_rate"] = dumbbot_stats["win_rate"]
            metrics["dumbbot/top3_rate"] = dumbbot_stats["top3_rate"]
            metrics["dumbbot/avg_sc_count"] = dumbbot_stats["avg_sc_count"]
            metrics["dumbbot/avg_placement"] = dumbbot_stats["avg_placement"]


# ============================================================================
# MODEL INITIALIZATION
# ============================================================================


def initialize_model_and_optimizer(
    cfg: ExperimentConfig,
) -> tuple[Any, Any, AutoTokenizer, GRPOLoss, AdaptiveKLController | None]:
    """Initialize model, optimizer, tokenizer, loss function, and optional KL controller."""
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_id)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )

    # Use configured alpha or default to 2x rank
    lora_alpha = cfg.lora_alpha if cfg.lora_alpha is not None else cfg.lora_rank * 2

    peft_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=lora_alpha,
        target_modules=cfg.lora_target_modules,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    logger.info(
        f"🔧 LoRA config: rank={cfg.lora_rank}, alpha={lora_alpha}, "
        f"modules={cfg.lora_target_modules}, dropout={cfg.lora_dropout}"
    )
    policy_model = get_peft_model(base_model, peft_config)
    policy_model.gradient_checkpointing_enable()  # pyright: ignore[reportCallIssue]
    logger.info("✅ Gradient checkpointing enabled")

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=cfg.learning_rate)

    # Create KL controller if warmup or adaptive control is enabled
    kl_controller: AdaptiveKLController | None = None
    if cfg.kl_beta_warmup_steps > 0 or cfg.kl_target is not None:
        kl_config = KLControllerConfig(
            initial_beta=cfg.kl_beta,
            warmup_steps=cfg.kl_beta_warmup_steps,
            target_kl=cfg.kl_target,
            horizon=cfg.kl_horizon,
            beta_min=cfg.kl_beta_min,
            beta_max=cfg.kl_beta_max,
        )
        kl_controller = AdaptiveKLController(kl_config)
        logger.info(
            f"🎛️ KL controller enabled: warmup={cfg.kl_beta_warmup_steps} steps, "
            f"target={cfg.kl_target}, beta={cfg.kl_beta}"
        )

    loss_fn = GRPOLoss(
        policy_model,  # pyright: ignore[reportArgumentType]
        beta=cfg.kl_beta,
        kl_controller=kl_controller,
        use_ppo_clipping=cfg.use_ppo_clipping,
        ppo_epsilon_low=cfg.ppo_epsilon_low,
        ppo_epsilon_high=cfg.ppo_epsilon_high,
        use_token_level_loss=cfg.use_token_level_loss,
        entropy_coef=cfg.entropy_coef,
        entropy_top_k=cfg.entropy_top_k,
        importance_sampling_correction=cfg.importance_sampling_correction,
        importance_sampling_mode=cfg.importance_sampling_mode,
        importance_sampling_cap=cfg.importance_sampling_cap,
    )
    if cfg.use_ppo_clipping:
        logger.info(
            f"📊 PPO clipping enabled: ε_low={cfg.ppo_epsilon_low}, ε_high={cfg.ppo_epsilon_high}"
        )
    if cfg.importance_sampling_correction:
        logger.info(
            f"📊 IS correction enabled: mode={cfg.importance_sampling_mode}, "
            f"cap={cfg.importance_sampling_cap}"
        )
    if cfg.use_token_level_loss:
        logger.info("📊 Token-level loss weighting enabled")
    if cfg.entropy_coef > 0:
        logger.info(f"📊 Entropy bonus enabled: coef={cfg.entropy_coef}, top_k={cfg.entropy_top_k}")

    return policy_model, optimizer, tokenizer, loss_fn, kl_controller


def initialize_wandb(
    cfg: ExperimentConfig, start_step: int, wandb_run_id: str | None, sim_years_per_step: float
) -> None:
    """Initialize or resume WandB run."""
    wandb_tags = [cfg.experiment_tag] if cfg.experiment_tag else []

    wandb_kwargs = {
        "project": cfg.wandb_project,
        "name": cfg.run_name,
        "tags": wandb_tags if wandb_tags else None,
        "config": {
            **cfg.model_dump(),
            "simulated_years_per_step": sim_years_per_step,
            "total_simulated_years": cfg.total_simulated_years,
        },
    }

    if wandb_run_id:
        wandb_kwargs["id"] = wandb_run_id
        wandb_kwargs["resume"] = "must"
        logger.info(f"🔄 Resuming existing WandB run: {wandb_run_id}")
    elif start_step > 0:
        wandb_kwargs["resume"] = "allow"
        logger.info("🔄 Attempting to resume WandB run by name (fallback)")

    wandb.init(**wandb_kwargs)


# ============================================================================
# PROFILING HELPERS
# ============================================================================


@contextmanager
def profile_section(step_profile: dict[str, Any] | None, name: str):
    """Context manager for profiling code sections."""
    if step_profile is None:
        yield
        return

    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        key = f"{name}_ms"
        step_profile[key] = step_profile.get(key, 0.0) + elapsed_ms


def setup_profiler(cfg: ExperimentConfig) -> tuple[Any | None, Path | None]:
    """Setup PyTorch profiler if enabled."""
    if cfg.profiling_mode not in {"trainer", "e2e"}:
        return None, None

    traces_root = TRACE_PATH / "trainer"
    trace_subdir = traces_root / (cfg.profile_run_name or cfg.run_name)
    trace_subdir.mkdir(parents=True, exist_ok=True)

    profiler = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(
            wait=1,
            warmup=1,
            active=max(1, cfg.profiling_trace_steps - 2),
            repeat=0,
        ),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        on_trace_ready=tensorboard_trace_handler(str(trace_subdir)),
    )
    profiler.__enter__()

    return profiler, trace_subdir


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================


@app.function(
    image=gpu_image,
    gpu="H100",
    volumes={str(VOLUME_PATH): volume, str(TRACE_PATH): trace_volume},
    timeout=60 * 60 * 24,
    retries=0,
    secrets=[
        modal.Secret.from_name("axiom-secrets"),
        modal.Secret.from_name("wandb-secret"),
    ],
    max_containers=1,
)
def train_grpo(config_dict: dict | None = None, **kwargs) -> dict:
    """
    Main GRPO training function.

    Args:
        config_dict: Optional dict of ExperimentConfig values
        **kwargs: Individual config overrides

    Returns:
        Dict with timing, throughput, and training metrics
    """
    # Build config
    config_values = {}
    if config_dict:
        config_values.update(config_dict)
    config_values.update({k: v for k, v in kwargs.items() if v is not None})

    if "run_name" not in config_values or config_values["run_name"] == "diplomacy-grpo-v1":
        config_values["run_name"] = f"grpo-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    cfg = ExperimentConfig(**config_values)
    sim_years_per_step = cfg.simulated_years_per_step

    # =========================================================================
    # PREFLIGHT VALIDATION - Run before any GPU work
    # =========================================================================
    # This catches config mismatches (like lora_rank > max_lora_rank) that would
    # otherwise cause silent timeouts and waste GPU hours.
    from src.utils.preflight import run_full_preflight

    preflight = run_full_preflight(cfg)
    preflight.log_warnings(logger)
    preflight.raise_if_failed()  # Hard error if validation fails
    logger.info("✅ Preflight validation passed")

    # Setup profiling
    profiler, trace_subdir = setup_profiler(cfg)
    profile_enabled = profiler is not None
    profile_snapshots: list[dict[str, float]] = []

    # Metrics collection
    metrics = {
        "config": cfg.model_dump(),
        "step_metrics": [],
        "timing": {},
    }

    benchmark_start = time.time()
    gpu_logger = GPUStatsLogger()
    gpu_logger.start(context=f"train_grpo_benchmark:{cfg.run_name}")

    try:
        logger.info(f"🚀 Starting GRPO Training: {cfg.run_name}")

        # Initialize model and optimizer
        model_load_start = time.time()
        policy_model, optimizer, tokenizer, loss_fn, kl_controller = initialize_model_and_optimizer(
            cfg
        )
        metrics["timing"]["model_load_s"] = time.time() - model_load_start
        logger.info(f"✅ Model loaded in {metrics['timing']['model_load_s']:.2f}s")

        # Setup checkpoint manager
        checkpoint_mgr = CheckpointManager(cfg.run_name, optimizer, policy_model)

        # Resume from checkpoint if needed
        start_step, wandb_run_id = _handle_checkpoint_resume(cfg, checkpoint_mgr)

        # Initialize WandB
        initialize_wandb(cfg, start_step, wandb_run_id, sim_years_per_step)

        # Initialize league training
        league_ctx = initialize_league_training(cfg)

        # Get Modal function handles
        run_rollout_fn = modal.Function.from_name("diplomacy-grpo", "run_rollout")
        InferenceEngineCls = modal.Cls.from_name("diplomacy-grpo", "InferenceEngine")
        evaluate_league_fn = modal.Function.from_name("diplomacy-grpo", "evaluate_league")
        evaluate_benchmarks_fn = modal.Function.from_name(
            "diplomacy-grpo", "evaluate_against_benchmarks"
        )

        # Initialize rollout manager
        rollout_mgr = RolloutManager(cfg, run_rollout_fn, league_ctx)

        # Prefill buffer
        initial_adapter = f"{cfg.run_name}/adapter_v{start_step}" if start_step > 0 else None
        if initial_adapter:
            logger.info(f"📦 Resuming - buffer will use adapter: {initial_adapter}")

        rollout_mgr.prefill_buffer(initial_adapter, cfg.buffer_depth)

        # Training state
        total_trajectories = 0
        all_rewards = []

        logger.info(
            f"🚀 Starting BUFFERED pipelined training loop (buffer_depth={cfg.buffer_depth})"
        )

        # Main training loop
        for step in range(start_step, cfg.total_steps):
            step_start = time.time()
            step_metrics: dict[str, Any] = {"step": step}
            step_profile: dict[str, Any] | None = {"step": step} if profile_enabled else None

            # Determine current adapter for rollouts
            # At step N, use the adapter trained through step N-1
            current_adapter = (
                f"{cfg.run_name}/adapter_v{step - 1}" if step >= 1 else initial_adapter
            )

            # Collect rollouts
            with stopwatch(f"Benchmark_Rollout_{step}"):
                collected, timing_stats = rollout_mgr.collect_rollouts(
                    cfg.num_groups_per_step, current_adapter, step
                )

            # Aggregate results
            raw_trajectories = []
            for result in collected:
                raw_trajectories.extend(result.data["trajectories"])

            extraction_stats = aggregate_extraction_stats(collected)
            game_stats = aggregate_game_stats(collected)
            match_results = [r.match_result for r in collected if r.match_result is not None]

            # Update metrics
            rollout_time = time.time() - step_start
            step_metrics.update(
                {
                    "rollout_time_s": rollout_time,
                    "raw_trajectories": len(raw_trajectories),
                    "rollout_lora": "mixed",
                    "buffer_depth_actual": len(rollout_mgr.buffer),
                    "containers_in_flight": rollout_mgr.containers_in_flight,
                    "container_cap": rollout_mgr.max_containers,
                    "extraction_stats": extraction_stats,
                    "failed_rollouts": timing_stats["failed_count"],
                    "rollouts_consumed": len(collected) + timing_stats["failed_count"],
                    "collection_polls": timing_stats["polls_without_progress"],
                    "max_volume_reload_s": timing_stats["max_volume_reload_s"],
                    "max_rollout_total_s": timing_stats["max_rollout_total_s"],
                }
            )

            if step_profile:
                step_profile["rollout_time_ms"] = rollout_time * 1000

            # Get cache stats
            cache_stats = _get_cache_stats(InferenceEngineCls, cfg.base_model_id, step)

            # Launch new rollouts to maintain buffer
            rollouts_spawned = 0
            steps_remaining = cfg.total_steps - step - 1
            rollouts_consumed = len(collected) + int(timing_stats["failed_count"])

            if steps_remaining >= cfg.buffer_depth and rollouts_consumed > 0:
                new_adapter = _save_and_register_adapter(
                    step, cfg, policy_model, checkpoint_mgr, league_ctx, evaluate_league_fn
                )
                rollouts_spawned = rollout_mgr.spawn_replacement_rollouts(
                    rollouts_consumed, new_adapter, step + cfg.buffer_depth
                )
                logger.info(
                    f"🔀 Launched replacements for step {step + cfg.buffer_depth} "
                    f"(replacing {rollouts_consumed}, using {new_adapter or 'base model'})"
                )

            step_metrics["rollouts_spawned"] = rollouts_spawned

            # Skip if no trajectories
            if not raw_trajectories:
                logger.warning(f"Step {step}: No trajectories, skipping")
                metrics["step_metrics"].append(step_metrics)
                continue

            total_trajectories += len(raw_trajectories)
            all_rewards.extend([t["reward"] for t in raw_trajectories])

            # Process trajectories
            process_start = time.time()
            with profile_section(step_profile, "tokenize"):
                batch_data, traj_stats = process_trajectories(
                    raw_trajectories,
                    tokenizer,
                    advantage_clip=cfg.advantage_clip,
                    advantage_min_std=cfg.advantage_min_std,
                )

            step_metrics["process_time_s"] = time.time() - process_start
            step_metrics["processed_trajectories"] = len(batch_data)

            if not batch_data:
                logger.warning(f"Step {step}: No valid batches")
                metrics["step_metrics"].append(step_metrics)
                continue

            # Training
            training_start = time.time()
            train_metrics = _run_training_step(
                batch_data, optimizer, loss_fn, policy_model, cfg, step_profile
            )
            training_time = time.time() - training_start

            # Update KL controller (if enabled) and get diagnostics
            kl_diagnostics: dict[str, float] = {}
            if kl_controller is not None:
                kl_diagnostics = kl_controller.step_update(train_metrics["kl"])

            # Update metrics
            step_metrics.update(
                {
                    "training_time_s": training_time,
                    "loss": train_metrics["loss"],
                    "kl": train_metrics["kl"],
                    "kl_max": train_metrics["kl_max"],
                    "kl_min": train_metrics["kl_min"],
                    "kl_std": train_metrics["kl_std"],
                    "effective_beta": train_metrics["effective_beta"],
                    "grad_norm": train_metrics["grad_norm"],
                    "used_cached_ref_logprobs": train_metrics["used_cached_ref_logprobs"],
                    # PPO clipping metrics (DAPO-style)
                    "ratio_mean": train_metrics["ratio_mean"],
                    "ratio_std": train_metrics["ratio_std"],
                    "ratio_clipped_fraction": train_metrics["ratio_clipped_fraction"],
                    "rollout_logprobs_rate": train_metrics["rollout_logprobs_rate"],
                    # Entropy metrics
                    "entropy_mean": train_metrics["entropy_mean"],
                    "entropy_std": train_metrics["entropy_std"],
                    # Importance sampling metrics (vLLM-HF mismatch correction)
                    "is_ratio_mean": train_metrics["is_ratio_mean"],
                    "is_ratio_std": train_metrics["is_ratio_std"],
                    "is_masked_fraction": train_metrics["is_masked_fraction"],
                    "reward_mean": traj_stats.reward_mean,
                    "reward_std": traj_stats.reward_std,
                    "total_tokens": traj_stats.total_tokens,
                    "total_time_s": time.time() - step_start,
                    # Effective speedup from buffered pipeline: time saved by not waiting for rollouts
                    # Positive = training is bottleneck (rollouts are fast/prefetched)
                    # Zero = rollout is bottleneck (waiting for results)
                    "pipeline_overlap_s": max(0, training_time - rollout_time) if step > 0 else 0,
                    # KL controller diagnostics (if enabled)
                    **{f"kl_ctrl_{k}": v for k, v in kl_diagnostics.items()},
                    # Trajectory stats for advantage/skip diagnostics
                    "traj_stats": traj_stats,
                }
            )

            if step_profile:
                step_profile.update(
                    {
                        "training_time_ms": training_time * 1000,
                        "process_time_ms": step_metrics["process_time_s"] * 1000,
                        "trajectories": len(batch_data),
                        "tokens": traj_stats.total_tokens,
                        "pipeline_overlap_ms": step_metrics["pipeline_overlap_s"] * 1000,
                    }
                )
                profile_snapshots.append(step_profile)

            metrics["step_metrics"].append(step_metrics)

            # Log to console
            extraction_rate_pct = extraction_stats["extraction_rate"] * 100
            beta_str = (
                f" | β={train_metrics['effective_beta']:.4f}" if kl_controller is not None else ""
            )
            logger.info(
                f"Step {step}: loss={train_metrics['loss']:.4f} | kl={train_metrics['kl']:.4f}{beta_str} | "
                f"reward={traj_stats.reward_mean:.2f}±{traj_stats.reward_std:.2f} | "
                f"extraction={extraction_rate_pct:.1f}% | "
                f"trajectories={len(batch_data)} | time={step_metrics['total_time_s']:.2f}s"
            )

            # Log to WandB
            wandb_metrics = build_wandb_metrics(
                step,
                step_metrics,
                extraction_stats,
                game_stats,
                traj_stats,
                cache_stats,
                league_ctx,
                match_results,
                sim_years_per_step,
                collected_results=collected,  # For per-opponent-type metrics
            )
            # Update TrueSkill ratings from rollout game outcomes (if league training enabled)
            # Note: We add rating metrics to wandb_metrics BEFORE logging to avoid
            # double-incrementing WandB's step counter (each wandb.log() call = 1 step)
            if league_ctx is not None:
                trueskill_updates = update_trueskill_from_rollouts(collected, league_ctx)
                if trueskill_updates:
                    # Add count and individual agent ratings to wandb_metrics
                    wandb_metrics["trueskill/rollout_updates_count"] = len(trueskill_updates)
                    for agent_name, (new_mu, new_sigma) in trueskill_updates.items():
                        # Sanitize agent name for WandB (replace / with _)
                        safe_name = agent_name.replace("/", "_")
                        display_rating = new_mu - 3 * new_sigma
                        wandb_metrics[f"trueskill/{safe_name}_mu"] = new_mu
                        wandb_metrics[f"trueskill/{safe_name}_sigma"] = new_sigma
                        wandb_metrics[f"trueskill/{safe_name}_rating"] = display_rating
                        # Also log under elo/ for backwards compatibility
                        wandb_metrics[f"elo/{safe_name}"] = (new_mu - 25.0) * 40 + 1000

            # Spawn benchmark evaluation if enabled and due
            if (
                cfg.benchmark_eval_every_n_steps > 0
                and step > 0
                and step % cfg.benchmark_eval_every_n_steps == 0
            ):
                adapter_rel_path = f"{cfg.run_name}/adapter_v{step}"
                logger.info(f"📊 Spawning benchmark evaluation for step {step}")
                evaluate_benchmarks_fn.spawn(
                    challenger_path=adapter_rel_path,
                    games_per_benchmark=cfg.benchmark_games_per_opponent,
                    max_years=cfg.benchmark_max_years,
                    model_id=cfg.base_model_id,
                    wandb_run_id=wandb.run.id if wandb.run else None,
                    training_step=step,
                    use_quick_suite=True,  # Use quick suite for frequent eval during training
                    show_valid_moves=cfg.show_valid_moves,
                    compact_prompts=cfg.compact_prompts,
                    prefix_cache_optimized=cfg.prefix_cache_optimized,
                    temperature=cfg.temperature,
                    max_new_tokens=cfg.max_new_tokens,
                )

            wandb.log(wandb_metrics)

            if profiler:
                profiler.step()

            # Periodic checkpoint save
            if cfg.save_state_every_n_steps > 0 and (step + 1) % cfg.save_state_every_n_steps == 0:
                checkpoint_mgr.save(step + 1, cfg, wandb.run.id if wandb.run else None)

        # Save final adapter
        final_path = checkpoint_mgr.save_adapter(cfg.total_steps)
        logger.info(f"💾 Saved final adapter to {final_path}")

        # Build final metrics
        total_time = time.time() - benchmark_start
        metrics["timing"]["total_s"] = total_time

        summary = _build_summary_metrics(
            cfg,
            total_trajectories,
            all_rewards,
            total_time,
            metrics,
            profile_snapshots,
            trace_subdir,
        )

        logger.info(f"\n{'=' * 60}")
        logger.info("🏁 BENCHMARK COMPLETE")
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(f"Trajectories: {total_trajectories}")
        logger.info(f"Throughput: {summary['trajectories_per_second']:.2f} traj/s")
        logger.info(f"Pipeline overlap: {summary['pipeline_overlap_total_s']:.2f}s")
        logger.info(f"{'=' * 60}")

        wandb.log({"benchmark/complete": True, **summary})

        return summary

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        wandb.log({"benchmark/error": str(e)})
        raise

    finally:
        logger.info("🔄 Training complete. Containers will scale down after inactivity.")
        gpu_logger.stop()
        if profiler:
            profiler.__exit__(None, None, None)
        asyncio.run(axiom.flush())
        wandb.finish()


# ============================================================================
# TRAINING STEP HELPERS
# ============================================================================


def _handle_checkpoint_resume(
    cfg: ExperimentConfig, checkpoint_mgr: CheckpointManager
) -> tuple[int, str | None]:
    """Handle checkpoint resumption logic. Returns (start_step, wandb_run_id)."""
    start_step = 0
    wandb_run_id: str | None = None
    volume.reload()

    if cfg.resume_from_run:
        try:
            start_step, wandb_run_id = checkpoint_mgr.load(
                cfg.resume_from_run, cfg.resume_from_step
            )
            if cfg.resume_from_run != cfg.run_name:
                logger.info(f"📦 Forking from {cfg.resume_from_run} to new run {cfg.run_name}")
                wandb_run_id = None
        except FileNotFoundError as e:
            logger.error(f"❌ Resume failed: {e}")
            raise

    elif not cfg.disable_auto_resume:
        # Auto-resume: check for existing checkpoints
        run_path = MODELS_PATH / cfg.run_name
        pattern = str(run_path / "training_state_v*.pt")
        existing_states = glob.glob(pattern)

        if existing_states:
            logger.warning(
                f"⚠️ Found {len(existing_states)} existing checkpoints for {cfg.run_name}"
            )
            logger.warning("🔄 AUTO-RESUMING from crash (use --disable-auto-resume to start fresh)")
            try:
                start_step, wandb_run_id = checkpoint_mgr.load(cfg.run_name, cfg.resume_from_step)
            except FileNotFoundError as e:
                logger.error(f"❌ Auto-resume failed: {e}, starting fresh")

    return start_step, wandb_run_id


def _get_cache_stats(inference_engine_cls: Any, model_id: str, step: int) -> dict | None:
    """Query cache statistics from inference engine."""
    try:
        result = inference_engine_cls(model_id=model_id).get_cache_stats.remote()
        cache_stats = result.get("cumulative", {})
        vllm_stats = result.get("vllm_stats")

        # Build info string
        vllm_info = "None"
        if vllm_stats:
            vllm_info = (
                f"queries={vllm_stats.get('queries')}, "
                f"hits={vllm_stats.get('hits')}, "
                f"hit_rate={vllm_stats.get('hit_rate', 0):.1%}"
            )

        logger.info(
            f"📊 Cache stats for step {step}: "
            f"batches={cache_stats.get('batches_processed', 0)}, "
            f"prompt_tokens={cache_stats.get('total_prompt_tokens', 0)}, "
            f"vllm=[{vllm_info}]"
        )
        return cache_stats
    except Exception as e:
        logger.warning(f"Could not get cache stats: {e}")
        return None


def _save_and_register_adapter(
    step: int,
    cfg: ExperimentConfig,
    policy_model: Any,
    checkpoint_mgr: CheckpointManager,
    league_ctx: LeagueContext | None,
    evaluate_league_fn: Any,
) -> str | None:
    """Save adapter and register with league if applicable. Returns adapter path.

    Saves adapter_v{step} after training step N completes, so step N+1 can use it.
    """

    adapter_rel_path = f"{cfg.run_name}/adapter_v{step}"
    adapter_full_path = MODELS_PATH / cfg.run_name / f"adapter_v{step}"

    # Save adapter with verification to prevent corrupted league entries
    policy_model.save_pretrained(str(adapter_full_path))

    # Verify critical adapter files were written before committing
    # LoRA adapters should have at minimum: adapter_config.json and adapter_model.safetensors
    required_files = ["adapter_config.json"]
    safetensors_file = adapter_full_path / "adapter_model.safetensors"
    bin_file = adapter_full_path / "adapter_model.bin"

    for required_file in required_files:
        file_path = adapter_full_path / required_file
        if not file_path.exists():
            raise RuntimeError(
                f"Adapter save failed: {file_path} not found after save_pretrained(). "
                "This would corrupt the league registry."
            )

    # Check that at least one model file exists (safetensors preferred, bin as fallback)
    if not safetensors_file.exists() and not bin_file.exists():
        raise RuntimeError(
            f"Adapter save failed: neither {safetensors_file} nor {bin_file} found. "
            "This would corrupt the league registry."
        )

    # Commit volume only after verification passes
    volume.commit()
    logger.info(f"💾 Saved and verified adapter to {adapter_full_path}")

    if league_ctx:
        add_checkpoint_to_league(step, adapter_rel_path, cfg, league_ctx, evaluate_league_fn)

    return adapter_rel_path


def _run_training_step(
    batch_data: list,
    optimizer: torch.optim.Optimizer,
    loss_fn: GRPOLoss,
    policy_model: Any,
    cfg: ExperimentConfig,
    step_profile: dict[str, Any] | None,
) -> dict[str, Any]:
    """Run a single training step. Returns training metrics."""
    optimizer.zero_grad()

    accum_loss = 0.0
    accum_kl = 0.0
    accum_kl_max = 0.0
    accum_kl_min = float("inf")
    accum_kl_std = 0.0
    last_effective_beta = 0.0
    num_chunks = 0
    used_cached_ref = False  # Track if ref logprobs came from rollouts vs computed here
    # PPO clipping metrics (DAPO-style)
    accum_ratio_mean = 0.0
    accum_ratio_std = 0.0
    accum_ratio_clipped = 0.0
    accum_rollout_logprobs_rate = 0.0
    # Entropy metrics (GTPO-style)
    accum_entropy_mean = 0.0
    accum_entropy_std = 0.0
    # Importance sampling metrics (vLLM-HF mismatch correction)
    accum_is_ratio_mean = 0.0
    accum_is_ratio_std = 0.0
    accum_is_masked_fraction = 0.0
    # Use ceiling division to correctly count partial chunks at the end
    total_chunks = (len(batch_data) + cfg.chunk_size - 1) // cfg.chunk_size

    for i in range(0, len(batch_data), cfg.chunk_size):
        chunk = batch_data[i : i + cfg.chunk_size]
        if not chunk:
            break

        with profile_section(step_profile, "loss_forward"):
            loss_output = loss_fn.compute_loss(chunk)

        scaled_loss = loss_output.loss / max(1, total_chunks)

        with profile_section(step_profile, "backward"):
            scaled_loss.backward()

        accum_loss += loss_output.loss.item()
        accum_kl += loss_output.kl
        accum_kl_max = max(accum_kl_max, loss_output.kl_max)
        accum_kl_min = min(accum_kl_min, loss_output.kl_min)
        accum_kl_std += loss_output.kl_std
        last_effective_beta = loss_output.effective_beta
        num_chunks += 1
        # Track if any chunk used cached ref logprobs (should be all or none)
        if loss_output.used_cached_ref_logprobs:
            used_cached_ref = True
        # PPO clipping metrics
        accum_ratio_mean += loss_output.ratio_mean
        accum_ratio_std += loss_output.ratio_std
        accum_ratio_clipped += loss_output.ratio_clipped_fraction
        accum_rollout_logprobs_rate += loss_output.rollout_logprobs_rate
        # Entropy metrics
        accum_entropy_mean += loss_output.entropy_mean
        accum_entropy_std += loss_output.entropy_std
        # Importance sampling metrics
        accum_is_ratio_mean += loss_output.is_ratio_mean
        accum_is_ratio_std += loss_output.is_ratio_std
        accum_is_masked_fraction += loss_output.is_masked_fraction

    with profile_section(step_profile, "optimizer_step"):
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy_model.parameters(), cfg.max_grad_norm
        ).item()
        optimizer.step()

    # Note: reported loss is consistent with gradient scaling because:
    # - Gradients are scaled by 1/total_chunks before backward()
    # - Loss is averaged over num_chunks (which equals total_chunks)
    # Both give the mean loss per sample group.
    return {
        "loss": accum_loss / max(1, num_chunks),
        "kl": accum_kl / max(1, num_chunks),
        "kl_max": accum_kl_max,
        "kl_min": accum_kl_min if accum_kl_min != float("inf") else 0.0,
        "kl_std": accum_kl_std / max(1, num_chunks),
        "effective_beta": last_effective_beta,
        "grad_norm": grad_norm,
        "used_cached_ref_logprobs": used_cached_ref,
        # PPO clipping metrics (DAPO-style)
        "ratio_mean": accum_ratio_mean / max(1, num_chunks),
        "ratio_std": accum_ratio_std / max(1, num_chunks),
        "ratio_clipped_fraction": accum_ratio_clipped / max(1, num_chunks),
        "rollout_logprobs_rate": accum_rollout_logprobs_rate / max(1, num_chunks),
        # Entropy metrics (GTPO-style)
        "entropy_mean": accum_entropy_mean / max(1, num_chunks),
        "entropy_std": accum_entropy_std / max(1, num_chunks),
        # Importance sampling metrics (vLLM-HF mismatch correction)
        "is_ratio_mean": accum_is_ratio_mean / max(1, num_chunks),
        "is_ratio_std": accum_is_ratio_std / max(1, num_chunks),
        "is_masked_fraction": accum_is_masked_fraction / max(1, num_chunks),
    }


def _build_summary_metrics(
    cfg: ExperimentConfig,
    total_trajectories: int,
    all_rewards: list[float],
    total_time: float,
    metrics: dict,
    profile_snapshots: list,
    trace_subdir: Path | None,
) -> dict:
    """Build final summary metrics."""
    total_simulated_years = (
        cfg.total_steps
        * cfg.num_groups_per_step
        * cfg.samples_per_group
        * cfg.rollout_horizon_years
    )

    total_pipeline_overlap = sum(m.get("pipeline_overlap_s", 0) for m in metrics["step_metrics"])

    summary = {
        "total_trajectories": total_trajectories,
        "total_simulated_years": total_simulated_years,
        "trajectories_per_second": total_trajectories / max(0.001, total_time),
        "simulated_years_per_second": total_simulated_years / max(0.001, total_time),
        "reward_mean": sum(all_rewards) / len(all_rewards) if all_rewards else 0,
        "reward_min": min(all_rewards) if all_rewards else 0,
        "reward_max": max(all_rewards) if all_rewards else 0,
        "pipeline_overlap_total_s": total_pipeline_overlap,
        "run_name": cfg.run_name,
        "profiling_mode": cfg.profiling_mode,
        "profile_snapshots": profile_snapshots if profile_snapshots else None,
        "trace_dir": str(trace_subdir) if trace_subdir else None,
    }

    if metrics["step_metrics"]:
        final = metrics["step_metrics"][-1]
        summary.update(
            {
                "final_loss": final.get("loss"),
                "final_kl": final.get("kl"),
                "final_reward_mean": final.get("reward_mean"),
            }
        )

    return summary
