import asyncio
import glob
import os
import random
import tempfile
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

import modal
import numpy as np
import torch
import wandb
from peft import LoraConfig, get_peft_model
from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.apps.common.images import gpu_image
from src.apps.common.volumes import MODELS_PATH, TRACE_PATH, VOLUME_PATH, trace_volume, volume
from src.training.loss import GRPOLoss
from src.training.trainer import process_trajectories
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

                # Restore seed
                saved_seed = state.get("seed")
                if saved_seed is not None:
                    random.seed(saved_seed)
                    np.random.seed(saved_seed)
                    torch.manual_seed(saved_seed)
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
    parent_key = f"{cfg.run_name}/adapter_v{step - 1}" if step > 1 else "base_model"

    if not should_add_to_league(step, league_ctx.registry):
        return

    league_ctx.registry.add_checkpoint(
        name=checkpoint_key,
        path=adapter_rel_path,
        step=step,
        parent=parent_key,
    )
    volume.commit()
    logger.info(f"🏆 Added checkpoint {checkpoint_key} to league")

    # Spawn Elo evaluation if enabled
    if cfg.elo_eval_every_n_steps > 0 and step % cfg.elo_eval_every_n_steps == 0:
        logger.info(f"🎯 Spawning Elo evaluation for {checkpoint_key}")
        registry_path_str = str(
            f"/data/league_{cfg.run_name}.json"
            if not cfg.league_registry_path
            else cfg.league_registry_path
        )

        evaluate_league_fn.spawn(
            challenger_path=adapter_rel_path,
            league_registry_path=registry_path_str,
            games_per_opponent=cfg.elo_eval_games_per_opponent,
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
    """Manages rollout spawning, collection, and buffering."""

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

    def spawn_batch(
        self, hero_adapter_path: str | None, current_step: int
    ) -> tuple[list[Any], list[Any]]:
        """Spawn a batch of rollouts with appropriate opponent sampling."""
        if self.league_ctx:
            reload_league_registry(self.league_ctx, current_step)

        handles = []
        match_results = []

        for _ in range(self.cfg.num_groups_per_step):
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

        # Sample opponents
        if self.league_ctx.registry.num_checkpoints > 0:
            hero_agent_name = hero_adapter_path or "base_model"
            hero_info = self.league_ctx.registry.get_agent(hero_agent_name)
            estimated_hero_elo = hero_info.elo if hero_info else self.league_ctx.registry.best_elo

            match_result = self.league_ctx.matchmaker.sample_opponents(
                hero_agent=hero_agent_name,
                hero_power=hero_power,
                num_opponents=6,
                hero_elo_override=estimated_hero_elo,
                hero_adapter_path=hero_adapter_path,
            )
        else:
            match_result = self.league_ctx.matchmaker.get_cold_start_opponents(hero_power)

        # Override hero adapter
        power_adapters = match_result.power_adapters.copy()
        power_adapters[hero_power] = hero_adapter_path

        handle = self.run_rollout_fn.spawn(
            self.cfg.model_dump(),
            power_adapters=power_adapters,
            hero_power=hero_power,
        )

        return handle, match_result

    def _spawn_legacy_rollout(self, hero_adapter_path: str | None) -> Any:
        """Spawn a single legacy (non-league) rollout."""
        return self.run_rollout_fn.spawn(self.cfg.model_dump(), lora_name=hero_adapter_path)

    def prefill_buffer(self, adapter_path: str | None, buffer_depth: int) -> None:
        """Prefill rollout buffer with initial batches."""
        for _ in range(buffer_depth):
            handles, match_results = self.spawn_batch(adapter_path, current_step=0)
            for h, mr in zip(handles, match_results, strict=False):
                self.buffer.append((h, adapter_path, mr))

        logger.info(f"📦 Buffer initialized: {buffer_depth} batches ({len(self.buffer)} rollouts)")

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
            # Spawn emergency rollout if buffer empty
            if not self.buffer:
                logger.warning("Rollout buffer unexpectedly empty; spawning emergency rollout")
                handles, match_results = self.spawn_batch(current_adapter, step)
                for h, mr in zip(handles, match_results, strict=False):
                    self.buffer.append((h, current_adapter, mr))
                polls_without_progress = 0

            handle, adapter_used, match_result = self.buffer.popleft()

            # Adaptive timeout
            timeout = (
                0.1 if polls_without_progress < MAX_POLLS else min(30.0, deadline - time.time())
            )

            try:
                result = handle.get(timeout=timeout)
                polls_without_progress = 0

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
                    failed_count += 1
                    polls_without_progress = 0
                    logger.warning(f"⚠️ Rollout failed (skipping): {type(e).__name__}: {e}")

        return collected, {
            "max_volume_reload_s": max_volume_reload_s,
            "max_rollout_total_s": max_rollout_total_s,
            "failed_count": failed_count,
            "polls_without_progress": polls_without_progress,
        }

    def spawn_replacement_rollouts(
        self, consumed_count: int, adapter_path: str | None, step: int
    ) -> int:
        """Spawn rollouts to replace consumed ones. Returns count spawned."""
        batches_needed = max(
            1, (consumed_count + self.cfg.num_groups_per_step - 1) // self.cfg.num_groups_per_step
        )

        spawned = 0
        for _ in range(batches_needed):
            handles, match_results = self.spawn_batch(adapter_path, step)
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


def build_wandb_metrics(
    step: int,
    step_metrics: dict[str, Any],
    extraction_stats: ExtractionStats,
    traj_stats: Any,
    cache_stats: dict | None,
    league_ctx: LeagueContext | None,
    match_results: list[Any],
    sim_years_per_step: float,
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
        # Rollout timing
        "rollout/max_volume_reload_s": step_metrics["max_volume_reload_s"],
        "rollout/max_total_s": step_metrics["max_rollout_total_s"],
        "rollout/failed_count": step_metrics["failed_rollouts"],
        # Buffer health
        "buffer/depth": step_metrics["buffer_depth_actual"],
        "buffer/consumed": step_metrics["rollouts_consumed"],
        "buffer/spawned": step_metrics.get("rollouts_spawned", 0),
        "buffer/collection_polls": step_metrics["collection_polls"],
        # Extraction
        "extraction/rate": extraction_stats["extraction_rate"],
        "extraction/orders_expected": extraction_stats["orders_expected"],
        "extraction/orders_extracted": extraction_stats["orders_extracted"],
        "extraction/empty_responses": extraction_stats["empty_responses"],
        "extraction/partial_responses": extraction_stats["partial_responses"],
        # Power law
        "power_law/cumulative_simulated_years": cumulative_sim_years,
        "power_law/simulated_years_per_step": sim_years_per_step,
        "power_law/reward_at_compute": traj_stats.reward_mean,
    }

    # Add cache metrics
    if cache_stats:
        _add_cache_metrics(metrics, cache_stats)

    # Add league metrics
    if league_ctx:
        _add_league_metrics(metrics, league_ctx, match_results)

    return metrics


def _add_cache_metrics(metrics: dict, cache_stats: dict) -> None:
    """Add cache statistics to metrics dict."""
    total_queries = cache_stats.get("total_queries", 0)
    total_hits = cache_stats.get("total_hits", 0)
    total_prompt_tokens = cache_stats.get("total_prompt_tokens", 0)
    batches = cache_stats.get("batches_processed", 0)
    batch_size_total = cache_stats.get("batch_size_total", 0)
    real_stats = cache_stats.get("real_stats_available", False)

    cache_metrics = {
        "cache/prompt_tokens": total_prompt_tokens,
        "cache/batches": batches,
        "cache/total_requests": batch_size_total,
    }

    if total_queries > 0:
        cache_hit_rate = total_hits / total_queries
        cache_metrics.update(
            {
                "cache/hit_rate": cache_hit_rate,
                "cache/total_queries": total_queries,
                "cache/total_hits": total_hits,
            }
        )

    cache_metrics["cache/real_stats"] = 1 if real_stats else 0
    metrics.update(cache_metrics)


def _add_league_metrics(metrics: dict, league_ctx: LeagueContext, match_results: list[Any]) -> None:
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


# ============================================================================
# MODEL INITIALIZATION
# ============================================================================


def initialize_model_and_optimizer(
    cfg: ExperimentConfig,
) -> tuple[Any, Any, AutoTokenizer, GRPOLoss]:
    """Initialize model, optimizer, tokenizer, and loss function."""
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_id)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )

    peft_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    policy_model = get_peft_model(base_model, peft_config)
    policy_model.gradient_checkpointing_enable()  # pyright: ignore[reportCallIssue]
    logger.info("✅ Gradient checkpointing enabled")

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=cfg.learning_rate)
    loss_fn = GRPOLoss(policy_model, beta=0.04)  # pyright: ignore[reportArgumentType]

    return policy_model, optimizer, tokenizer, loss_fn


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
        policy_model, optimizer, tokenizer, loss_fn = initialize_model_and_optimizer(cfg)
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
            current_adapter = f"{cfg.run_name}/adapter_v{step}" if step >= 1 else initial_adapter

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
            match_results = [r.match_result for r in collected if r.match_result is not None]

            # Update metrics
            rollout_time = time.time() - step_start
            step_metrics.update(
                {
                    "rollout_time_s": rollout_time,
                    "raw_trajectories": len(raw_trajectories),
                    "rollout_lora": "mixed",
                    "buffer_depth_actual": len(rollout_mgr.buffer),
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
                batch_data, traj_stats = process_trajectories(raw_trajectories, tokenizer)

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

            # Update metrics
            step_metrics.update(
                {
                    "training_time_s": training_time,
                    "loss": train_metrics["loss"],
                    "kl": train_metrics["kl"],
                    "grad_norm": train_metrics["grad_norm"],
                    "reward_mean": traj_stats.reward_mean,
                    "reward_std": traj_stats.reward_std,
                    "total_tokens": traj_stats.total_tokens,
                    "total_time_s": time.time() - step_start,
                    "pipeline_overlap_s": max(0, training_time - rollout_time) if step > 0 else 0,
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
            logger.info(
                f"Step {step}: loss={train_metrics['loss']:.4f} | kl={train_metrics['kl']:.4f} | "
                f"reward={traj_stats.reward_mean:.2f}±{traj_stats.reward_std:.2f} | "
                f"extraction={extraction_rate_pct:.1f}% | "
                f"trajectories={len(batch_data)} | time={step_metrics['total_time_s']:.2f}s"
            )

            # Log to WandB
            wandb_metrics = build_wandb_metrics(
                step,
                step_metrics,
                extraction_stats,
                traj_stats,
                cache_stats,
                league_ctx,
                match_results,
                sim_years_per_step,
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
        logger.info(
            f"📊 Cache stats for step {step}: "
            f"batches={cache_stats.get('batches_processed', 0)}, "
            f"prompt_tokens={cache_stats.get('total_prompt_tokens', 0)}, "
            f"vllm_stats={'available' if vllm_stats else 'None'}"
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
    """Save adapter and register with league if applicable. Returns adapter path."""
    if step < 1:
        return None

    adapter_rel_path = f"{cfg.run_name}/adapter_v{step}"
    adapter_full_path = MODELS_PATH / cfg.run_name / f"adapter_v{step}"
    policy_model.save_pretrained(str(adapter_full_path))
    volume.commit()
    logger.info(f"💾 Saved adapter to {adapter_full_path}")

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
) -> dict[str, float]:
    """Run a single training step. Returns training metrics."""
    optimizer.zero_grad()

    accum_loss = 0.0
    accum_kl = 0.0
    num_chunks = 0

    for i in range(0, len(batch_data), cfg.chunk_size):
        chunk = batch_data[i : i + cfg.chunk_size]
        if not chunk:
            break

        with profile_section(step_profile, "loss_forward"):
            loss_output = loss_fn.compute_loss(chunk)

        scaled_loss = loss_output.loss / max(1, len(batch_data) // cfg.chunk_size)

        with profile_section(step_profile, "backward"):
            scaled_loss.backward()

        accum_loss += loss_output.loss.item()
        accum_kl += loss_output.kl
        num_chunks += 1

    with profile_section(step_profile, "optimizer_step"):
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy_model.parameters(), cfg.max_grad_norm
        ).item()
        optimizer.step()

    return {
        "loss": accum_loss / max(1, num_chunks),
        "kl": accum_kl / max(1, num_chunks),
        "grad_norm": grad_norm,
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
