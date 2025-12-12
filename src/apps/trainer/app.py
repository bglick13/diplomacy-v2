import asyncio
import time
from datetime import datetime
from typing import Any

import modal
import torch
import wandb
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.apps.common.images import gpu_image
from src.apps.common.volumes import MODELS_PATH, TRACE_PATH, VOLUME_PATH, trace_volume, volume
from src.training.loss import GRPOLoss
from src.training.trainer import process_trajectories
from src.utils.config import ExperimentConfig
from src.utils.observability import GPUStatsLogger, axiom, logger, stopwatch

app = modal.App("diplomacy-grpo-trainer")


@app.function(
    image=gpu_image,
    gpu="H100",
    volumes={str(VOLUME_PATH): volume, str(TRACE_PATH): trace_volume},
    timeout=60 * 60 * 24,  # 24 hours max
    retries=0,  # CRITICAL: Don't auto-retry training - would restart from scratch
    secrets=[
        modal.Secret.from_name("axiom-secrets"),
        modal.Secret.from_name("wandb-secret"),
    ],
)
def train_grpo(config_dict: dict | None = None, **kwargs) -> dict:
    """
    Main GRPO training function.

    This is the unified training entrypoint that supports both production runs
    and profiling/benchmarking. All parameters come from ExperimentConfig.

    Args:
        config_dict: Optional dict of ExperimentConfig values. If provided,
                    these take precedence over kwargs.
        **kwargs: Individual config overrides (matched to ExperimentConfig fields).
                 Common overrides:
                 - total_steps: Number of training steps
                 - num_groups_per_step: Rollout groups per step (G in GRPO)
                 - samples_per_group: Samples per group (N in GRPO)
                 - rollout_horizon_years: Years to simulate per rollout
                 - learning_rate: Learning rate for AdamW optimizer
                 - profiling_mode: "rollout", "trainer", or "e2e" for profiling
                 - experiment_tag: Tag for grouping runs in WandB

    Returns:
        Dict with timing, throughput, and training metrics

    Example:
        # From CLI via Modal
        modal run app.py::train_grpo --total-steps 10 --learning-rate 1e-5

        # From Python
        train_grpo.remote(config_dict={"total_steps": 10, "learning_rate": 1e-5})
    """

    # Build config from dict or kwargs
    # Priority: config_dict > kwargs > defaults
    config_values = {}
    if config_dict:
        config_values.update(config_dict)
    config_values.update({k: v for k, v in kwargs.items() if v is not None})

    # Generate run_name if not provided
    if "run_name" not in config_values or config_values["run_name"] == "diplomacy-grpo-v1":
        config_values["run_name"] = f"grpo-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    cfg = ExperimentConfig(**config_values)

    # Pre-compute simulated years metrics for power law analysis
    sim_years_per_step = cfg.simulated_years_per_step

    # Profiling setup
    profile_enabled = cfg.profiling_mode in {"trainer", "e2e"}
    profile_snapshots: list[dict[str, float]] = []

    from contextlib import contextmanager

    from torch.profiler import (
        ProfilerActivity,
        tensorboard_trace_handler,
    )
    from torch.profiler import (
        profile as torch_profile,
    )
    from torch.profiler import (
        schedule as profiler_schedule,
    )

    @contextmanager
    def profile_section(step_profile: dict[str, Any], name: str):
        if not profile_enabled:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            key = f"{name}_ms"
            step_profile[key] = step_profile.get(key, 0.0) + elapsed_ms

    # Metrics collection
    metrics = {
        "config": cfg.model_dump(),
        "step_metrics": [],
        "timing": {},
    }

    benchmark_start = time.time()
    gpu_logger = GPUStatsLogger()
    gpu_logger.start(context=f"train_grpo_benchmark:{cfg.run_name}")
    profiler = None
    trace_subdir = None
    if profile_enabled:
        traces_root = TRACE_PATH / "trainer"
        trace_subdir = traces_root / (cfg.profile_run_name or cfg.run_name)  # type: ignore[arg-type]
        trace_subdir.mkdir(parents=True, exist_ok=True)
        profiler = torch_profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=profiler_schedule(
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

    try:
        # ==========================================================================
        # 0. Pre-warm Inference Containers via Autoscaler
        # ==========================================================================
        # vLLM cold start is slow (~30-60s). The InferenceEngine class is configured
        # with buffer_containers=1 at the decorator level, which keeps at least 1
        # container warm automatically. We spawn warmup calls to ensure containers
        # are actually ready.

        # ==========================================================================
        # 1. Model Loading (Timed)
        # ==========================================================================
        logger.info(f"🚀 Starting GRPO Training: {cfg.run_name}")

        model_load_start = time.time()

        tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_id)
        tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",  # PyTorch native scaled dot product attention
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

        # Enable gradient checkpointing to reduce memory (trades compute for memory)
        policy_model.gradient_checkpointing_enable()  # type: ignore[attr-defined]
        logger.info("✅ Gradient checkpointing enabled")

        # NOTE: torch.compile() with reduce-overhead mode uses CUDA graphs which
        # conflict with LoRA's dynamic tensor operations. Disabling for now.
        # See: https://github.com/huggingface/peft/issues/1043

        optimizer = torch.optim.AdamW(policy_model.parameters(), lr=cfg.learning_rate)
        loss_fn = GRPOLoss(policy_model, beta=0.04)  # type: ignore[arg-type]

        # ==========================================================================
        # Training State Checkpointing Helpers
        # ==========================================================================
        def save_training_state(step: int) -> None:
            """Save full training state for resume capability with atomic writes."""
            import tempfile

            run_path = MODELS_PATH / cfg.run_name
            run_path.mkdir(parents=True, exist_ok=True)
            state_path = run_path / f"training_state_v{step}.pt"

            state = {
                "step": step,
                "optimizer_state_dict": optimizer.state_dict(),
                "config": cfg.model_dump(),
                "seed": cfg.seed,  # Save seed for reproducibility
            }
            # Save wandb run ID if wandb is initialized (for resume)
            if wandb.run is not None:
                state["wandb_run_id"] = wandb.run.id

            # Atomic write: write to temp file first, then rename
            # This prevents corruption if save is interrupted
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=str(run_path), delete=False, suffix=".tmp"
            ) as tmp_file:
                tmp_path = tmp_file.name
                torch.save(state, tmp_path)

            # Atomic rename (should work on most filesystems)
            import os

            os.rename(tmp_path, str(state_path))
            volume.commit()
            logger.info(f"💾 Saved training state to {state_path}")

        def load_training_state(
            run_name: str, step: int | None = None, allow_fallback: bool = True
        ) -> tuple[int, str | None]:
            """
            Load training state and return the step to resume from and wandb run ID (if available).

            Args:
                run_name: Name of the run to load from
                step: Specific step to load (None = latest)
                allow_fallback: If True, try earlier checkpoints if latest is corrupted

            Returns:
                Tuple of (step, wandb_run_id)
            """
            import glob

            run_path = MODELS_PATH / run_name

            # Find latest checkpoint if step not specified
            if step is None:
                pattern = str(run_path / "training_state_v*.pt")
                state_files = glob.glob(pattern)
                if not state_files:
                    raise FileNotFoundError(f"No training states found in {run_path}")
                # Extract step numbers and find max
                steps = []
                for f in state_files:
                    try:
                        s = int(f.split("_v")[-1].replace(".pt", ""))
                        steps.append(s)
                    except ValueError:
                        pass
                if not steps:
                    raise FileNotFoundError(f"No valid training states found in {run_path}")
                step = max(steps)

            # At this point step is guaranteed to be an int (either passed in or from max(steps))
            assert step is not None, "step should be set by now"

            # Try loading checkpoint, with fallback to earlier ones if corrupted
            attempts = [step]
            if allow_fallback:
                # Get all available steps sorted descending
                pattern = str(run_path / "training_state_v*.pt")
                all_files = glob.glob(pattern)
                all_steps = sorted(
                    [
                        int(f.split("_v")[-1].replace(".pt", ""))
                        for f in all_files
                        if f.split("_v")[-1].replace(".pt", "").isdigit()
                    ],
                    reverse=True,
                )
                # Add earlier steps as fallbacks (up to 3 attempts)
                attempts.extend([s for s in all_steps if s < step][:2])

            last_error = None
            for attempt_step in attempts:
                state_path = run_path / f"training_state_v{attempt_step}.pt"
                if not state_path.exists():
                    continue

                try:
                    # Load and validate checkpoint
                    state = torch.load(str(state_path), weights_only=False)

                    # Validate required keys exist
                    required_keys = ["step", "optimizer_state_dict", "config"]
                    missing_keys = [k for k in required_keys if k not in state]
                    if missing_keys:
                        raise ValueError(f"Checkpoint missing required keys: {missing_keys}")

                    # Verify step matches filename
                    if state["step"] != attempt_step:
                        logger.warning(
                            f"⚠️ Step mismatch in checkpoint: filename={attempt_step}, "
                            f"state={state['step']}"
                        )

                    # Load optimizer state
                    optimizer.load_state_dict(state["optimizer_state_dict"])

                    # Load the adapter for this step
                    adapter_path = run_path / f"adapter_v{attempt_step}"
                    if adapter_path.exists():
                        policy_model.load_adapter(str(adapter_path), adapter_name="default")
                        logger.info(f"📂 Loaded adapter from {adapter_path}")
                    else:
                        logger.warning(
                            f"⚠️ Adapter not found at {adapter_path}, continuing with current weights"
                        )

                    # Extract wandb run ID if available (for resume)
                    wandb_run_id = state.get("wandb_run_id")

                    # Restore random seed if saved
                    saved_seed = state.get("seed")
                    if saved_seed is not None:
                        import random

                        import numpy as np

                        random.seed(saved_seed)
                        np.random.seed(saved_seed)
                        torch.manual_seed(saved_seed)
                        logger.info(f"🌱 Restored random seed: {saved_seed}")

                    # Warn about config mismatches (non-critical)
                    saved_config = state.get("config", {})
                    current_config = cfg.model_dump()
                    config_diff = {
                        k: (saved_config.get(k), current_config.get(k))
                        for k in set(saved_config.keys()) | set(current_config.keys())
                        if saved_config.get(k) != current_config.get(k)
                        and k not in ["run_name"]  # run_name can differ when forking
                    }
                    if config_diff:
                        logger.warning(
                            f"⚠️ Config differences detected (non-critical): {list(config_diff.keys())}"
                        )

                    if attempt_step != step:
                        logger.warning(
                            f"⚠️ Loaded checkpoint from step {attempt_step} "
                            f"(requested {step} was corrupted/unavailable)"
                        )

                    logger.info(f"✅ Resumed from step {attempt_step} (run: {run_name})")
                    return attempt_step, wandb_run_id

                except Exception as e:
                    last_error = e
                    logger.warning(f"⚠️ Failed to load checkpoint at step {attempt_step}: {e}")
                    if attempt_step == step:
                        logger.warning("   Will try earlier checkpoints if available...")
                    continue

            # All attempts failed
            raise FileNotFoundError(
                f"Failed to load any valid checkpoint from {run_path}. Last error: {last_error}"
            )

        # Resume from checkpoint if specified OR if checkpoints exist for this run
        start_step = 0
        wandb_run_id: str | None = None
        volume.reload()  # Ensure we see latest files

        if cfg.resume_from_run:
            # Explicit resume from another run
            try:
                start_step, wandb_run_id = load_training_state(
                    cfg.resume_from_run, cfg.resume_from_step
                )
                if cfg.resume_from_run != cfg.run_name:
                    logger.info(f"📦 Forking from {cfg.resume_from_run} to new run {cfg.run_name}")
                    # When forking, don't reuse the old wandb run ID
                    wandb_run_id = None
            except FileNotFoundError as e:
                logger.error(f"❌ Resume failed: {e}")
                raise
        elif not cfg.disable_auto_resume:
            # Auto-resume: Check if this run has existing checkpoints (crash recovery)
            import glob

            run_path = MODELS_PATH / cfg.run_name
            pattern = str(run_path / "training_state_v*.pt")
            existing_states = glob.glob(pattern)

            if existing_states:
                # Found existing checkpoints - auto-resume from latest
                logger.warning(
                    f"⚠️ Found {len(existing_states)} existing checkpoints for {cfg.run_name}"
                )
                logger.warning(
                    "🔄 AUTO-RESUMING from crash (use --disable-auto-resume to start fresh)"
                )
                try:
                    start_step, wandb_run_id = load_training_state(
                        cfg.run_name, cfg.resume_from_step
                    )
                except FileNotFoundError as e:
                    logger.error(f"❌ Auto-resume failed: {e}, starting fresh")
                    start_step = 0
                    wandb_run_id = None

        # Initialize WandB (after training state loading to support resume)
        wandb_tags = []
        if cfg.experiment_tag:
            wandb_tags.append(cfg.experiment_tag)

        # Resume existing wandb run if auto-resuming from crash
        wandb_init_kwargs: dict[str, Any] = {
            "project": cfg.wandb_project,
            "name": cfg.run_name,
            "tags": wandb_tags if wandb_tags else None,
            "config": {
                **cfg.model_dump(),
                "simulated_years_per_step": sim_years_per_step,
                "total_simulated_years": cfg.total_simulated_years,
            },
        }

        # If we have a wandb run ID from auto-resume, reuse that run
        if wandb_run_id:
            wandb_init_kwargs["id"] = wandb_run_id
            wandb_init_kwargs["resume"] = "must"
            logger.info(f"🔄 Resuming existing WandB run: {wandb_run_id}")
        elif start_step > 0:
            # Auto-resuming but no run ID saved (backward compatibility)
            # Try to resume by name, but allow creating new if not found
            wandb_init_kwargs["resume"] = "allow"
            logger.info("🔄 Attempting to resume WandB run by name (fallback)")

        wandb.init(**wandb_init_kwargs)

        model_load_time = time.time() - model_load_start
        metrics["timing"]["model_load_s"] = model_load_time
        logger.info(f"✅ Model loaded in {model_load_time:.2f}s")

        # ==========================================================================
        # 1.5 League Training Initialization (if enabled)
        # ==========================================================================
        league_registry = None
        pfsp_matchmaker = None
        last_registry_reload_step = -1  # Track last reload to throttle

        if cfg.league_training:
            from pathlib import Path

            from src.league import LeagueRegistry, PFSPConfig, PFSPMatchmaker

            logger.info("🏆 League training enabled - initializing registry and matchmaker")

            # Initialize or load league registry (default: per-run file)
            if cfg.league_registry_path:
                registry_path = Path(cfg.league_registry_path)
            else:
                # Per-run league file to avoid key collisions across runs
                registry_path = Path(f"/data/league_{cfg.run_name}.json")

            logger.info(f"📂 League registry path: {registry_path}")
            league_registry = LeagueRegistry(registry_path, run_name=cfg.run_name)

            # Optionally inherit opponents from a previous run (for curriculum learning)
            if cfg.league_inherit_from:
                inherit_path = Path(f"/data/league_{cfg.league_inherit_from}.json")
                if inherit_path.exists():
                    logger.info(f"📚 Inheriting opponents from {cfg.league_inherit_from}")
                    parent_registry = LeagueRegistry(inherit_path, run_name=cfg.league_inherit_from)
                    inherited_count = 0
                    for agent in parent_registry.get_checkpoints():
                        # Copy checkpoint as opponent (keep original key for path lookup)
                        if agent.name not in [a.name for a in league_registry.get_all_agents()]:
                            # Only copy if we have required fields (path, step)
                            if agent.path and agent.step is not None:
                                league_registry.add_checkpoint(
                                    name=agent.name,
                                    path=agent.path,
                                    step=agent.step,
                                    parent=agent.parent,
                                    initial_elo=agent.elo,
                                )
                                inherited_count += 1
                    logger.info(
                        f"✅ Inherited {inherited_count} checkpoints from {cfg.league_inherit_from}"
                    )
                else:
                    logger.warning(f"⚠️ Inherit league not found: {inherit_path}")

            # Configure PFSP with weights from config
            pfsp_config = PFSPConfig(
                self_play_weight=cfg.pfsp_self_play_weight,
                peer_weight=cfg.pfsp_peer_weight,
                exploitable_weight=cfg.pfsp_exploitable_weight,
                baseline_weight=cfg.pfsp_baseline_weight,
            )
            pfsp_matchmaker = PFSPMatchmaker(league_registry, pfsp_config)

            logger.info(
                f"📊 League status: {league_registry.num_checkpoints} checkpoints, "
                f"best Elo: {league_registry.best_elo:.0f} ({league_registry.best_agent})"
            )

            # Log league config to WandB
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

        # ==========================================================================
        # 2. Training Loop (BUFFERED PIPELINE)
        # ==========================================================================
        # Buffering strategy: Keep `buffer_depth` batches of rollouts in flight
        # - This ensures we always have rollouts ready even when they take
        #   longer than training (common with longer horizons)
        # - Higher buffer_depth = more rollouts in flight = less GPU idle time
        #   but also more "stale" trajectories (trained on older adapter)
        #
        # Timeline visualization (buffer_depth=3):
        #   Rollout[0] ──────────────────────►
        #   Rollout[1] ──────────────────────────────────────►
        #   Rollout[2] ──────────────────────────────────────────────────►
        #                                      Train[0] ────► Train[1] ────►
        #                                                     ↑ No GPU idle!
        #
        # Adapter versioning:
        # - All pre-launched batches use base model (no adapter trained yet)
        # - After step N training, new batches use adapter_v{N+1}
        # ==========================================================================
        from collections import deque

        total_trajectories = 0
        all_rewards = []
        buffer_depth = cfg.buffer_depth

        logger.info(f"🚀 Starting BUFFERED pipelined training loop (buffer_depth={buffer_depth})")

        # Pre-launch `buffer_depth` batches of rollouts
        # All use base model initially (no adapter trained yet)
        logger.info(f"Step 0: Pre-launching {buffer_depth} batches of rollouts with base model")

        # Get Modal function/class handles from the deployed app at runtime
        # This ensures they're properly hydrated in the combined app context
        run_rollout_fn = modal.Function.from_name("diplomacy-grpo", "run_rollout")
        InferenceEngineCls = modal.Cls.from_name("diplomacy-grpo", "InferenceEngine")
        evaluate_league_fn = modal.Function.from_name("diplomacy-grpo", "evaluate_league")

        def spawn_rollouts_batch(hero_adapter_path: str | None) -> tuple[list, list]:
            """
            Spawn a batch of rollouts with appropriate opponent sampling.

            Args:
                hero_adapter_path: Path to the hero's LoRA adapter (e.g., "run/adapter_v5"),
                                  or None for base model.
            """
            import random

            nonlocal last_registry_reload_step

            handles = []
            step_match_results = []  # Track PFSP sampling for observability
            for _ in range(cfg.num_groups_per_step):
                if cfg.league_training and pfsp_matchmaker is not None:
                    # League training: hero uses latest adapter, opponents from registry
                    hero_power = random.choice(pfsp_matchmaker.POWERS)

                    # Sample opponents based on registry (cold start = all baselines)
                    if league_registry and league_registry.num_checkpoints > 0:
                        # Reload registry periodically to get latest Elo updates from evaluate_league
                        # Reload every 5 steps to balance freshness vs. performance
                        # (evaluate_league typically runs every 50 steps, so this ensures we see updates)
                        if step - last_registry_reload_step >= 5:  # type: ignore[possibly-unbound]
                            try:
                                # CRITICAL: Must reload volume first to see commits from evaluate_league
                                # Modal Volumes don't auto-sync - each container has a local view
                                old_best_elo = league_registry.best_elo
                                volume.reload()
                                league_registry.reload()
                                last_registry_reload_step = step
                                new_best_elo = league_registry.best_elo

                                # Log whether Elo changed (validates sync from evaluate_league)
                                if new_best_elo != old_best_elo:
                                    logger.info(
                                        f"🔄 Registry reloaded at step {step}: "
                                        f"best_elo {old_best_elo:.0f} → {new_best_elo:.0f} "
                                        f"(+{new_best_elo - old_best_elo:.0f})"
                                    )
                                else:
                                    logger.debug(
                                        f"🔄 Registry reloaded at step {step}: "
                                        f"best_elo unchanged at {new_best_elo:.0f}"
                                    )
                            except Exception as e:
                                logger.warning(
                                    f"⚠️ Failed to reload registry: {e}, using cached state"
                                )

                        # Estimate hero's Elo for peer matching
                        # Hero adapter may not be registered yet (we checkpoint periodically)
                        # Use registry lookup if available, otherwise estimate from best_elo
                        hero_agent_name = hero_adapter_path or "base_model"
                        hero_info = league_registry.get_agent(hero_agent_name)
                        if hero_info:
                            estimated_hero_elo = hero_info.elo
                        else:
                            # Checkpoint not registered yet - use best_elo as estimate
                            # Current policy is likely similar strength to best registered checkpoint
                            estimated_hero_elo = league_registry.best_elo

                        # Use public API for clean opponent sampling
                        match_result = pfsp_matchmaker.sample_opponents(
                            hero_agent=hero_agent_name,
                            hero_power=hero_power,
                            num_opponents=6,
                            hero_elo_override=estimated_hero_elo,
                            hero_adapter_path=hero_adapter_path,  # For self-play when unregistered
                        )

                        # Use matched power_adapters, but override hero with exact training adapter
                        # (matchmaker may return a different path for the hero agent)
                        power_adapters = match_result.power_adapters.copy()
                        power_adapters[hero_power] = hero_adapter_path
                    else:
                        # Cold start: use matchmaker's cold start helper
                        match_result = pfsp_matchmaker.get_cold_start_opponents(hero_power)
                        power_adapters = match_result.power_adapters.copy()
                        power_adapters[hero_power] = hero_adapter_path

                    handles.append(
                        run_rollout_fn.spawn(  # pyright: ignore[reportCallIssue]
                            cfg.model_dump(),
                            power_adapters=power_adapters,
                            hero_power=hero_power,
                        )
                    )
                    step_match_results.append(match_result)
                else:
                    # Legacy mode: same adapter for all powers
                    handles.append(
                        run_rollout_fn.spawn(cfg.model_dump(), lora_name=hero_adapter_path)
                    )
            return handles, step_match_results

        # Each entry is (handles_list, hero_adapter_path, match_results for PFSP tracking)
        rollout_buffer: deque[tuple[list, str | None, list]] = deque()

        # Determine initial adapter for buffer (use resumed adapter if applicable)
        initial_adapter: str | None = None
        if start_step > 0:
            # Resuming: use the adapter from the resumed step
            initial_adapter = f"{cfg.run_name}/adapter_v{start_step}"
            logger.info(f"📦 Resuming - buffer will use adapter: {initial_adapter}")

        for _ in range(buffer_depth):
            handles, match_results = spawn_rollouts_batch(hero_adapter_path=initial_adapter)
            rollout_buffer.append((handles, initial_adapter, match_results))

        total_in_flight = buffer_depth * cfg.num_groups_per_step
        logger.info(
            f"📦 Buffer initialized: {buffer_depth} batches ({total_in_flight} rollouts) in flight"
        )

        for step in range(start_step, cfg.total_steps):
            step_start = time.time()
            step_metrics: dict[str, Any] = {"step": step}
            step_profile: dict[str, Any] | None = {"step": step} if profile_enabled else None

            # A. Wait for OLDEST rollouts (front of buffer)
            current_handles, current_lora_name, current_match_results = rollout_buffer.popleft()
            rollout_start = time.time()
            with stopwatch(f"Benchmark_Rollout_{step}"):
                raw_trajectories = []
                # Aggregate extraction stats across all rollouts in this batch
                step_extraction_stats: dict[str, int | float] = {
                    "orders_expected": 0,
                    "orders_extracted": 0,
                    "empty_responses": 0,
                    "partial_responses": 0,
                    "extraction_rate": 1.0,
                }
                # Aggregate timing stats from rollouts
                max_volume_reload_s = 0.0
                max_rollout_total_s = 0.0
                # Track prefix cache stats (query from engines after step)
                step_cache_stats_available = False

                failed_rollouts = 0
                for handle in current_handles:
                    try:
                        result = handle.get()  # Block until this rollout completes
                    except Exception as e:
                        # Log the failure but continue with other rollouts
                        failed_rollouts += 1
                        logger.warning(
                            f"⚠️ Rollout failed (will continue with others): {type(e).__name__}: {e}"
                        )
                        continue

                    # Unpack new return format: {"trajectories": [...], "extraction_stats": {...}, "timing": {...}}
                    raw_trajectories.extend(result["trajectories"])
                    # Aggregate extraction stats
                    stats = result["extraction_stats"]
                    step_extraction_stats["orders_expected"] += stats["orders_expected"]  # type: ignore[operator]
                    step_extraction_stats["orders_extracted"] += stats["orders_extracted"]  # type: ignore[operator]
                    step_extraction_stats["empty_responses"] += stats["empty_responses"]  # type: ignore[operator]
                    step_extraction_stats["partial_responses"] += stats["partial_responses"]  # type: ignore[operator]
                    # Track max timing stats (slowest rollout determines wait time)
                    timing = result.get("timing", {})
                    max_volume_reload_s = max(
                        max_volume_reload_s, timing.get("volume_reload_s", 0.0)
                    )
                    max_rollout_total_s = max(max_rollout_total_s, timing.get("total_s", 0.0))

                if failed_rollouts > 0:
                    logger.warning(
                        f"⚠️ Step {step}: {failed_rollouts}/{len(current_handles)} rollouts failed"
                    )

                # Compute step-level extraction rate
                orders_expected = step_extraction_stats["orders_expected"]
                orders_extracted = step_extraction_stats["orders_extracted"]
                if isinstance(orders_expected, int) and orders_expected > 0:
                    step_extraction_stats["extraction_rate"] = float(orders_extracted) / float(
                        orders_expected
                    )

            rollout_time = time.time() - rollout_start
            step_metrics["rollout_time_s"] = rollout_time
            step_metrics["raw_trajectories"] = len(raw_trajectories)
            step_metrics["rollout_lora"] = current_lora_name or "base_model"
            step_metrics["buffer_depth_actual"] = (
                len(rollout_buffer) + 1
            )  # +1 for the one we just popped
            step_metrics["extraction_stats"] = step_extraction_stats
            step_metrics["failed_rollouts"] = failed_rollouts
            # Track slowest rollout timing (identifies bottlenecks)
            step_metrics["max_volume_reload_s"] = max_volume_reload_s
            step_metrics["max_rollout_total_s"] = max_rollout_total_s
            if step_profile is not None:
                step_profile["rollout_time_ms"] = rollout_time * 1000

            # Query prefix cache stats from inference engine
            try:
                cache_stats_result = InferenceEngineCls(  # pyright: ignore[reportCallIssue]
                    model_id=cfg.base_model_id
                ).get_cache_stats.remote()
                step_cache_stats = cache_stats_result.get("cumulative", {})
                step_cache_stats_available = True
                # Debug: Log what we got
                vllm_stats = cache_stats_result.get("vllm_stats")
                logger.info(
                    f"📊 Cache stats for step {step}: "
                    f"batches={step_cache_stats.get('batches_processed', 0)}, "
                    f"prompt_tokens={step_cache_stats.get('total_prompt_tokens', 0)}, "
                    f"vllm_stats={'available' if vllm_stats else 'None'}"
                )
            except Exception as e:
                logger.warning(f"Could not get cache stats: {e}")
                step_cache_stats = {}
                step_cache_stats_available = False

            # B. Launch NEW batch to maintain buffer (if not near the end)
            # This batch runs during training, keeping the pipeline full
            steps_remaining = cfg.total_steps - step - 1
            if steps_remaining >= buffer_depth:
                # Determine which adapter to use for the new batch
                # After step 0 training completes, we'll have adapter_v1
                new_hero_agent: str | None = None

                if step >= 1:
                    adapter_rel_path = f"{cfg.run_name}/adapter_v{step}"
                    adapter_full_path = MODELS_PATH / cfg.run_name / f"adapter_v{step}"
                    policy_model.save_pretrained(str(adapter_full_path))
                    volume.commit()
                    logger.info(f"Saved adapter to {adapter_full_path}")

                    # League training: Add checkpoint to registry if criteria met
                    if cfg.league_training and league_registry is not None:
                        from src.league import should_add_to_league

                        # Use full path as key to ensure uniqueness across runs
                        # Format: "{run_name}/adapter_v{step}" e.g., "grpo-20251206/adapter_v50"
                        checkpoint_key = adapter_rel_path  # Already "{run_name}/adapter_v{step}"
                        parent_key = (
                            f"{cfg.run_name}/adapter_v{step - 1}" if step > 1 else "base_model"
                        )

                        if should_add_to_league(step, league_registry):
                            league_registry.add_checkpoint(
                                name=checkpoint_key,
                                path=adapter_rel_path,
                                step=step,
                                parent=parent_key,
                            )
                            # CRITICAL: Commit registry to volume so evaluate_league can see it
                            volume.commit()
                            logger.info(f"🏆 Added checkpoint {checkpoint_key} to league")

                            # Spawn async Elo evaluation if enabled
                            if (
                                cfg.elo_eval_every_n_steps > 0
                                and step % cfg.elo_eval_every_n_steps == 0
                            ):
                                logger.info(f"🎯 Spawning Elo evaluation for {checkpoint_key}")
                                # Get league registry path
                                registry_path_str = str(
                                    f"/data/league_{cfg.run_name}.json"
                                    if not cfg.league_registry_path
                                    else cfg.league_registry_path
                                )
                                # Spawn async - doesn't block training
                                # Pass all prompt/inference settings to ensure eval matches training
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

                    # Always use the adapter path directly for rollouts
                    # (checkpoint registry is for Elo tracking, not adapter loading)
                    new_hero_agent = adapter_rel_path

                target_step = step + buffer_depth
                logger.info(
                    f"🔀 Launching rollouts for step {target_step} "
                    f"(using {'base model' if not new_hero_agent else new_hero_agent})"
                )

                new_handles, new_match_results = spawn_rollouts_batch(
                    hero_adapter_path=new_hero_agent
                )
                rollout_buffer.append((new_handles, new_hero_agent, new_match_results))

            if not raw_trajectories:
                logger.warning(f"Step {step}: No trajectories, skipping")
                metrics["step_metrics"].append(step_metrics)
                continue

            total_trajectories += len(raw_trajectories)
            all_rewards.extend([t["reward"] for t in raw_trajectories])

            # C. Process trajectories
            process_start = time.time()
            profile_target = step_profile if step_profile is not None else {}
            with profile_section(profile_target, "tokenize"):
                batch_data, traj_stats = process_trajectories(raw_trajectories, tokenizer)
            process_time = time.time() - process_start
            step_metrics["process_time_s"] = process_time
            step_metrics["processed_trajectories"] = len(batch_data)

            if not batch_data:
                logger.warning(f"Step {step}: No valid batches")
                metrics["step_metrics"].append(step_metrics)
                continue

            # D. Training
            training_start = time.time()
            optimizer.zero_grad()

            accum_loss = 0.0
            accum_kl = 0.0
            num_chunks = 0

            for i in range(0, len(batch_data), cfg.chunk_size):
                chunk = batch_data[i : i + cfg.chunk_size]
                if not chunk:
                    break

                section_profile = step_profile if step_profile is not None else {}
                with profile_section(section_profile, "loss_forward"):
                    loss_output = loss_fn.compute_loss(chunk)
                scaled_loss = loss_output.loss / max(1, len(batch_data) // cfg.chunk_size)
                with profile_section(section_profile, "backward"):
                    scaled_loss.backward()

                accum_loss += loss_output.loss.item()
                accum_kl += loss_output.kl
                num_chunks += 1

            section_profile = step_profile if step_profile is not None else {}
            with profile_section(section_profile, "optimizer_step"):
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    policy_model.parameters(), cfg.max_grad_norm
                ).item()
                optimizer.step()

            training_time = time.time() - training_start

            # Record step metrics
            avg_loss = accum_loss / max(1, num_chunks)
            avg_kl = accum_kl / max(1, num_chunks)

            step_metrics["training_time_s"] = training_time
            step_metrics["loss"] = avg_loss
            step_metrics["kl"] = avg_kl
            step_metrics["grad_norm"] = grad_norm
            step_metrics["reward_mean"] = traj_stats.reward_mean
            step_metrics["reward_std"] = traj_stats.reward_std
            step_metrics["total_tokens"] = traj_stats.total_tokens
            if step_profile is not None:
                step_profile["training_time_ms"] = training_time * 1000
                step_profile["process_time_ms"] = process_time * 1000
                step_profile["trajectories"] = len(batch_data)
                step_profile["tokens"] = traj_stats.total_tokens

            step_total = time.time() - step_start
            step_metrics["total_time_s"] = step_total

            metrics["step_metrics"].append(step_metrics)

            extraction_rate_pct = float(step_extraction_stats["extraction_rate"]) * 100
            logger.info(
                f"Step {step}: loss={avg_loss:.4f} | kl={avg_kl:.4f} | "
                f"reward={traj_stats.reward_mean:.2f}±{traj_stats.reward_std:.2f} | "
                f"extraction={extraction_rate_pct:.1f}% | "
                f"trajectories={len(batch_data)} | time={step_total:.2f}s"
            )

            # Calculate pipeline efficiency: how much time was hidden by overlap
            # If rollout_time < training_time, we got good overlap
            pipeline_overlap = max(0, training_time - rollout_time) if step > 0 else 0
            step_metrics["pipeline_overlap_s"] = pipeline_overlap
            if step_profile is not None:
                step_profile["pipeline_overlap_ms"] = pipeline_overlap * 1000
                profile_snapshots.append(step_profile)

            # Calculate cumulative simulated years for power law X-axis
            cumulative_sim_years = (step + 1) * sim_years_per_step

            # Build wandb metrics dict
            wandb_metrics = {
                "benchmark/step": step,
                "benchmark/loss": avg_loss,
                "benchmark/kl": avg_kl,
                "benchmark/reward_mean": traj_stats.reward_mean,
                "benchmark/reward_std": traj_stats.reward_std,
                "benchmark/rollout_time_s": rollout_time,
                "benchmark/training_time_s": training_time,
                "benchmark/trajectories": len(batch_data),
                "benchmark/grad_norm": grad_norm,
                "benchmark/pipeline_overlap_s": pipeline_overlap,
                # Rollout timing breakdown (diagnose spikes)
                "rollout/max_volume_reload_s": max_volume_reload_s,
                "rollout/max_total_s": max_rollout_total_s,
                "rollout/failed_count": failed_rollouts,
                # Order extraction metrics (monitor prompt structure regressions)
                "extraction/rate": step_extraction_stats["extraction_rate"],
                "extraction/orders_expected": step_extraction_stats["orders_expected"],
                "extraction/orders_extracted": step_extraction_stats["orders_extracted"],
                "extraction/empty_responses": step_extraction_stats["empty_responses"],
                "extraction/partial_responses": step_extraction_stats["partial_responses"],
                # Power Law metrics (for X-axis comparison across runs)
                "power_law/cumulative_simulated_years": cumulative_sim_years,
                "power_law/simulated_years_per_step": sim_years_per_step,
                "power_law/reward_at_compute": traj_stats.reward_mean,
            }

            # Add prefix cache metrics (always log what we can track)
            if step_cache_stats_available and step_cache_stats:
                total_queries = step_cache_stats.get("total_queries", 0)
                total_hits = step_cache_stats.get("total_hits", 0)
                total_prompt_tokens = step_cache_stats.get("total_prompt_tokens", 0)
                batches = step_cache_stats.get("batches_processed", 0)
                batch_size_total = step_cache_stats.get("batch_size_total", 0)
                real_stats = step_cache_stats.get("real_stats_available", False)

                # Calculate hit rate from available data
                cache_hit_rate = total_hits / total_queries if total_queries > 0 else 0.0

                # Always log what we can measure
                cache_metrics = {
                    "cache/prompt_tokens": total_prompt_tokens,
                    "cache/batches": batches,
                    "cache/total_requests": batch_size_total,
                }

                # Only log hit rate if we have meaningful query data
                if total_queries > 0:
                    cache_metrics.update(
                        {
                            "cache/hit_rate": cache_hit_rate,
                            "cache/total_queries": total_queries,
                            "cache/total_hits": total_hits,
                        }
                    )

                # Flag if we got real vLLM stats vs estimates
                cache_metrics["cache/real_stats"] = 1 if real_stats else 0

                wandb_metrics.update(cache_metrics)

                logger.info(
                    f"📈 Logging cache metrics to WandB: "
                    f"batches={batches}, tokens={total_prompt_tokens}, "
                    f"queries={total_queries}, hits={total_hits}"
                )
            else:
                logger.warning(
                    f"⚠️ No cache stats to log: available={step_cache_stats_available}, "
                    f"stats={step_cache_stats}"
                )

            # Add league training metrics if enabled
            if cfg.league_training and league_registry is not None:
                wandb_metrics.update(
                    {
                        "league/num_checkpoints": league_registry.num_checkpoints,
                        "league/best_elo": league_registry.best_elo,
                        "league/latest_step": league_registry.latest_step,
                    }
                )

                # Add PFSP distribution metrics (validates matchmaking is working correctly)
                if current_match_results and pfsp_matchmaker is not None:
                    pfsp_stats = pfsp_matchmaker.get_sampling_stats(current_match_results)
                    category_rates = pfsp_stats.get("category_rates", {})
                    hero_powers = pfsp_stats.get("hero_power_distribution", {})

                    # Log category sampling rates
                    for category, rate in category_rates.items():
                        wandb_metrics[f"pfsp/{category}_rate"] = rate

                    # Log hero power distribution (should be roughly uniform)
                    total_games = sum(hero_powers.values()) or 1
                    for power, count in hero_powers.items():
                        wandb_metrics[f"pfsp/hero_{power.lower()}"] = count / total_games

            wandb.log(wandb_metrics)
            if profiler is not None:
                profiler.step()

            # Periodic checkpoint save for resume capability
            if cfg.save_state_every_n_steps > 0 and (step + 1) % cfg.save_state_every_n_steps == 0:
                save_training_state(step + 1)

        # Save final adapter
        final_adapter_path = MODELS_PATH / cfg.run_name / f"adapter_v{cfg.total_steps}"
        policy_model.save_pretrained(str(final_adapter_path))
        volume.commit()
        logger.info(f"Saved final adapter to {final_adapter_path}")

        # ==========================================================================
        # 3. Final Metrics
        # ==========================================================================
        total_time = time.time() - benchmark_start
        metrics["timing"]["total_s"] = total_time

        # Compute summary
        total_simulated_years = (
            cfg.total_steps
            * cfg.num_groups_per_step
            * cfg.samples_per_group
            * cfg.rollout_horizon_years
        )

        # Calculate total pipeline savings
        total_pipeline_overlap = sum(
            m.get("pipeline_overlap_s", 0) for m in metrics["step_metrics"]
        )

        metrics["summary"] = {
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
            "profile_snapshots": profile_snapshots if profile_enabled else None,
            "trace_dir": str(trace_subdir) if trace_subdir else None,
        }

        # Get final step metrics
        if metrics["step_metrics"]:
            final = metrics["step_metrics"][-1]
            metrics["summary"]["final_loss"] = final.get("loss")
            metrics["summary"]["final_kl"] = final.get("kl")
            metrics["summary"]["final_reward_mean"] = final.get("reward_mean")

        logger.info(f"\n{'=' * 60}")
        logger.info("🏁 BENCHMARK COMPLETE")
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(f"Trajectories: {total_trajectories}")
        logger.info(f"Throughput: {metrics['summary']['trajectories_per_second']:.2f} traj/s")
        logger.info(f"Pipeline overlap (time saved): {total_pipeline_overlap:.2f}s")
        logger.info(f"{'=' * 60}")

        wandb.log({"benchmark/complete": True, **metrics["summary"]})

        return metrics["summary"]

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        wandb.log({"benchmark/error": str(e)})
        raise

    finally:
        # Note: Autoscaler is configured via buffer_containers=1 at the class decorator level.
        # Containers will scale down automatically based on scaledown_window after inactivity.
        logger.info("🔄 Training complete. Containers will scale down after inactivity.")

        gpu_logger.stop()
        if profiler is not None:
            profiler.__exit__(None, None, None)
        asyncio.run(axiom.flush())
        wandb.finish()
