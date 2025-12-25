# src/apps/sweep/app.py
"""
Modal orchestrator for sweep experiments.

This lightweight CPU function orchestrates sequential training runs,
allowing users to close their laptops while sweeps run in the cloud.

Key features:
- Fire-and-forget: Local script spawns and exits immediately
- Auto-resume: Saves progress after each run; respawns self if timeout nears
- Progress tracking: sweep_state.json on volume tracks completed runs
- Inference warmup: Warms up inference engine before first run
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

import modal

from src.apps.common.images import cpu_image
from src.apps.common.volumes import VOLUME_PATH, volume
from src.utils.observability import axiom, logger
from src.utils.sweep_config import SweepConfig, SweepState

# Default model ID for inference warmup
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

app = modal.App("diplomacy-grpo-sweep")

# Leave 1 hour buffer before Modal's 24hr timeout for safe self-respawn
SWEEP_TIMEOUT_HOURS = 23
SWEEP_TIMEOUT_SECONDS = SWEEP_TIMEOUT_HOURS * 60 * 60

# Path for sweep state files
SWEEPS_PATH = VOLUME_PATH / "sweeps"


def warmup_inference_engine(model_id: str = DEFAULT_MODEL_ID) -> float:
    """
    Warm up the InferenceEngine before training runs.

    Args:
        model_id: HuggingFace model ID to warm up

    Returns:
        Warmup duration in seconds
    """
    logger.info(f"Warming up InferenceEngine with model: {model_id}")
    start = time.time()

    InferenceEngine = modal.Cls.from_name("diplomacy-grpo", "InferenceEngine")
    engine = InferenceEngine(model_id=model_id)

    # Make a minimal call to trigger container startup
    _ = engine.generate.remote(
        prompts=["<orders>"],
        valid_moves=[{"A PAR": ["A PAR - BUR"]}],
    )

    duration = time.time() - start
    logger.info(f"InferenceEngine ready! ({duration:.2f}s)")
    return duration


@app.function(
    image=cpu_image,
    volumes={str(VOLUME_PATH): volume},
    timeout=24 * 60 * 60,  # 24 hours (Modal max)
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("axiom-secrets"),
    ],
    retries=0,
)
def orchestrate_sweep(
    sweep_config_dict: dict[str, Any],
    run_ids: list[str] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Orchestrate a sweep, running training jobs sequentially.

    Auto-resumes if timeout approaches (respawns self with remaining runs).

    Args:
        sweep_config_dict: Serialized SweepConfig
        run_ids: Specific runs to execute (defaults to all runs)
        dry_run: If True, validate and return without running

    Returns:
        Dict with status, completed runs, and results
    """
    start_time = time.time()
    sweep_config = SweepConfig(**sweep_config_dict)
    sweep_name = sweep_config.metadata.name

    logger.info(f"Starting sweep orchestration: {sweep_name}")
    axiom.log(
        {
            "event": "sweep_started",
            "sweep_name": sweep_name,
            "hypothesis": sweep_config.metadata.hypothesis,
            "run_ids": run_ids or list(sweep_config.runs.keys()),
        }
    )

    # Load or create state
    state = SweepState.load_or_create(sweep_config)

    # Determine which runs to execute
    requested_runs = run_ids or sweep_config.get_run_ids()
    pending_runs = state.get_pending_runs(requested_runs)

    if not pending_runs:
        logger.info(f"All requested runs already completed: {requested_runs}")
        return {
            "status": "already_completed",
            "completed_runs": state.completed_runs,
            "results": state.run_results,
        }

    logger.info(f"Pending runs: {pending_runs}")
    logger.info(f"Already completed: {state.completed_runs}")

    if dry_run:
        # Just validate and show what would run
        for run_id in pending_runs:
            cfg = sweep_config.build_experiment_config(run_id, state.timestamp)
            logger.info(f"[DRY RUN] Would run {run_id}: {cfg.run_name}")
        return {
            "status": "dry_run",
            "pending_runs": pending_runs,
            "completed_runs": state.completed_runs,
        }

    # Warm up inference engine before training runs
    # Get model_id from the first pending run's config
    first_run_cfg = sweep_config.build_experiment_config(pending_runs[0], state.timestamp)
    model_id = first_run_cfg.base_model_id
    warmup_duration = warmup_inference_engine(model_id)
    axiom.log(
        {
            "event": "sweep_warmup_completed",
            "sweep_name": sweep_name,
            "model_id": model_id,
            "warmup_duration_s": warmup_duration,
        }
    )

    # Get reference to training function
    train_grpo = modal.Function.from_name("diplomacy-grpo", "train_grpo")

    for run_id in pending_runs:
        # Check timeout - respawn if approaching limit
        elapsed = time.time() - start_time
        if elapsed > SWEEP_TIMEOUT_SECONDS:
            logger.warning(
                f"Approaching timeout ({elapsed / 3600:.1f}h), respawning for remaining runs"
            )
            state.save()

            # Respawn with remaining runs
            remaining = [r for r in pending_runs if r not in state.completed_runs]
            orchestrate_sweep.spawn(sweep_config_dict, remaining)

            axiom.log(
                {
                    "event": "sweep_respawned",
                    "sweep_name": sweep_name,
                    "completed_runs": state.completed_runs,
                    "remaining_runs": remaining,
                }
            )

            return {
                "status": "respawned",
                "completed_runs": state.completed_runs,
                "remaining_runs": remaining,
            }

        # Build config for this run
        cfg = sweep_config.build_experiment_config(run_id, state.timestamp)
        run_info = sweep_config.runs[run_id]

        logger.info("=" * 60)
        logger.info(f"Starting run {run_id}: {run_info.name}")
        logger.info(f"  Description: {run_info.description}")
        logger.info(f"  Run name: {cfg.run_name}")
        logger.info("=" * 60)

        # Mark run as started
        state.mark_run_started(run_id, cfg.run_name)
        state.save()

        axiom.log(
            {
                "event": "sweep_run_started",
                "sweep_name": sweep_name,
                "run_id": run_id,
                "run_name": cfg.run_name,
                "config": cfg.model_dump(),
            }
        )

        # Execute training run (blocking)
        run_start = time.time()
        try:
            result = train_grpo.remote(config_dict=cfg.model_dump())
            run_duration = time.time() - run_start

            logger.info(f"Run {run_id} completed in {run_duration / 3600:.2f}h")
            logger.info(f"  Final reward: {result.get('final_reward_mean', 'N/A')}")

            # Mark run as completed
            state.mark_run_completed(
                run_id,
                {
                    "result": result,
                    "duration_s": run_duration,
                    "completed_at": datetime.now().isoformat(),
                },
            )
            state.save()

            axiom.log(
                {
                    "event": "sweep_run_completed",
                    "sweep_name": sweep_name,
                    "run_id": run_id,
                    "run_name": cfg.run_name,
                    "duration_hours": run_duration / 3600,
                    "final_reward": result.get("final_reward_mean"),
                }
            )

        except Exception as e:
            logger.error(f"Run {run_id} failed: {e}")
            axiom.log(
                {
                    "event": "sweep_run_failed",
                    "sweep_name": sweep_name,
                    "run_id": run_id,
                    "run_name": cfg.run_name,
                    "error": str(e),
                }
            )
            # Don't mark as completed - can retry on next invocation
            state.current_run = None
            state.save()
            raise

    # All runs completed
    total_duration = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"Sweep {sweep_name} completed!")
    logger.info(f"  Total duration: {total_duration / 3600:.2f}h")
    logger.info(f"  Completed runs: {state.completed_runs}")
    logger.info("=" * 60)

    axiom.log(
        {
            "event": "sweep_completed",
            "sweep_name": sweep_name,
            "completed_runs": state.completed_runs,
            "total_duration_hours": total_duration / 3600,
        }
    )

    return {
        "status": "completed",
        "completed_runs": state.completed_runs,
        "results": state.run_results,
        "total_duration_hours": total_duration / 3600,
    }


@app.function(
    image=cpu_image,
    volumes={str(VOLUME_PATH): volume},
    timeout=60,  # Quick status check
)
def get_sweep_status(sweep_name: str) -> dict[str, Any]:
    """
    Get the current status of a sweep.

    Args:
        sweep_name: Name of the sweep to check

    Returns:
        Dict with sweep status information
    """
    volume.reload()
    state = SweepState.load(sweep_name)

    if state is None:
        return {
            "status": "not_found",
            "sweep_name": sweep_name,
        }

    # Load config to get total runs
    sweep_config = SweepConfig(**state.sweep_config_dict)
    total_runs = len(sweep_config.runs)
    completed = len(state.completed_runs)

    return {
        "status": "in_progress" if state.current_run else "idle",
        "sweep_name": sweep_name,
        "started_at": state.started_at,
        "current_run": state.current_run,
        "completed_runs": state.completed_runs,
        "pending_runs": state.get_pending_runs(),
        "progress": f"{completed}/{total_runs}",
        "run_names": state.run_names,
    }


@app.function(
    image=cpu_image,
    volumes={str(VOLUME_PATH): volume},
    timeout=60,
)
def list_sweeps() -> list[dict[str, Any]]:
    """
    List all sweeps on the volume.

    Returns:
        List of sweep status dicts
    """
    volume.reload()
    sweeps_dir = SWEEPS_PATH

    if not sweeps_dir.exists():
        return []

    results = []
    for sweep_dir in sweeps_dir.iterdir():
        if sweep_dir.is_dir():
            state_file = sweep_dir / "state.json"
            if state_file.exists():
                state = SweepState.load(sweep_dir.name)
                if state:
                    sweep_config = SweepConfig(**state.sweep_config_dict)
                    total_runs = len(sweep_config.runs)
                    results.append(
                        {
                            "name": sweep_dir.name,
                            "started_at": state.started_at,
                            "progress": f"{len(state.completed_runs)}/{total_runs}",
                            "current_run": state.current_run,
                            "status": "in_progress" if state.current_run else "idle",
                        }
                    )

    return sorted(results, key=lambda x: x["started_at"], reverse=True)
