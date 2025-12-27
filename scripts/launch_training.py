#!/usr/bin/env python3
"""
Training script for the Diplomacy GRPO pipeline.

This script launches training runs on Modal with full configurability.
All arguments are auto-generated from ExperimentConfig.

Usage:
    # Quick smoke test (1 step, 2 rollouts)
    python scripts/benchmark_training.py --smoke

    # Standard training run
    python scripts/benchmark_training.py --total-steps 10 --num-groups-per-step 8

    # With profiling
    python scripts/benchmark_training.py --profiling-mode trainer --total-steps 3

    # Skip inference warmup (if engine is already running)
    python scripts/benchmark_training.py --no-warmup

    # Fire and forget (detach after launching)
    python scripts/benchmark_training.py --detach
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import modal

from src.utils.config import ExperimentConfig, add_config_args, config_from_args
from src.utils.git import ensure_clean_state_for_experiment


@dataclass
class TrainingResult:
    """Results from a training run."""

    # Config summary
    run_name: str
    total_steps: int
    num_groups_per_step: int
    samples_per_group: int
    rollout_horizon_years: int

    # Timing
    total_duration_s: float
    warmup_duration_s: float

    # Throughput
    total_trajectories: int
    total_simulated_years: int
    trajectories_per_second: float
    simulated_years_per_second: float

    # Training metrics (from final step)
    final_loss: float | None = None
    final_kl: float | None = None
    final_reward_mean: float | None = None

    # Profiling
    trace_dir: str | None = None
    profile_snapshots: list[dict[str, Any]] | None = None

    def print_report(self):
        """Print a formatted training report."""
        print("\n" + "=" * 60)
        print("🏁 TRAINING COMPLETE")
        print("=" * 60)

        print("\n📋 Configuration:")
        print(f"   Run Name:           {self.run_name}")
        print(f"   Steps:              {self.total_steps}")
        print(f"   Groups per step:    {self.num_groups_per_step}")
        print(f"   Samples per group:  {self.samples_per_group}")
        print(f"   Rollout horizon:    {self.rollout_horizon_years} years")
        expected = self.num_groups_per_step * self.samples_per_group * 7
        print(f"   Expected batch:     ~{expected} trajectories/step")

        print("\n⏱️  Timing:")
        print(f"   Total wall time:    {self.total_duration_s:.2f}s")
        print(f"   Warmup time:        {self.warmup_duration_s:.2f}s")
        if self.total_steps > 0:
            avg_step = (self.total_duration_s - self.warmup_duration_s) / self.total_steps
            print(f"   Avg step time:      {avg_step:.2f}s")

        print("\n📊 Throughput:")
        print(f"   Total trajectories: {self.total_trajectories}")
        print(f"   Simulated years:    {self.total_simulated_years}")
        print(f"   Trajectories/sec:   {self.trajectories_per_second:.2f}")
        print(f"   Sim years/sec:      {self.simulated_years_per_second:.2f}")

        if self.final_loss is not None:
            print("\n📈 Final Training Metrics:")
            print(f"   Loss:               {self.final_loss:.4f}")
            if self.final_kl is not None:
                print(f"   KL Divergence:      {self.final_kl:.4f}")
            if self.final_reward_mean is not None:
                print(f"   Reward Mean:        {self.final_reward_mean:.2f}")

        if self.trace_dir:
            print(f"\n🔬 Profiling trace: {self.trace_dir}")

        print("\n" + "=" * 60)


def warmup_inference_engine(model_id: str = "Qwen/Qwen2.5-7B-Instruct") -> float:
    """Warm up the InferenceEngine and return warmup duration."""
    print("🔥 Warming up InferenceEngine...")
    start = time.time()

    InferenceEngine = modal.Cls.from_name("diplomacy-grpo", "InferenceEngine")
    engine = InferenceEngine(model_id=model_id)

    # Make a minimal call to trigger container startup
    _ = engine.generate.remote(
        prompts=["<orders>"],
        valid_moves=[{"A PAR": ["A PAR - BUR"]}],
    )

    duration = time.time() - start
    print(f"✅ InferenceEngine ready! ({duration:.2f}s)")
    return duration


def run_training(cfg: ExperimentConfig, skip_warmup: bool = False) -> TrainingResult:
    """
    Run a training job on Modal.

    Args:
        cfg: ExperimentConfig with all training settings
        skip_warmup: Skip InferenceEngine warmup

    Returns:
        TrainingResult with timing and training metrics
    """
    print(f"\n🚀 Starting Training: {cfg.run_name}")
    print(
        f"   Config: {cfg.total_steps} steps × {cfg.num_groups_per_step} groups × {cfg.samples_per_group} samples"
    )

    total_start = time.time()

    # 1. Warmup
    warmup_duration = 0.0
    if not skip_warmup:
        warmup_duration = warmup_inference_engine(model_id=cfg.base_model_id)
    else:
        print("⏭️  Skipping warmup (--no-warmup)")

    # 2. Launch the training function
    train_grpo = modal.Function.from_name("diplomacy-grpo", "train_grpo")

    print("\n🏋️ Launching training job on Modal...")
    result = train_grpo.remote(config_dict=cfg.model_dump())

    total_duration = time.time() - total_start

    # 3. Build result from returned metrics
    return TrainingResult(
        run_name=result.get("run_name", cfg.run_name),
        total_steps=cfg.total_steps,
        num_groups_per_step=cfg.num_groups_per_step,
        samples_per_group=cfg.samples_per_group,
        rollout_horizon_years=cfg.rollout_horizon_years,
        total_duration_s=total_duration,
        warmup_duration_s=warmup_duration,
        total_trajectories=result.get("total_trajectories", 0),
        total_simulated_years=result.get("total_simulated_years", 0),
        trajectories_per_second=result.get("trajectories_per_second", 0),
        simulated_years_per_second=result.get("simulated_years_per_second", 0),
        final_loss=result.get("final_loss"),
        final_kl=result.get("final_kl"),
        final_reward_mean=result.get("final_reward_mean"),
        trace_dir=result.get("trace_dir"),
        profile_snapshots=result.get("profile_snapshots"),
    )


def main():
    # Create parser with auto-generated arguments from ExperimentConfig
    parser = argparse.ArgumentParser(
        description="Run Diplomacy GRPO training on Modal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Add ExperimentConfig args
    add_config_args(parser, ExperimentConfig)

    # Add script-specific args
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Quick smoke test (1 step, 2 groups, 2 samples)",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip InferenceEngine warmup",
    )
    parser.add_argument(
        "--detach",
        default=True,
        action="store_true",
        help="Launch and exit without waiting for results",
    )

    args = parser.parse_args()

    # Apply smoke test overrides
    if args.smoke:
        args.total_steps = 1
        args.num_groups_per_step = 2
        args.samples_per_group = 2
        args.rollout_horizon_years = 1
        print("🔬 Running smoke test configuration")

    # Auto-generate run_name if not set
    if not hasattr(args, "run_name") or args.run_name == "diplomacy-grpo-v1":
        args.run_name = f"grpo-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # Build config from args
    cfg = config_from_args(args, ExperimentConfig)  # type: ignore[type-var]
    assert isinstance(cfg, ExperimentConfig)

    # Ensure git state is clean for reproducibility
    print("\nChecking git state...")
    git_info = ensure_clean_state_for_experiment(f"run-{cfg.run_name}")
    print(f"  Branch: {git_info['branch']}")
    print(f"  Commit: {git_info['commit'][:8]}")

    # Update config with git info
    cfg.git_branch = git_info["branch"]
    cfg.git_commit = git_info["commit"]

    print(f"\n📦 Config:\n{cfg.model_dump_json(indent=2)}")

    if args.detach:
        # Fire and forget
        train_grpo = modal.Function.from_name("diplomacy-grpo", "train_grpo")

        # Warmup first if needed
        if not args.no_warmup:
            warmup_inference_engine(model_id=cfg.base_model_id)

        handle = train_grpo.spawn(config_dict=cfg.model_dump())
        print(f"\n✅ Training launched! Function ID: {handle.object_id}")
        print("\nTo check status later:")
        print(f"   modal function get {handle.object_id}")
        print("\nOr monitor in Modal dashboard:")
        print("   https://modal.com/apps")
        print("\nResults will be logged to WandB when complete.")
        return

    # Run and wait
    result = run_training(cfg, skip_warmup=args.no_warmup)
    result.print_report()


if __name__ == "__main__":
    main()
