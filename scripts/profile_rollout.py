#!/usr/bin/env python3
"""
Profiling script for Diplomacy GRPO training.

This script runs training with profiling enabled to capture PyTorch traces
for performance analysis. Traces are saved to Modal volumes for later viewing.

Usage:
    # Profile trainer (model forward/backward passes)
    python scripts/profile_rollout.py --profiling-mode trainer

    # Profile rollouts only
    python scripts/profile_rollout.py --profiling-mode rollout

    # Full end-to-end profiling
    python scripts/profile_rollout.py --profiling-mode e2e

    # Custom steps
    python scripts/profile_rollout.py --profiling-mode trainer --total-steps 5
"""

from __future__ import annotations

import argparse
from datetime import datetime

import modal

from src.utils.config import ExperimentConfig, add_config_args, config_from_args


def main():
    parser = argparse.ArgumentParser(
        description="Profile Diplomacy GRPO training on Modal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Add ExperimentConfig args
    add_config_args(parser, ExperimentConfig)

    # Override defaults for profiling (smaller runs)
    parser.set_defaults(
        total_steps=3,
        num_groups_per_step=2,
        samples_per_group=2,
        rollout_horizon_years=1,
    )

    # Add script-specific args
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip InferenceEngine warmup",
    )

    args = parser.parse_args()

    # Require profiling mode
    if not args.profiling_mode:
        parser.error("--profiling-mode is required (trainer, rollout, or e2e)")

    # Auto-generate profile name if not set
    if not args.profile_run_name:
        args.profile_run_name = f"profile-{args.profiling_mode}-{datetime.now():%Y%m%d-%H%M%S}"

    # Auto-generate run name if not set
    if args.run_name == "diplomacy-grpo-v1":
        args.run_name = args.profile_run_name

    # Build config from args
    cfg = config_from_args(args, ExperimentConfig)  # type: ignore[type-var]
    assert isinstance(cfg, ExperimentConfig)

    print(f"\n🔬 Profiling Mode: {cfg.profiling_mode}")
    print(f"   Run Name: {cfg.run_name}")
    print(f"   Steps: {cfg.total_steps}")
    print(f"   Groups: {cfg.num_groups_per_step}")
    print(f"   Samples: {cfg.samples_per_group}")

    # Optional warmup
    if not args.no_warmup:
        print("\n🔥 Warming up InferenceEngine...")
        InferenceEngine = modal.Cls.from_name("diplomacy-grpo", "InferenceEngine")
        engine = InferenceEngine(model_id=cfg.base_model_id)
        _ = engine.generate.remote(
            prompts=["<orders>"],
            valid_moves=[{"A PAR": ["A PAR - BUR"]}],
        )
        print("✅ InferenceEngine ready!")

    # Launch training with profiling
    print("\n🏋️ Launching profiled training on Modal...")
    train_grpo = modal.Function.from_name("diplomacy-grpo", "train_grpo")

    result = train_grpo.remote(config_dict=cfg.model_dump())

    print("\n✅ Profiling complete!")
    print(f"   Run name: {result.get('run_name')}")

    trace_dir = result.get("trace_dir")
    if trace_dir:
        print(f"   Trace directory: {trace_dir}")
        print("\n   To view traces:")
        print("   1. Download from Modal volume: /traces/trainer/")
        print("   2. Open in TensorBoard: tensorboard --logdir=<path>")

    # Print summary
    if result.get("total_trajectories"):
        print("\n📊 Summary:")
        print(f"   Trajectories: {result['total_trajectories']}")
        print(f"   Throughput: {result.get('trajectories_per_second', 0):.2f} traj/s")
        if result.get("final_reward_mean") is not None:
            print(f"   Final Reward: {result['final_reward_mean']:.2f}")


if __name__ == "__main__":
    main()
