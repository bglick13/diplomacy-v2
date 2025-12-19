#!/usr/bin/env python3
"""
Rollout-only benchmark script for the Diplomacy GRPO pipeline.

This script benchmarks rollout performance without training updates.
Useful for measuring inference throughput and rollout worker scaling.

Usage:
    # Default benchmark (2 games)
    python scripts/benchmark_rollouts.py

    # More games
    python scripts/benchmark_rollouts.py --num-games 10

    # Custom config
    python scripts/benchmark_rollouts.py --samples-per-group 16 --rollout-horizon-years 4
"""

from __future__ import annotations

import argparse
import time

import modal

from src.utils.config import ExperimentConfig, add_config_args, config_from_args


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Diplomacy GRPO rollouts on Modal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Add subset of ExperimentConfig args relevant to rollouts
    add_config_args(
        parser,
        ExperimentConfig,
        exclude={
            "run_name",
            "total_steps",
            "num_groups_per_step",
            "learning_rate",
            "max_grad_norm",
            "chunk_size",
            "profiling_mode",
            "profile_run_name",
            "profiling_trace_steps",
            "experiment_tag",
            "wandb_project",
        },
    )

    # Rollout-specific args
    parser.add_argument(
        "--num-games",
        type=int,
        default=2,
        help="Number of concurrent rollouts to run (default: 2)",
    )

    args = parser.parse_args()
    num_games = args.num_games

    # Build config from args
    cfg = config_from_args(args, ExperimentConfig)  # type: ignore[type-var]
    assert isinstance(cfg, ExperimentConfig)

    print(f"🚀 Benchmarking: Launching {num_games} concurrent rollouts...")
    print(f"   Samples per group: {cfg.samples_per_group}")
    print(f"   Horizon years: {cfg.rollout_horizon_years}")

    # Warmup: Ensure InferenceEngine is ready
    print("\n🔥 Warming up InferenceEngine...")
    InferenceEngine = modal.Cls.from_name("diplomacy-grpo-inference-engine", "InferenceEngine")
    engine = InferenceEngine(model_id=cfg.base_model_id)
    _ = engine.generate.remote(
        prompts=["<orders>"],
        valid_moves=[{"A PAR": ["A PAR - BUR"]}],
    )
    print("✅ InferenceEngine ready!")

    # Run rollouts
    run_rollout = modal.Function.from_name("diplomacy-grpo-rollouts", "run_rollout")

    start_time = time.time()
    results = []

    try:
        # Iterate as they complete to show progress
        for i, res in enumerate(run_rollout.map([cfg.model_dump()] * num_games)):
            results.append(res)
            print(f"[{i + 1}/{num_games}] ✅ Game finished. Trajectories collected: {len(res)}")

    except Exception as e:
        print(f"❌ Critical Failure: {e}")
        return

    duration = time.time() - start_time

    # Calculate metrics
    total_trajectories = sum(len(r) for r in results)
    total_simulated_years = num_games * cfg.samples_per_group * cfg.rollout_horizon_years

    print("\n" + "=" * 50)
    print("📊 BENCHMARK COMPLETE")
    print("=" * 50)
    print(f"Total Wall Time:     {duration:.2f}s")
    print(f"Simulated Years:     {total_simulated_years}")
    print(f"Real Throughput:     {total_simulated_years / duration:.2f} Years/Sec")
    print(f"Total Trajectories:  {total_trajectories}")
    print("=" * 50)

    # Validation
    if results:
        sample = results[0][0]
        required_keys = ["prompt", "completion", "reward", "group_id"]
        missing = [k for k in required_keys if k not in sample]
        if missing:
            print(f"⚠️  Missing keys in trajectory: {missing}")
        else:
            print("✅ Data Structure Validation: PASS")


if __name__ == "__main__":
    main()
