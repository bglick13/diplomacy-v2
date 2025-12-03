#!/usr/bin/env python3
"""
Profiling harness for Diplomacy GRPO rollouts and trainer steps.

This script wraps the existing benchmark helpers and forces profiling-friendly
configurations so we can capture PyTorch traces directly on Modal volumes.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from scripts.benchmark_training import (
    _persist_profile_snapshot,
    run_benchmark,
    run_full_training_benchmark,
)


def profile_trainer(args: argparse.Namespace):
    """Run train_grpo_benchmark with profiling enabled."""
    profile_name = args.profile_name or f"trainer-profile-{datetime.now():%Y%m%d-%H%M%S}"
    result = run_full_training_benchmark(
        total_steps=args.steps,
        num_groups_per_step=args.groups,
        samples_per_group=args.samples,
        rollout_horizon_years=args.horizon,
        learning_rate=args.lr,
        skip_warmup=args.no_warmup,
        profiling_mode="trainer",
        profile_run_name=profile_name,
        buffer_depth=args.buffer_depth,
        max_policy_lag_steps=args.policy_lag,
        compact_prompts=args.compact_prompts,
    )
    payload = result.to_profile_payload("trainer")
    _persist_profile_snapshot(profile_name, payload)

    trace_dir = payload["summary"].get("trace_dir")
    print("\n✅ Trainer profiling complete.")
    if result.run_name:
        print(f"   Run name: {result.run_name}")
    if trace_dir:
        print(f"   Trace directory: {trace_dir} (inside /traces on Modal)")
    result.print_report()


def profile_rollouts(args: argparse.Namespace):
    """Run rollout-only benchmark with profiling enabled."""
    profile_name = args.profile_name or f"rollout-profile-{datetime.now():%Y%m%d-%H%M%S}"
    result = run_benchmark(
        total_steps=args.steps,
        num_groups_per_step=args.groups,
        samples_per_group=args.samples,
        rollout_horizon_years=args.horizon,
        skip_warmup=args.no_warmup,
        run_name=args.name,
        profiling_mode="rollout",
        buffer_depth=args.buffer_depth,
        max_policy_lag_steps=args.policy_lag,
        compact_prompts=args.compact_prompts,
    )
    payload = result.to_profile_payload("rollout")
    _persist_profile_snapshot(profile_name, payload)

    print("\n✅ Rollout profiling complete.")
    if result.run_name:
        print(f"   Run name: {result.run_name}")
    print("   Snapshots saved to /data/benchmarks via persist_profile_snapshot.")
    result.print_report()


def main():
    parser = argparse.ArgumentParser(
        description="Profile Diplomacy GRPO rollouts or trainer steps on Modal."
    )
    parser.add_argument(
        "--target",
        choices=["trainer", "rollout"],
        default="trainer",
        help="Which pipeline stage to profile (default: trainer).",
    )
    parser.add_argument("--steps", type=int, default=2, help="Number of steps to run.")
    parser.add_argument(
        "--groups", type=int, default=2, help="Rollout groups per step (G)."
    )
    parser.add_argument(
        "--samples", type=int, default=2, help="Samples per group (N)."
    )
    parser.add_argument(
        "--horizon", type=int, default=1, help="Rollout horizon in years."
    )
    parser.add_argument(
        "--lr", type=float, default=1e-5, help="Learning rate for trainer profiling."
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip inference engine warmup if already running.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional run name override for rollout profiling.",
    )
    parser.add_argument(
        "--profile-name",
        type=str,
        default=None,
        help="Custom identifier for persisted profiling payloads.",
    )
    parser.add_argument(
        "--buffer-depth",
        type=int,
        default=2,
        help="Rollout queue depth when profiling.",
    )
    parser.add_argument(
        "--policy-lag",
        type=int,
        default=1,
        help="Maximum allowed policy lag when profiling.",
    )
    parser.add_argument(
        "--compact-prompts",
        action="store_true",
        help="Use compact prompt template during profiling runs.",
    )

    args = parser.parse_args()

    if args.target == "trainer":
        profile_trainer(args)
    else:
        profile_rollouts(args)


if __name__ == "__main__":
    main()
