#!/usr/bin/env python3
"""
Single-game benchmark for measuring token counts and TPS.

This script runs a single Diplomacy game and measures:
- Total prompt tokens (all powers, all steps)
- Total completion tokens
- Tokens per power per step (average)
- Game duration in steps
- Wall clock time
- Tokens per second (input and output)

Usage:
    python scripts/benchmark_single_game.py
    python scripts/benchmark_single_game.py --horizon-years 5
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field

import modal

from src.utils.config import ExperimentConfig


@dataclass
class TokenMetrics:
    """Accumulated token metrics from a game."""

    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    num_inference_calls: int = 0
    prompt_tokens_per_call: list[int] = field(default_factory=list)
    completion_tokens_per_call: list[int] = field(default_factory=list)

    def add_trajectory(self, traj: dict) -> None:
        """Add token counts from a trajectory."""
        prompt_tids = traj.get("prompt_token_ids", [])
        completion_tids = traj.get("completion_token_ids", [])

        prompt_count = len(prompt_tids) if prompt_tids else 0
        completion_count = len(completion_tids) if completion_tids else 0

        self.total_prompt_tokens += prompt_count
        self.total_completion_tokens += completion_count
        self.num_inference_calls += 1
        self.prompt_tokens_per_call.append(prompt_count)
        self.completion_tokens_per_call.append(completion_count)

    def report(self, wall_time_s: float) -> dict:
        """Generate a report of token metrics."""
        total_tokens = self.total_prompt_tokens + self.total_completion_tokens

        avg_prompt = (
            sum(self.prompt_tokens_per_call) / len(self.prompt_tokens_per_call)
            if self.prompt_tokens_per_call
            else 0
        )
        avg_completion = (
            sum(self.completion_tokens_per_call) / len(self.completion_tokens_per_call)
            if self.completion_tokens_per_call
            else 0
        )

        return {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": total_tokens,
            "num_inference_calls": self.num_inference_calls,
            "avg_prompt_tokens_per_call": round(avg_prompt, 1),
            "avg_completion_tokens_per_call": round(avg_completion, 1),
            "wall_time_s": round(wall_time_s, 2),
            "input_tps": round(self.total_prompt_tokens / wall_time_s, 1) if wall_time_s > 0 else 0,
            "output_tps": round(self.total_completion_tokens / wall_time_s, 1)
            if wall_time_s > 0
            else 0,
            "total_tps": round(total_tokens / wall_time_s, 1) if wall_time_s > 0 else 0,
        }


def run_single_game(cfg: ExperimentConfig) -> tuple[dict, float]:
    """Run a single game and return full result dict with wall time."""
    run_rollout = modal.Function.from_name("diplomacy-grpo", "run_rollout")

    start_time = time.time()
    result = run_rollout.remote(cfg.model_dump())
    wall_time = time.time() - start_time

    return result, wall_time


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark token counts and TPS for a single Diplomacy game",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--horizon-years",
        type=int,
        default=3,
        help="Number of years to simulate (default: 3)",
    )
    parser.add_argument(
        "--samples-per-group",
        type=int,
        default=6,
        help="Number of forks per group (default: 6)",
    )
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Skip inference engine warmup",
    )

    args = parser.parse_args()

    # Create config
    cfg = ExperimentConfig(
        rollout_horizon_years=args.horizon_years,
        samples_per_group=args.samples_per_group,
    )

    print("=" * 60)
    print("Single Game Token Benchmark")
    print("=" * 60)
    print(f"Horizon years: {cfg.rollout_horizon_years}")
    print(f"Samples per group: {cfg.samples_per_group}")
    print(f"Base model: {cfg.base_model_id}")

    # Warmup
    if not args.skip_warmup:
        print("\n🔥 Warming up InferenceEngine...")
        InferenceEngine = modal.Cls.from_name("diplomacy-grpo", "InferenceEngine")
        engine = InferenceEngine(model_id=cfg.base_model_id)
        _ = engine.generate.remote(
            prompts=["<orders>"],
            valid_moves=[{"A PAR": ["A PAR - BUR"]}],
        )
        print("✅ InferenceEngine ready!")

    # Run single game
    print("\n🎮 Running single game...")
    result, wall_time = run_single_game(cfg)

    # Handle both old (list) and new (dict) return formats
    if isinstance(result, dict):
        trajectories = result.get("trajectories", [])
        timing = result.get("timing", {})
    else:
        trajectories = result if isinstance(result, list) else []
        timing = {}

    if not trajectories:
        print("❌ No trajectories returned!")
        return

    print(f"✅ Game complete! Got {len(trajectories)} trajectories in {wall_time:.2f}s")

    # Show timing breakdown if available
    if timing:
        print("\n⏱️  Timing Breakdown (from rollout):")
        for key, value in timing.items():
            if isinstance(value, (int, float)) and value > 0:
                print(f"   {key}: {value:.2f}s")

    # Collect token metrics
    metrics = TokenMetrics()
    for traj in trajectories:
        if isinstance(traj, dict):
            metrics.add_trajectory(traj)

    # Generate report
    report = metrics.report(wall_time)

    print("\n" + "=" * 60)
    print("📊 TOKEN METRICS")
    print("=" * 60)
    print(f"Total Prompt Tokens:      {report['total_prompt_tokens']:,}")
    print(f"Total Completion Tokens:  {report['total_completion_tokens']:,}")
    print(f"Total Tokens:             {report['total_tokens']:,}")
    print(f"Inference Calls:          {report['num_inference_calls']}")
    print(f"Avg Prompt/Call:          {report['avg_prompt_tokens_per_call']}")
    print(f"Avg Completion/Call:      {report['avg_completion_tokens_per_call']}")

    print("\n" + "-" * 40)
    print("⏱️  THROUGHPUT")
    print("-" * 40)
    print(f"Wall Time:                {report['wall_time_s']}s")
    print(f"Input TPS (prefill):      {report['input_tps']:,}")
    print(f"Output TPS (decode):      {report['output_tps']:,}")
    print(f"Total TPS:                {report['total_tps']:,}")

    # Calculate per-step estimates
    if report["num_inference_calls"] > 0:
        # Estimate steps (each power gets 1 inference per step, 7 powers)
        estimated_steps = report["num_inference_calls"] / 7 / cfg.samples_per_group
        print(f"\nEstimated game steps:     {estimated_steps:.1f}")

    print("\n" + "=" * 60)

    # Also show trajectory sample for validation
    if trajectories:
        sample = trajectories[0]
        print("\n📝 Sample Trajectory Keys:")
        for key in sorted(sample.keys()):
            val = sample[key]
            if isinstance(val, list):
                print(f"   {key}: list[{len(val)}]")
            elif isinstance(val, str) and len(val) > 50:
                print(f"   {key}: str[{len(val)} chars]")
            else:
                print(f"   {key}: {type(val).__name__}")


if __name__ == "__main__":
    main()
