#!/usr/bin/env python3
"""
Benchmark script for the GRPO training pipeline.

Launches a small training run on Modal to debug and benchmark the full pipeline:
- InferenceEngine warmup
- Rollout generation
- Trajectory processing
- GRPO loss computation
- Model updates

Usage:
    # Quick smoke test (1 step, 2 rollouts)
    python scripts/benchmark_training.py --smoke

    # Standard benchmark (3 steps, 4 rollouts)
    python scripts/benchmark_training.py

    # Custom config
    python scripts/benchmark_training.py --steps 5 --groups 8 --samples 4

    # Skip warmup (if engine is already running)
    python scripts/benchmark_training.py --no-warmup
"""

import argparse
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import modal


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    # Config
    total_steps: int
    num_groups_per_step: int
    samples_per_group: int
    rollout_horizon_years: int

    # Timing
    total_duration_s: float
    warmup_duration_s: float
    training_duration_s: float

    # Throughput
    total_trajectories: int
    total_simulated_years: int
    trajectories_per_second: float
    simulated_years_per_second: float

    run_name: str | None = None
    trace_dir: str | None = None

    # Training metrics (from final step)
    final_loss: float | None = None
    final_kl: float | None = None
    final_reward_mean: float | None = None
    profile_snapshots: list[dict[str, Any]] | None = None

    def to_profile_payload(self, mode: str | None) -> dict[str, Any]:
        return {
            "run_name": self.run_name or "unspecified",
            "profiling_mode": mode,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "summary": {
                "total_duration_s": self.total_duration_s,
                "training_duration_s": self.training_duration_s,
                "total_steps": self.total_steps,
                "num_groups_per_step": self.num_groups_per_step,
                "samples_per_group": self.samples_per_group,
                "trace_dir": self.trace_dir,
            },
            "snapshots": self.profile_snapshots or [],
        }

    def print_report(self):
        """Print a formatted benchmark report."""
        print("\n" + "=" * 60)
        print("🏁 TRAINING BENCHMARK COMPLETE")
        print("=" * 60)

        print("\n📋 Configuration:")
        print(f"   Steps:              {self.total_steps}")
        print(f"   Groups per step:    {self.num_groups_per_step}")
        print(f"   Samples per group:  {self.samples_per_group}")
        print(f"   Rollout horizon:    {self.rollout_horizon_years} years")
        expected = self.num_groups_per_step * self.samples_per_group * 7
        print(f"   Expected batch:     ~{expected} trajectories/step")

        print("\n⏱️  Timing:")
        print(f"   Total wall time:    {self.total_duration_s:.2f}s")
        print(f"   Warmup time:        {self.warmup_duration_s:.2f}s")
        print(f"   Training time:      {self.training_duration_s:.2f}s")
        print(f"   Avg step time:      {self.training_duration_s / max(1, self.total_steps):.2f}s")

        print("\n📊 Throughput:")
        print(f"   Total trajectories: {self.total_trajectories}")
        print(f"   Simulated years:    {self.total_simulated_years}")
        print(f"   Trajectories/sec:   {self.trajectories_per_second:.2f}")
        print(f"   Sim years/sec:      {self.simulated_years_per_second:.2f}")

        if self.final_loss is not None:
            print("\n📈 Final Training Metrics:")
            print(f"   Loss:               {self.final_loss:.4f}")
            print(f"   KL Divergence:      {self.final_kl:.4f}")
            print(f"   Reward Mean:        {self.final_reward_mean:.2f}")

        print("\n" + "=" * 60)


def _persist_profile_snapshot(profile_name: str, payload: dict[str, Any]) -> None:
    """Persist profiling payload to Modal volume via helper function."""
    persist_fn = modal.Function.from_name("diplomacy-grpo", "persist_profile_snapshot")
    persist_fn.remote(profile_name, payload)


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


def run_benchmark(
    total_steps: int = 3,
    num_groups_per_step: int = 4,
    samples_per_group: int = 4,
    rollout_horizon_years: int = 1,
    skip_warmup: bool = False,
    run_name: str | None = None,
    profiling_mode: str | None = None,
) -> BenchmarkResult:
    """
    Run a benchmark training job on Modal.

    Args:
        total_steps: Number of training steps
        num_groups_per_step: Number of rollout groups per step
        samples_per_group: Number of trajectory samples per group
        rollout_horizon_years: How many years to simulate per rollout
        skip_warmup: Skip InferenceEngine warmup
        run_name: Optional run name (defaults to timestamped name)

    Returns:
        BenchmarkResult with timing and throughput metrics
    """
    if run_name is None:
        run_name = f"benchmark-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    print(f"\n🚀 Starting Training Benchmark: {run_name}")
    print(
        f"   Config: {total_steps} steps × {num_groups_per_step} groups × {samples_per_group} samples"
    )

    total_start = time.time()
    profile_enabled = profiling_mode in {"rollout", "e2e"}
    profile_snapshots: list[dict[str, Any]] = []

    # 1. Warmup
    warmup_duration = 0.0
    if not skip_warmup:
        warmup_duration = warmup_inference_engine()
    else:
        print("⏭️  Skipping warmup (--no-warmup)")

    # 2. Build config for the training run
    # We need to create a modified config and pass it to the trainer
    # For now, we'll use the function's internal config but with our overrides
    from src.utils.config import ExperimentConfig

    cfg = ExperimentConfig(
        base_model_id="Qwen/Qwen2.5-7B-Instruct",
        run_name=run_name,
        total_steps=total_steps,
        num_groups_per_step=num_groups_per_step,
        samples_per_group=samples_per_group,
        rollout_horizon_years=rollout_horizon_years,
        rollout_visualize_chance=0.0,  # Disable visualization for benchmarks
        profiling_mode=profiling_mode,  # pyright: ignore[reportArgumentType]
        profile_run_name=run_name,
    )

    print(f"\n📦 Config: {cfg.model_dump()}")

    # 3. Run training
    print("\n🏋️ Launching training job on Modal...")
    training_start = time.time()

    # Run rollouts manually and time them
    # This gives us more control and visibility into the benchmark
    run_rollout = modal.Function.from_name("diplomacy-grpo", "run_rollout")

    all_trajectories = []
    step_times = []

    for step in range(total_steps):
        step_start = time.time()
        print(f"\n--- Step {step + 1}/{total_steps} ---")

        # Launch rollouts
        print(f"   Launching {num_groups_per_step} rollout workers...")
        rollout_start = time.time()

        step_trajectories = []
        for i, res in enumerate(run_rollout.map([cfg.model_dump()] * num_groups_per_step)):
            step_trajectories.extend(res)
            print(f"   [{i + 1}/{num_groups_per_step}] Rollout complete: {len(res)} trajectories")

        rollout_duration = time.time() - rollout_start
        all_trajectories.extend(step_trajectories)

        step_duration = time.time() - step_start
        step_times.append(step_duration)

        print(
            f"   Step {step + 1} complete: {len(step_trajectories)} trajectories in {step_duration:.2f}s"
        )
        print(f"   (Rollout: {rollout_duration:.2f}s)")

        if profile_enabled:
            profile_snapshots.append(
                {
                    "step": step,
                    "trajectories": len(step_trajectories),
                    "rollout_duration_ms": int(rollout_duration * 1000),
                    "step_duration_ms": int(step_duration * 1000),
                    "groups": num_groups_per_step,
                    "samples_per_group": samples_per_group,
                }
            )

    training_duration = time.time() - training_start
    total_duration = time.time() - total_start

    # 4. Calculate metrics
    total_trajectories = len(all_trajectories)
    total_simulated_years = (
        total_steps * num_groups_per_step * samples_per_group * rollout_horizon_years
    )

    result = BenchmarkResult(
        total_steps=total_steps,
        num_groups_per_step=num_groups_per_step,
        samples_per_group=samples_per_group,
        rollout_horizon_years=rollout_horizon_years,
        run_name=run_name,
        total_duration_s=total_duration,
        warmup_duration_s=warmup_duration,
        training_duration_s=training_duration,
        total_trajectories=total_trajectories,
        total_simulated_years=total_simulated_years,
        trajectories_per_second=total_trajectories / max(0.001, training_duration),
        simulated_years_per_second=total_simulated_years / max(0.001, training_duration),
        profile_snapshots=profile_snapshots if profile_enabled else None,
    )

    # 5. Validate data structure
    if all_trajectories:
        sample = all_trajectories[0]
        required_keys = ["prompt", "completion", "reward", "group_id"]
        missing = [k for k in required_keys if k not in sample]
        if missing:
            print(f"⚠️  Warning: Missing keys in trajectory: {missing}")
        else:
            print("✅ Data structure validation: PASS")

        # Show sample reward distribution
        rewards = [t["reward"] for t in all_trajectories]
        print("\n📊 Reward Distribution:")
        print(f"   Min:  {min(rewards):.2f}")
        print(f"   Max:  {max(rewards):.2f}")
        print(f"   Mean: {sum(rewards) / len(rewards):.2f}")

    return result


def run_full_training_benchmark(
    total_steps: int = 2,
    num_groups_per_step: int = 2,
    samples_per_group: int = 4,
    rollout_horizon_years: int = 1,
    learning_rate: float = 1e-5,
    skip_warmup: bool = False,
    profiling_mode: str | None = None,
    profile_run_name: str | None = None,
    compact_prompts: bool = False,
    rollout_visualize_chance: float = 0.01,
    model_id: str = "Qwen/Qwen2.5-7B-Instruct",
) -> BenchmarkResult:
    """
    Run the FULL training pipeline including model updates.

    This launches train_grpo_benchmark which includes:
    - Model loading
    - LoRA adapter creation
    - Rollouts
    - Loss computation
    - Gradient updates

    Note: Requires the train_grpo_benchmark function to be deployed.
    """
    print(f"\n🚀 Starting FULL Training Benchmark (with model updates, lr={learning_rate})")

    total_start = time.time()

    # Warmup
    warmup_duration = 0.0
    if not skip_warmup:
        warmup_duration = warmup_inference_engine(model_id=model_id)

    # Launch the benchmark training function
    train_grpo_benchmark = modal.Function.from_name("diplomacy-grpo", "train_grpo_benchmark")

    training_start = time.time()

    # This will return metrics from the training run
    result = train_grpo_benchmark.remote(
        total_steps=total_steps,
        num_groups_per_step=num_groups_per_step,
        samples_per_group=samples_per_group,
        rollout_horizon_years=rollout_horizon_years,
        learning_rate=learning_rate,
        profiling_mode=profiling_mode,
        profile_run_name=profile_run_name,
        compact_prompts=compact_prompts,
        rollout_visualize_chance=rollout_visualize_chance,
        model_id=model_id,
    )

    training_duration = time.time() - training_start
    total_duration = time.time() - total_start

    # Build result from returned metrics
    result_run_name = result.get("run_name", profile_run_name)
    return BenchmarkResult(
        total_steps=total_steps,
        num_groups_per_step=num_groups_per_step,
        samples_per_group=samples_per_group,
        rollout_horizon_years=rollout_horizon_years,
        run_name=result_run_name,
        trace_dir=result.get("trace_dir"),
        total_duration_s=total_duration,
        warmup_duration_s=warmup_duration,
        training_duration_s=training_duration,
        total_trajectories=result.get("total_trajectories", 0),
        total_simulated_years=result.get("total_simulated_years", 0),
        trajectories_per_second=result.get("trajectories_per_second", 0),
        simulated_years_per_second=result.get("simulated_years_per_second", 0),
        final_loss=result.get("final_loss"),
        final_kl=result.get("final_kl"),
        final_reward_mean=result.get("final_reward_mean"),
        profile_snapshots=result.get("profile_snapshots"),
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark the GRPO training pipeline on Modal")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Quick smoke test (1 step, 2 groups, 2 samples)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=3,
        help="Number of training steps (default: 3)",
    )
    parser.add_argument(
        "--groups",
        type=int,
        default=4,
        help="Number of rollout groups per step (default: 4)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=4,
        help="Samples per group (default: 4)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=2,
        help="Rollout horizon in years (default: 1)",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip InferenceEngine warmup",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full training with model updates (requires train_grpo_benchmark)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate (default: 1e-5). Try 1e-6 for more stable training.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Run name (default: auto-generated timestamp)",
    )
    parser.add_argument(
        "--profile",
        choices=["rollout", "trainer", "e2e"],
        default=None,
        help="Enable profiling and persist timing snapshots to /data/benchmarks",
    )
    parser.add_argument(
        "--profile-name",
        type=str,
        default=None,
        help="Optional name for saved profiling payload (defaults to autogenerated).",
    )
    parser.add_argument(
        "--compact-prompts",
        action="store_true",
        help="Use compact prompts (default: False)",
    )
    parser.add_argument(
        "--rollout-visualize-chance",
        type=float,
        default=0.01,
        help="Chance of visualizing a rollout (default: 0.01)",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model ID (default: Qwen/Qwen2.5-7B-Instruct)",
    )

    args = parser.parse_args()

    # Apply smoke test overrides
    if args.smoke:
        args.steps = 1
        args.groups = 2
        args.samples = 2
        args.horizon = 1
        print("🔬 Running smoke test configuration")

    profile_mode = args.profile
    profile_name = args.profile_name
    if profile_mode and not profile_name:
        profile_name = (
            args.name or f"profile-{profile_mode}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )

    if args.full:
        result = run_full_training_benchmark(
            total_steps=args.steps,
            num_groups_per_step=args.groups,
            samples_per_group=args.samples,
            rollout_horizon_years=args.horizon,
            learning_rate=args.lr,
            skip_warmup=args.no_warmup,
            profiling_mode=profile_mode,
            profile_run_name=profile_name,
            compact_prompts=args.compact_prompts,
            rollout_visualize_chance=args.rollout_visualize_chance,
            model_id=args.model_id,
        )
    else:
        result = run_benchmark(
            total_steps=args.steps,
            num_groups_per_step=args.groups,
            samples_per_group=args.samples,
            rollout_horizon_years=args.horizon,
            skip_warmup=args.no_warmup,
            run_name=args.name,
            profiling_mode=profile_mode,
        )

    if profile_mode:
        snapshot_name = profile_name or result.run_name or "profile-run"
        _persist_profile_snapshot(
            snapshot_name,
            result.to_profile_payload(profile_mode),
        )

    result.print_report()


if __name__ == "__main__":
    main()
