#!/usr/bin/env python3
"""
Power Laws Experiment: Compute vs. Insight

This script runs the "Mini Power Laws" experiment to test whether
scaling (compute) or engineering (reward shaping) solves the problem
of models learning suicidal rushing behavior in Diplomacy.

The Hypothesis:
- Engineering approach: Add penalties for leaving home centers empty
- Scaling approach: Look further into the future (longer horizons)

Experiment Configurations:
- Run A (Baseline): horizon=2, samples=8  → 1x compute
- Run B (Deep Search): horizon=4, samples=8  → 2x compute
- Run C (Broad Search): horizon=2, samples=16 → 2x compute

After 100 steps, we compare reward slopes to determine:
1. If Run B wins → Longer horizons work, simple rewards are sufficient
2. If Run C wins → More samples work, noise filtering is the bottleneck
3. If neither wins → Reward engineering is necessary

Usage:
    # Launch sweep on Modal (can close laptop after)
    python scripts/launch_sweep.py

    # Run specific configuration
    python scripts/launch_sweep.py --run A
    python scripts/launch_sweep.py --run B
    python scripts/launch_sweep.py --run C

    # Run with fewer steps for testing
    python scripts/launch_sweep.py --steps 10 --run A

    # Dry run (show config without launching)
    python scripts/launch_sweep.py --dry-run

    # Run in parallel mode (3x GPUs, 3x cost, 3x faster)
    python scripts/launch_sweep.py --parallel

    # Fire and forget (don't wait for results)
    python scripts/launch_sweep.py --detach

    # Use smaller model for faster iteration
    python scripts/launch_sweep.py --fast
"""

import argparse
import json
from dataclasses import dataclass
from datetime import datetime

import modal


@dataclass
class SweepConfig:
    """Configuration for a single sweep run."""

    name: str
    tag: str  # WandB tag for grouping
    rollout_horizon_years: int
    samples_per_group: int
    compute_multiplier: float
    description: str

    def simulated_years_per_step(self, num_groups: int) -> int:
        """Calculate simulated years per training step."""
        return num_groups * self.samples_per_group * self.rollout_horizon_years

    def total_simulated_years(self, num_groups: int, total_steps: int) -> int:
        """Calculate total simulated years for the full run."""
        return self.simulated_years_per_step(num_groups) * total_steps


# Define the three experimental configurations
SWEEP_CONFIGS = {
    "A": SweepConfig(
        name="baseline",
        tag="power-laws-baseline",
        rollout_horizon_years=2,
        samples_per_group=8,
        compute_multiplier=1.0,
        description="Baseline: Fast & Loose (horizon=2, samples=8)",
    ),
    "B": SweepConfig(
        name="deep-search",
        tag="power-laws-deep",
        rollout_horizon_years=4,
        samples_per_group=8,
        compute_multiplier=2.0,
        description="Deep Search: Time Scaling (horizon=4, samples=8)",
    ),
    "C": SweepConfig(
        name="broad-search",
        tag="power-laws-broad",
        rollout_horizon_years=2,
        samples_per_group=16,
        compute_multiplier=2.0,
        description="Broad Search: Variance Scaling (horizon=2, samples=16)",
    ),
}


def print_comparison_table(comparison: list[dict], analysis: dict):
    """Print a comparison table of all sweep results."""
    print("\n")
    print("=" * 80)
    print("📊 POWER LAWS EXPERIMENT COMPARISON")
    print("=" * 80)
    print()

    # Header
    print(f"{'Config':<12} {'Compute':<8} {'Sim Years':<12} {'Reward Mean':<12} {'Time (s)':<10}")
    print("-" * 60)

    for r in comparison:
        reward_str = f"{r['final_reward_mean']:.2f}" if r.get("final_reward_mean") else "N/A"
        print(
            f"{r['name']:<12} "
            f"{r['compute_multiplier']:<8.1f}x "
            f"{r['simulated_years']:<12} "
            f"{reward_str:<12} "
            f"{r['duration_s']:<10.1f}"
        )

    print()
    print("=" * 80)
    print("📈 ANALYSIS")
    print("=" * 80)

    if analysis.get("winner"):
        print(f"\n🏆 Winner: {analysis['winner']}")
        print(f"   Best Reward: {analysis.get('best_reward', 'N/A')}")
        print(f"\n📝 {analysis['interpretation']}")

        # Decision guidance
        print("\n📋 Recommendations:")
        if analysis["winner"] == "deep-search":
            print("   → Increase rollout_horizon_years in production config")
            print("   → The model benefits from seeing longer-term consequences")
        elif analysis["winner"] == "broad-search":
            print("   → Increase samples_per_group in production config")
            print("   → More samples help filter noise and stabilize gradients")
        else:
            print("   → Consider reward engineering approaches:")
            print("     - Add defensive bonuses for holding home centers")
            print("     - Add penalties for overextension")
            print("     - Consider curriculum learning")

    print()
    print("=" * 80)
    print("📊 WandB Visualization")
    print("=" * 80)
    print("\nTo create the Power Law plot in WandB:")
    print("1. Go to your WandB project: diplomacy-grpo")
    print("2. Filter runs by names starting with 'power-laws-'")
    print("3. Create a custom chart with:")
    print("   - X-Axis: 'power_law/cumulative_simulated_years'")
    print("   - Y-Axis: 'power_law/reward_at_compute'")
    print("4. Group by experiment tag to compare configurations")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Run the Power Laws scaling experiment on Modal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--run",
        type=str,
        choices=["A", "B", "C", "all"],
        default="all",
        help="Which configuration to run (A=baseline, B=deep, C=broad, all=sequential)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of training steps per run (default: 100)",
    )
    parser.add_argument(
        "--groups",
        type=int,
        default=8,
        help="Number of rollout groups per step (default: 8)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate (default: 1e-5)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configurations without launching jobs",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run all configurations in parallel (3x cost, 3x faster)",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model ID for inference (default: Qwen/Qwen2.5-7B-Instruct)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use smaller 3B model for faster iteration (good for testing hypotheses)",
    )
    parser.add_argument(
        "--detach",
        action="store_true",
        help="Fire and forget - launch sweep and exit without waiting for results",
    )

    args = parser.parse_args()

    # Apply --fast flag to use smaller model
    if args.fast:
        args.model_id = "Qwen/Qwen2.5-3B-Instruct"
        print("⚡ FAST MODE: Using Qwen2.5-3B-Instruct for faster iteration")

    # Determine which configs to run
    if args.run == "all":
        run_configs = ["A", "B", "C"]
        configs_to_show = list(SWEEP_CONFIGS.values())
    else:
        run_configs = [args.run]
        configs_to_show = [SWEEP_CONFIGS[args.run]]

    print("\n" + "=" * 80)
    print("🔬 POWER LAWS EXPERIMENT: Compute vs. Insight")
    print("=" * 80)
    print(f"\nTotal Steps: {args.steps}")
    print(f"Groups/Step: {args.groups}")
    print(f"Learning Rate: {args.lr}")
    print(f"Model: {args.model_id}")
    print(f"Parallel: {args.parallel}")
    print(f"Detach: {args.detach}")
    print("\nConfigurations to run:")

    for config in configs_to_show:
        sim_years = config.total_simulated_years(args.groups, args.steps)
        print(f"  [{config.name}] {config.description}")
        print(
            f"       Horizon: {config.rollout_horizon_years} | Samples: {config.samples_per_group}"
        )
        print(f"       Total Simulated Years: {sim_years}")
        print()

    if args.dry_run:
        print("🔍 DRY RUN - No jobs launched")
        return

    # Launch the sweep on Modal
    print("\n🚀 Launching Power Laws sweep on Modal...")
    print("   (You can close your laptop - the sweep runs in the cloud)")
    print()

    sweep_fn = modal.Function.from_name("diplomacy-grpo", "run_power_laws_sweep")

    if args.detach:
        # Fire and forget
        handle = sweep_fn.spawn(
            total_steps=args.steps,
            num_groups_per_step=args.groups,
            learning_rate=args.lr,
            model_id=args.model_id,
            run_configs=run_configs,
            parallel=args.parallel,
        )
        print(f"✅ Sweep launched! Function ID: {handle.object_id}")
        print("\nTo check status later:")
        print(f"   modal function get {handle.object_id}")
        print("\nOr monitor in Modal dashboard:")
        print("   https://modal.com/apps")
        return

    # Wait for results
    print("⏳ Waiting for sweep to complete...")
    print("   (Press Ctrl+C to detach - sweep will continue in cloud)")
    print()

    try:
        result = sweep_fn.remote(
            total_steps=args.steps,
            num_groups_per_step=args.groups,
            learning_rate=args.lr,
            model_id=args.model_id,
            run_configs=run_configs,
            parallel=args.parallel,
        )

        # Print results
        print("\n" + "=" * 80)
        print("🏁 SWEEP COMPLETE")
        print("=" * 80)
        print(f"\nTotal Duration: {result['total_duration_hours']:.2f} hours")

        if result.get("comparison"):
            print_comparison_table(result["comparison"], result.get("analysis", {}))

        # Save results to file
        timestamp = result.get("timestamp", datetime.now().strftime("%Y%m%d-%H%M%S"))
        output_file = f"power_laws_results_{timestamp}.json"
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\n📁 Full results saved to: {output_file}")

    except KeyboardInterrupt:
        print("\n\n⚠️ Detached from sweep (Ctrl+C)")
        print("   The sweep continues running on Modal.")
        print("   Check the Modal dashboard for progress.")


if __name__ == "__main__":
    main()
