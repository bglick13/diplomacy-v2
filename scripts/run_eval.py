#!/usr/bin/env python3
"""
Evaluation script for Diplomacy GRPO checkpoints.

This script evaluates trained checkpoints against baseline opponents
and logs results + visualizations to WandB.

Usage:
    # Evaluate a specific checkpoint
    python scripts/run_eval.py --checkpoint "benchmark-20251205/adapter_v50"

    # Evaluate against specific opponents
    python scripts/run_eval.py --checkpoint "..." --opponents random chaos

    # More games for statistical significance
    python scripts/run_eval.py --checkpoint "..." --games 20

    # Quick smoke test
    python scripts/run_eval.py --checkpoint "..." --games 2 --smoke

    # List available checkpoints
    python scripts/run_eval.py --list-checkpoints

    # Fire and forget (detach after launching)
    python scripts/run_eval.py --checkpoint "..." --detach
"""

import argparse

import modal

volume = modal.Volume.from_name("diplomacy-data")


def list_checkpoints():
    """List available checkpoints in the Modal volume."""
    print("\n📁 Listing checkpoints in /data/models...")

    models_dir = volume.listdir("/models")

    checkpoints = []
    for run_name in models_dir:
        run_path = run_name.path
        run_files = volume.listdir(run_path)
        for file in run_files:
            checkpoints.append(file.path)
    return sorted(checkpoints)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Diplomacy GRPO checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Checkpoint path relative to /data/models (e.g., 'benchmark-20251205/adapter_v50')",
    )
    parser.add_argument(
        "--opponents",
        nargs="+",
        default=["random", "chaos"],
        help="Opponent types to evaluate against (default: random chaos)",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=10,
        help="Number of games per opponent (default: 10)",
    )
    parser.add_argument(
        "--max-years",
        type=int,
        default=10,
        help="Maximum game length in years (default: 10)",
    )
    parser.add_argument(
        "--powers",
        nargs="+",
        default=None,
        help="Powers for checkpoint to play (default: FRANCE)",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Base model ID (default: Qwen/Qwen2.5-7B-Instruct)",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Disable visualization generation",
    )
    parser.add_argument(
        "--viz-rate",
        type=float,
        default=0.3,
        help="Fraction of games to visualize (default: 0.3)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable WandB logging",
    )
    parser.add_argument(
        "--wandb-name",
        type=str,
        default=None,
        help="Custom WandB run name",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Quick smoke test (2 games, no viz)",
    )
    parser.add_argument(
        "--detach",
        action="store_true",
        help="Launch and exit without waiting for results",
    )
    parser.add_argument(
        "--list-checkpoints",
        action="store_true",
        help="List available checkpoints and exit",
    )

    args = parser.parse_args()

    # List checkpoints mode
    if args.list_checkpoints:
        list_checkpoints()
        return

    # Require checkpoint for evaluation
    if not args.checkpoint:
        parser.error("--checkpoint is required (use --list-checkpoints to see available)")

    # Apply smoke test overrides
    if args.smoke:
        args.games = 2
        args.no_viz = True
        print("🔬 Smoke test mode: 2 games, no visualizations")

    # Print config
    print("\n" + "=" * 60)
    print("🎯 CHECKPOINT EVALUATION")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Opponents: {args.opponents}")
    print(f"Games per opponent: {args.games}")
    print(f"Max years: {args.max_years}")
    print(f"Eval powers: {args.powers or ['FRANCE']}")
    print(f"Visualize: {not args.no_viz} (rate: {args.viz_rate})")
    print(f"WandB: {not args.no_wandb}")
    print()

    # Get the evaluation function
    run_evaluation = modal.Function.from_name("diplomacy-grpo", "run_evaluation")

    if args.detach:
        # Fire and forget
        handle = run_evaluation.spawn(
            checkpoint_path=args.checkpoint,
            opponents=args.opponents,
            games_per_opponent=args.games,
            max_years=args.max_years,
            eval_powers=args.powers,
            model_id=args.model_id,
            visualize=not args.no_viz,
            visualize_sample_rate=args.viz_rate,
            log_to_wandb=not args.no_wandb,
            wandb_run_name=args.wandb_name,
        )
        print(f"✅ Evaluation launched! Function ID: {handle.object_id}")
        print("\nTo check status later:")
        print(f"   modal function get {handle.object_id}")
        print("\nResults will be logged to WandB when complete.")
        return

    # Run and wait
    print("🚀 Running evaluation...")
    print("   (This may take a while depending on game count)")
    print()

    result = run_evaluation.remote(
        checkpoint_path=args.checkpoint,
        opponents=args.opponents,
        games_per_opponent=args.games,
        max_years=args.max_years,
        eval_powers=args.powers,
        model_id=args.model_id,
        visualize=not args.no_viz,
        visualize_sample_rate=args.viz_rate,
        log_to_wandb=not args.no_wandb,
        wandb_run_name=args.wandb_name,
    )

    # Print results
    print("\n" + "=" * 60)
    print("✅ EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Duration: {result['total_duration_s']:.1f}s")
    print()
    print("Results:")
    print("-" * 40)

    for r in result["results"]:
        print(f"\n  vs {r['opponent']}:")
        print(f"    Win Rate:      {r['win_rate']:.1%}")
        print(f"    Survival Rate: {r['survival_rate']:.1%}")
        print(f"    Avg Centers:   {r['avg_centers']:.1f}")

    if result.get("visualization_paths"):
        print(f"\n📊 Visualizations saved: {len(result['visualization_paths'])}")
        print("   (Check WandB for interactive replays)")

    print()


if __name__ == "__main__":
    main()
