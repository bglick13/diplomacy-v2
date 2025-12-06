#!/usr/bin/env python3
"""
Evaluation script for Diplomacy GRPO checkpoints.

This script evaluates trained checkpoints against baseline opponents
and logs results + visualizations to WandB.

Usage:
    # Evaluate a specific checkpoint
    python scripts/run_eval.py --checkpoint-path "benchmark-20251205/adapter_v50"

    # Evaluate against specific opponents
    python scripts/run_eval.py --checkpoint-path "..." --opponents random chaos

    # More games for statistical significance
    python scripts/run_eval.py --checkpoint-path "..." --games-per-opponent 20

    # Quick smoke test
    python scripts/run_eval.py --checkpoint-path "..." --smoke

    # List available checkpoints
    python scripts/run_eval.py --list-checkpoints

    # Fire and forget (detach after launching)
    python scripts/run_eval.py --checkpoint-path "..." --detach
"""

from __future__ import annotations

import argparse

import modal

from src.utils.config import EvalConfig, add_config_args, config_from_args

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

    for ckpt in sorted(checkpoints):
        print(f"  {ckpt}")

    return sorted(checkpoints)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Diplomacy GRPO checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Add EvalConfig args (auto-generated from Pydantic model)
    add_config_args(parser, EvalConfig)

    # Add script-specific args
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Quick smoke test (2 games, no visualizations)",
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
    if not hasattr(args, "checkpoint_path") or not args.checkpoint_path:
        parser.error("--checkpoint-path is required (use --list-checkpoints to see available)")

    # Apply smoke test overrides
    if args.smoke:
        args.games_per_opponent = 2
        args.visualize = False
        print("🔬 Smoke test mode: 2 games, no visualizations")

    # Build config from args
    cfg = config_from_args(args, EvalConfig)  # type: ignore[type-var]
    assert isinstance(cfg, EvalConfig)

    # Print config
    print("\n" + "=" * 60)
    print("🎯 CHECKPOINT EVALUATION")
    print("=" * 60)
    print(f"Checkpoint: {cfg.checkpoint_path}")
    print(f"Opponents: {cfg.opponents}")
    print(f"Games per opponent: {cfg.games_per_opponent}")
    print(f"Max years: {cfg.max_years}")
    print(f"Eval powers: {cfg.eval_powers}")
    print(f"Visualize: {cfg.visualize} (rate: {cfg.visualize_sample_rate})")
    print(f"WandB: {cfg.log_to_wandb}")
    print()

    # Get the evaluation function
    run_evaluation = modal.Function.from_name("diplomacy-grpo", "run_evaluation")

    if args.detach:
        # Fire and forget
        handle = run_evaluation.spawn(
            checkpoint_path=cfg.checkpoint_path,
            opponents=cfg.opponents,
            games_per_opponent=cfg.games_per_opponent,
            max_years=cfg.max_years,
            eval_powers=cfg.eval_powers,
            model_id=cfg.base_model_id,
            visualize=cfg.visualize,
            visualize_sample_rate=cfg.visualize_sample_rate,
            log_to_wandb=cfg.log_to_wandb,
            wandb_run_name=cfg.wandb_run_name,
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
        checkpoint_path=cfg.checkpoint_path,
        opponents=cfg.opponents,
        games_per_opponent=cfg.games_per_opponent,
        max_years=cfg.max_years,
        eval_powers=cfg.eval_powers,
        model_id=cfg.base_model_id,
        visualize=cfg.visualize,
        visualize_sample_rate=cfg.visualize_sample_rate,
        log_to_wandb=cfg.log_to_wandb,
        wandb_run_name=cfg.wandb_run_name,
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
