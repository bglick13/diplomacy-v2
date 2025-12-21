#!/usr/bin/env python3
"""
Benchmark Evaluation Script for Diplomacy GRPO Checkpoints.

This script evaluates checkpoints against a fixed benchmark suite to measure
absolute skill improvement. Unlike ELO (which is relative to peers), benchmark
scores provide consistent reference points across training runs.

Usage:
    # Evaluate a checkpoint against full benchmark suite
    python scripts/run_benchmark_eval.py --checkpoint "grpo-20251219-143727/adapter_v150"

    # Quick evaluation (3 benchmarks only)
    python scripts/run_benchmark_eval.py --checkpoint "..." --quick

    # More games for statistical significance
    python scripts/run_benchmark_eval.py --checkpoint "..." --games 10

    # Fire and forget
    python scripts/run_benchmark_eval.py --checkpoint "..." --detach

    # List available benchmarks
    python scripts/run_benchmark_eval.py --list-benchmarks
"""

from __future__ import annotations

import argparse

import modal


def list_benchmarks() -> None:
    """List available benchmark agents."""
    from src.league.benchmarks import BENCHMARK_SUITE, QUICK_BENCHMARK_SUITE

    print("\n📊 BENCHMARK SUITE")
    print("=" * 60)
    print("\nFull Suite:")
    for b in BENCHMARK_SUITE:
        floor_pct = f"{b.expected_winrate_floor:.0%}"
        print(f"  {b.name:20s} ({b.tier.value:10s}) - target: {floor_pct:>4s}")
        print(f"      {b.description}")

    print("\nQuick Suite (for frequent evaluation):")
    for b in QUICK_BENCHMARK_SUITE:
        print(f"  {b.name}")

    print()


def list_checkpoints() -> None:
    """List available checkpoints in the Modal volume."""
    volume = modal.Volume.from_name("diplomacy-data")

    print("\n📁 Listing checkpoints in /data/models...")

    try:
        models_dir = volume.listdir("/models")
    except Exception as e:
        print(f"  Error: {e}")
        return

    checkpoints = []
    for run_name in models_dir:
        run_path = run_name.path
        try:
            run_files = volume.listdir(run_path)
            for file in run_files:
                checkpoints.append(file.path)
        except Exception:
            continue

    for ckpt in sorted(checkpoints):
        print(f"  {ckpt}")

    return


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate checkpoints against fixed benchmark suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Checkpoint path (e.g., 'grpo-20251219-143727/adapter_v150')",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=5,
        help="Games per benchmark agent (default: 5)",
    )
    parser.add_argument(
        "--max-years",
        type=int,
        default=8,
        help="Maximum game length in years (default: 8)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use quick benchmark suite (3 benchmarks instead of full suite)",
    )
    parser.add_argument(
        "--detach",
        action="store_true",
        help="Launch and exit without waiting for results",
    )
    parser.add_argument(
        "--list-benchmarks",
        action="store_true",
        help="List available benchmark agents and exit",
    )
    parser.add_argument(
        "--list-checkpoints",
        action="store_true",
        help="List available checkpoints and exit",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Base model ID",
    )

    args = parser.parse_args()

    # List modes
    if args.list_benchmarks:
        list_benchmarks()
        return

    if args.list_checkpoints:
        list_checkpoints()
        return

    # Require checkpoint for evaluation
    if not args.checkpoint:
        parser.error("--checkpoint is required (use --list-checkpoints to see available)")

    # Print config
    suite_type = "Quick" if args.quick else "Full"
    print("\n" + "=" * 60)
    print("📊 BENCHMARK EVALUATION")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Suite: {suite_type}")
    print(f"Games per benchmark: {args.games}")
    print(f"Max years: {args.max_years}")
    print()

    # Get the evaluation function
    evaluate_against_benchmarks = modal.Function.from_name(
        "diplomacy-grpo", "evaluate_against_benchmarks"
    )

    if args.detach:
        # Fire and forget
        handle = evaluate_against_benchmarks.spawn(
            challenger_path=args.checkpoint,
            games_per_benchmark=args.games,
            max_years=args.max_years,
            model_id=args.model_id,
            use_quick_suite=args.quick,
        )
        print(f"✅ Benchmark evaluation launched! Function ID: {handle.object_id}")
        print("\nTo check status later:")
        print(f"   modal function get {handle.object_id}")
        return

    # Run and wait
    print("🚀 Running benchmark evaluation...")
    print("   (This may take a while depending on game count)")
    print()

    result = evaluate_against_benchmarks.remote(
        challenger_path=args.checkpoint,
        games_per_benchmark=args.games,
        max_years=args.max_years,
        model_id=args.model_id,
        use_quick_suite=args.quick,
    )

    # Print results
    print("\n" + "=" * 60)
    print("📊 BENCHMARK EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Duration: {result['duration_s']:.1f}s")
    print()
    print(f"Overall Benchmark Score: {result['overall_score']:.1f}/100")
    print(f"Benchmarks Passed: {result['benchmarks_passed']}/{result['benchmarks_total']}")
    print()
    print("Results:")
    print("-" * 40)

    for name, r in result["results"].items():
        status = "PASS" if r["meets_floor"] else "FAIL"
        print(f"  vs {name:20s}: {r['win_rate']:>5.1%} [{status}]")

    print()


if __name__ == "__main__":
    main()
