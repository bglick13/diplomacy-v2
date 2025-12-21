#!/usr/bin/env python3
"""
Initialize Benchmark Checkpoints.

This script copies checkpoints from a training run to the benchmarks directory,
creating a frozen set of reference points for absolute skill measurement.

Usage:
    # Copy checkpoints from a training run to benchmarks
    python scripts/init_benchmarks.py --run grpo-20251219-143727

    # Specify which steps to copy
    python scripts/init_benchmarks.py --run grpo-20251219-143727 --steps 10,50,150

    # List existing benchmarks
    python scripts/init_benchmarks.py --list

    # Verify benchmark integrity
    python scripts/init_benchmarks.py --verify
"""

from __future__ import annotations

import argparse

import modal

volume = modal.Volume.from_name("diplomacy-data")


def list_benchmarks() -> None:
    """List existing benchmark checkpoints."""
    print("\n📊 EXISTING BENCHMARKS")
    print("=" * 60)

    try:
        benchmark_dir = volume.listdir("/models/benchmarks")
        if not benchmark_dir:
            print("  No benchmarks found. Use --run to create them.")
            return

        for entry in sorted(benchmark_dir, key=lambda x: x.path):
            print(f"  {entry.path}")

    except Exception as e:
        print(f"  Benchmarks directory not found: {e}")
        print("  Use --run to create benchmarks from a training run.")


def list_checkpoints(run_name: str) -> list[str]:
    """List available checkpoints for a run."""
    try:
        run_dir = volume.listdir(f"/models/{run_name}")
        checkpoints = []
        for entry in run_dir:
            if entry.path.endswith("/") or "adapter_v" in entry.path:
                checkpoints.append(entry.path)
        return sorted(checkpoints)
    except Exception as e:
        print(f"  Error listing checkpoints: {e}")
        return []


def copy_checkpoint(src_path: str, dest_name: str) -> bool:
    """Copy a checkpoint to the benchmarks directory."""
    # We need to use Modal's volume copy functionality
    # Since Modal volumes don't have a direct copy API, we'll document the manual approach
    print(f"  Copying {src_path} -> /models/benchmarks/{dest_name}")

    # Check if destination exists
    try:
        dest_path = f"/models/benchmarks/{dest_name}"
        volume.listdir(dest_path)
        print(f"    Warning: {dest_path} already exists, skipping")
        return False
    except Exception:
        pass  # Destination doesn't exist, good to proceed

    # Note: Modal volumes don't have a built-in copy command
    # We need to use the Modal CLI or copy via container
    print(f"    Run: modal volume get diplomacy-data {src_path}")
    print(f"    Then: modal volume put diplomacy-data <local-path> {dest_name}")
    return True


def init_benchmarks(run_name: str, steps: list[int] | None = None) -> None:
    """Initialize benchmarks from a training run."""
    print(f"\n🔧 INITIALIZING BENCHMARKS FROM {run_name}")
    print("=" * 60)

    # Default steps if not specified
    if steps is None:
        steps = [10, 50, 150]

    print(f"Steps to copy: {steps}")

    # Check if run exists
    checkpoints = list_checkpoints(run_name)
    if not checkpoints:
        print(f"  Error: No checkpoints found for run '{run_name}'")
        return

    print(f"\nAvailable checkpoints in {run_name}:")
    for ckpt in checkpoints[-10:]:  # Show last 10
        print(f"  {ckpt}")

    # Create benchmarks directory if needed
    print("\n📁 Creating benchmarks directory...")
    print("  Run: modal volume put diplomacy-data --empty /models/benchmarks")

    # Copy each step
    print("\n📋 To copy checkpoints, run these commands:")
    print("-" * 60)

    for step in steps:
        src_adapter = f"/models/{run_name}/adapter_v{step}"
        dest_name = f"frozen_v{step}"

        # Check if source exists
        matching = [c for c in checkpoints if f"adapter_v{step}" in c]
        if not matching:
            print(f"\n  Warning: adapter_v{step} not found in {run_name}")
            # Find closest step
            available_steps = []
            for c in checkpoints:
                try:
                    s = int(c.split("adapter_v")[-1].rstrip("/"))
                    available_steps.append(s)
                except ValueError:
                    pass
            if available_steps:
                closest = min(available_steps, key=lambda x: abs(x - step))
                print(f"  Closest available: adapter_v{closest}")
            continue

        print(f"\n# Copy adapter_v{step} -> frozen_v{step}")
        print(f"modal volume get diplomacy-data {src_adapter} /tmp/{dest_name}")
        print(f"modal volume put diplomacy-data /tmp/{dest_name} /models/benchmarks/{dest_name}")

    print("\n" + "-" * 60)
    print("\nAfter running the commands above, verify with:")
    print("  python scripts/init_benchmarks.py --list")


def verify_benchmarks() -> None:
    """Verify benchmark checkpoint integrity."""
    from src.league.benchmarks import CHECKPOINT_BENCHMARKS

    print("\n🔍 VERIFYING BENCHMARKS")
    print("=" * 60)

    all_ok = True

    for benchmark in CHECKPOINT_BENCHMARKS:
        path = f"/models/{benchmark.path}"
        try:
            files = volume.listdir(path)
            file_count = len(files) if files else 0
            print(f"  {benchmark.name}: {path} ({file_count} files)")
        except Exception as e:
            print(f"  {benchmark.name}: MISSING - {e}")
            all_ok = False

    if all_ok:
        print("\n✅ All benchmark checkpoints verified!")
    else:
        print("\n❌ Some benchmarks are missing. Run --run to create them.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Initialize benchmark checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--run",
        type=str,
        help="Training run name to copy checkpoints from",
    )
    parser.add_argument(
        "--steps",
        type=str,
        help="Comma-separated list of steps to copy (default: 10,50,150)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List existing benchmark checkpoints",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify benchmark checkpoint integrity",
    )

    args = parser.parse_args()

    if args.list:
        list_benchmarks()
        return

    if args.verify:
        verify_benchmarks()
        return

    if not args.run:
        parser.error("--run is required to initialize benchmarks")

    # Parse steps
    steps = None
    if args.steps:
        try:
            steps = [int(s.strip()) for s in args.steps.split(",")]
        except ValueError:
            parser.error(f"Invalid steps format: {args.steps}")

    init_benchmarks(args.run, steps)


if __name__ == "__main__":
    main()
