#!/usr/bin/env python3
"""
Launch a sweep experiment on Modal.

This script loads a sweep configuration from YAML, validates it, and spawns
the Modal orchestrator which runs training jobs sequentially in the cloud.

Usage:
    # Launch all runs in a sweep (fire-and-forget)
    python scripts/launch_sweep.py experiments/sweeps/my-ablation/

    # Launch specific runs only
    python scripts/launch_sweep.py experiments/sweeps/my-ablation/ --run A B

    # Dry run (validate config, show what would run)
    python scripts/launch_sweep.py experiments/sweeps/my-ablation/ --dry-run

    # Check status of a running sweep
    python scripts/launch_sweep.py experiments/sweeps/my-ablation/ --status

    # Show sweep info without launching
    python scripts/launch_sweep.py experiments/sweeps/my-ablation/ --info

    # List all sweeps
    python scripts/launch_sweep.py --list
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import modal

from src.utils.sweep_config import SweepConfig


def print_sweep_info(sweep_config: SweepConfig, run_ids: list[str] | None = None) -> None:
    """Print detailed information about a sweep."""
    meta = sweep_config.metadata
    runs_to_show = run_ids or sweep_config.get_run_ids()

    print("\n" + "=" * 70)
    print(f"SWEEP: {meta.name}")
    print("=" * 70)
    print(f"\nDescription: {meta.description}")
    if meta.hypothesis:
        print(f"\nHypothesis:\n  {meta.hypothesis}")
    print(f"\nExperiment tag prefix: {meta.experiment_tag_prefix}")
    print(f"Author: {meta.author or 'N/A'}")
    print(f"Created: {meta.created}")

    print("\n" + "-" * 70)
    print("DEFAULTS")
    print("-" * 70)
    if sweep_config.defaults:
        for key, value in sweep_config.defaults.items():
            print(f"  {key}: {value}")
    else:
        print("  (none)")

    print("\n" + "-" * 70)
    print("RUNS")
    print("-" * 70)
    for run_id in runs_to_show:
        run = sweep_config.runs[run_id]
        print(f"\n  [{run_id}] {run.name}")
        print(f"      {run.description}")
        if run.config:
            print("      Config overrides:")
            for key, value in run.config.items():
                print(f"        {key}: {value}")

    print("\n" + "-" * 70)
    print("ANALYSIS")
    print("-" * 70)
    analysis = sweep_config.analysis
    print(f"  Primary metric: {analysis.primary_metric}")
    print(f"  Secondary metrics: {', '.join(analysis.secondary_metrics)}")
    if analysis.expected_ranking:
        print(f"  Expected ranking: {' > '.join(analysis.expected_ranking)}")
    if analysis.success_criteria:
        print("  Success criteria:")
        for criterion in analysis.success_criteria:
            print(f"    - {criterion}")

    print("\n" + "=" * 70 + "\n")


def get_sweep_status(sweep_name: str) -> None:
    """Get and print the status of a sweep."""
    get_status = modal.Function.from_name("diplomacy-grpo", "get_sweep_status")

    print(f"\nChecking status of sweep: {sweep_name}")
    try:
        status = get_status.remote(sweep_name)

        if status["status"] == "not_found":
            print(f"  Sweep not found: {sweep_name}")
            print("  (Either it hasn't been launched or the name is incorrect)")
            return

        print(f"\n  Status: {status['status']}")
        print(f"  Started: {status['started_at']}")
        print(f"  Progress: {status['progress']}")
        print(f"  Current run: {status['current_run'] or '(none)'}")
        print(f"  Completed: {', '.join(status['completed_runs']) or '(none)'}")
        print(f"  Pending: {', '.join(status['pending_runs']) or '(none)'}")

        if status["run_names"]:
            print("\n  WandB run names:")
            for run_id, run_name in status["run_names"].items():
                print(f"    [{run_id}] {run_name}")

    except Exception as e:
        print(f"  Error checking status: {e}")
        print("  (Is the sweep app deployed? Run: uv run modal deploy -m src.apps.deploy)")


def list_sweeps() -> None:
    """List all sweeps on the volume."""
    list_fn = modal.Function.from_name("diplomacy-grpo", "list_sweeps")

    print("\nListing all sweeps...")
    try:
        sweeps = list_fn.remote()

        if not sweeps:
            print("  No sweeps found.")
            return

        print(f"\n  Found {len(sweeps)} sweep(s):\n")
        for sweep in sweeps:
            status_emoji = "" if sweep["status"] == "in_progress" else ""
            print(f"  {status_emoji} {sweep['name']}")
            print(f"       Progress: {sweep['progress']}")
            print(f"       Started: {sweep['started_at']}")
            if sweep["current_run"]:
                print(f"       Current: {sweep['current_run']}")
            print()

    except Exception as e:
        print(f"  Error listing sweeps: {e}")
        print("  (Is the sweep app deployed? Run: uv run modal deploy -m src.apps.deploy)")


def launch_sweep(
    sweep_config: SweepConfig,
    run_ids: list[str] | None = None,
    dry_run: bool = False,
) -> None:
    """Launch a sweep on Modal."""
    meta = sweep_config.metadata
    requested_runs = run_ids or sweep_config.get_run_ids()

    print("\n" + "=" * 70)
    print(f"LAUNCHING SWEEP: {meta.name}")
    print("=" * 70)
    print(f"\nRuns to execute: {', '.join(requested_runs)}")
    print(f"Dry run: {dry_run}")

    # Get reference to orchestrator
    orchestrate_sweep = modal.Function.from_name("diplomacy-grpo", "orchestrate_sweep")

    if dry_run:
        print("\n[DRY RUN] Validating configuration...")
        # Run dry_run remotely to validate
        result = orchestrate_sweep.remote(
            sweep_config_dict=sweep_config.model_dump(),
            run_ids=requested_runs,
            dry_run=True,
        )
        print("\n[DRY RUN] Configuration valid!")
        print(f"  Would execute runs: {result.get('pending_runs', [])}")
        print(f"  Already completed: {result.get('completed_runs', [])}")
        return

    # Fire and forget - spawn and exit
    print("\nSpawning sweep orchestrator on Modal...")
    handle = orchestrate_sweep.spawn(
        sweep_config_dict=sweep_config.model_dump(),
        run_ids=requested_runs,
        dry_run=False,
    )

    print(f"\n Sweep launched! Function ID: {handle.object_id}")
    print("\nThe sweep is now running in the cloud. You can close your laptop.")
    print("\nTo check status:")
    print(f"  python scripts/launch_sweep.py experiments/sweeps/{meta.name}/ --status")
    print("\nTo monitor in Modal dashboard:")
    print("  https://modal.com/apps")
    print("\nResults will be logged to WandB with tag prefix:")
    print(f"  {meta.experiment_tag_prefix}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Launch a sweep experiment on Modal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "path",
        type=str,
        nargs="?",
        help="Path to sweep directory or sweep.yaml file",
    )
    parser.add_argument(
        "--run",
        type=str,
        nargs="+",
        help="Specific run IDs to execute (e.g., --run A B)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and show what would run without launching",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show sweep info without launching",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Check status of a running sweep",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all sweeps on the volume",
    )

    args = parser.parse_args()

    # Handle --list (doesn't need path)
    if args.list:
        list_sweeps()
        return

    # All other commands need a path
    if not args.path:
        parser.error("path is required (unless using --list)")
        return

    # Load sweep config
    path = Path(args.path)
    try:
        sweep_config = SweepConfig.from_yaml(path)
    except FileNotFoundError:
        print(f"Error: Sweep config not found at {path}")
        print("Expected sweep.yaml in the specified directory.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading sweep config: {e}")
        sys.exit(1)

    # Validate run IDs if specified
    if args.run:
        valid_runs = sweep_config.get_run_ids()
        invalid = [r for r in args.run if r not in valid_runs]
        if invalid:
            print(f"Error: Invalid run IDs: {invalid}")
            print(f"Valid run IDs: {valid_runs}")
            sys.exit(1)

    # Handle commands
    if args.info:
        print_sweep_info(sweep_config, args.run)
        return

    if args.status:
        get_sweep_status(sweep_config.metadata.name)
        return

    # Default: launch the sweep
    launch_sweep(sweep_config, args.run, args.dry_run)


if __name__ == "__main__":
    main()
