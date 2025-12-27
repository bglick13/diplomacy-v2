#!/usr/bin/env python3
"""
Sweep analysis: Compare multiple runs using fixed reference metrics.

The key insight: Win rate against a dynamic league is meaningless.
Use FIXED references (base_model, baseline bots) to measure true learning.

Usage:
    uv run python .claude/skills/experiment-analysis/analyze_sweep.py <run1> <run2> ...
    uv run python .claude/skills/experiment-analysis/analyze_sweep.py --sweep stability-ablation

Example:
    uv run python .claude/skills/experiment-analysis/analyze_sweep.py \\
        stability-ablation-A-20251226-124256 \\
        stability-ablation-B-20251226-124256
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import wandb

# Fixed reference agents (these don't change, so Elo changes = learning)
FIXED_REFERENCES = [
    "base_model",
    "chaos_bot",
    "defensive_bot",
    "territorial_bot",
    "coordinated_bot",
]


@dataclass
class RunMetrics:
    """Key metrics for a single run."""

    name: str
    short_name: str
    steps: int

    # Fixed reference Elo (lower = we're beating them more)
    base_model_elo: float | None
    chaos_bot_elo: float | None
    defensive_bot_elo: float | None
    territorial_bot_elo: float | None
    coordinated_bot_elo: float | None

    # Best checkpoint Elo
    best_checkpoint_elo: float | None
    best_checkpoint_name: str | None

    # Stability metrics
    kl_mean: float | None
    kl_max: float | None
    grad_norm: float | None

    # Performance metrics
    avg_sc: float | None
    win_rate: float | None
    top3_rate: float | None
    reward_mean: float | None

    @property
    def fixed_ref_total_drop(self) -> float:
        """Total Elo drop across all fixed references (higher = more learning)."""
        # Initial Elo is ~1000 for all agents
        drops = []
        for ref in [
            "base_model_elo",
            "chaos_bot_elo",
            "defensive_bot_elo",
            "territorial_bot_elo",
            "coordinated_bot_elo",
        ]:
            val = getattr(self, ref)
            if val is not None:
                drops.append(1000 - val)  # Assume started at ~1000
        return sum(drops) if drops else 0

    @property
    def elo_gap(self) -> float | None:
        """Gap between best checkpoint and base model."""
        if self.best_checkpoint_elo and self.base_model_elo:
            return self.best_checkpoint_elo - self.base_model_elo
        return None


def fetch_run_metrics(run_name: str, project: str = "diplomacy-grpo") -> RunMetrics | None:
    """Fetch metrics for a single run."""
    api = wandb.Api()

    # Find run
    runs = api.runs(project, filters={"displayName": run_name})
    runs_list = list(runs)

    if not runs_list:
        print(f"  Warning: Could not find run {run_name}")
        return None

    run = runs_list[0]
    summary = run.summary

    # Extract short name (e.g., "A-baseline" from "stability-ablation-A-20251226-124256")
    parts = run_name.split("-")
    if len(parts) >= 3:
        short_name = parts[2]  # "A", "B", etc.
    else:
        short_name = run_name[:20]

    # Get best checkpoint Elo
    best_ckpt_elo = None
    best_ckpt_name = None
    for key, val in summary.items():
        if key.startswith("elo/") and "adapter" in key:
            if best_ckpt_elo is None or val > best_ckpt_elo:
                best_ckpt_elo = val
                best_ckpt_name = key.replace("elo/", "")

    return RunMetrics(
        name=run_name,
        short_name=short_name,
        steps=summary.get("_step", 0),
        base_model_elo=summary.get("elo/base_model"),
        chaos_bot_elo=summary.get("elo/chaos_bot"),
        defensive_bot_elo=summary.get("elo/defensive_bot"),
        territorial_bot_elo=summary.get("elo/territorial_bot"),
        coordinated_bot_elo=summary.get("elo/coordinated_bot"),
        best_checkpoint_elo=best_ckpt_elo,
        best_checkpoint_name=best_ckpt_name,
        kl_mean=summary.get("kl/mean"),
        kl_max=summary.get("kl/max"),
        grad_norm=summary.get("benchmark/grad_norm"),
        avg_sc=summary.get("game/avg_sc_count"),
        win_rate=summary.get("game/win_bonus_rate"),
        top3_rate=summary.get("placement/top3_rate"),
        reward_mean=summary.get("benchmark/reward_mean"),
    )


def find_sweep_runs(sweep_prefix: str, project: str = "diplomacy-grpo") -> list[str]:
    """Find all runs matching a sweep prefix."""
    api = wandb.Api()
    runs = api.runs(project)

    matching = []
    for run in runs:
        if run.name.startswith(sweep_prefix):
            matching.append(run.name)

    return sorted(matching)


def fmt(val: float | None, decimals: int = 2, pct: bool = False) -> str:
    """Format a value for display."""
    if val is None:
        return "-"
    if pct:
        return f"{val * 100:.0f}%"
    if decimals == 0:
        return f"{val:.0f}"
    return f"{val:.{decimals}f}"


def print_comparison_table(runs: list[RunMetrics]) -> None:
    """Print a comparison table of all runs."""
    print("\n" + "=" * 80)
    print("SWEEP COMPARISON: Fixed Reference Analysis")
    print("=" * 80)

    # Header
    print(
        f"\n{'Run':<12} | {'Steps':>5} | {'Base':>6} | {'Chaos':>6} | {'Def':>6} | {'Terr':>6} | {'Best Ckpt':>9} | {'Gap':>5}"
    )
    print("-" * 80)

    for r in runs:
        gap = fmt(r.elo_gap, 0) if r.elo_gap else "-"
        print(
            f"{r.short_name:<12} | {r.steps:>5} | "
            f"{fmt(r.base_model_elo, 0):>6} | "
            f"{fmt(r.chaos_bot_elo, 0):>6} | "
            f"{fmt(r.defensive_bot_elo, 0):>6} | "
            f"{fmt(r.territorial_bot_elo, 0):>6} | "
            f"{fmt(r.best_checkpoint_elo, 0):>9} | "
            f"{gap:>5}"
        )

    # Stability table
    print(
        f"\n{'Run':<12} | {'KL Mean':>8} | {'KL Max':>8} | {'Grad':>6} | {'SC':>5} | {'Win%':>5} | {'Top3%':>6} | {'Reward':>7}"
    )
    print("-" * 80)

    for r in runs:
        print(
            f"{r.short_name:<12} | "
            f"{fmt(r.kl_mean, 3):>8} | "
            f"{fmt(r.kl_max, 1):>8} | "
            f"{fmt(r.grad_norm, 1):>6} | "
            f"{fmt(r.avg_sc, 2):>5} | "
            f"{fmt(r.win_rate, 0, pct=True):>5} | "
            f"{fmt(r.top3_rate, 0, pct=True):>6} | "
            f"{fmt(r.reward_mean, 2):>7}"
        )


def print_ranking(runs: list[RunMetrics]) -> None:
    """Print runs ranked by learning signal."""
    print("\n" + "=" * 80)
    print("RANKING BY LEARNING SIGNAL")
    print("=" * 80)

    # Rank by Elo gap (best checkpoint - base model)
    ranked_by_gap = sorted(
        [r for r in runs if r.elo_gap is not None],
        key=lambda x: x.elo_gap or 0,
        reverse=True,
    )

    print("\nBy Elo Gap (Best Checkpoint - Base Model):")
    print("  Higher = trained model beats base model by more")
    for i, r in enumerate(ranked_by_gap, 1):
        print(f"  {i}. {r.short_name}: +{r.elo_gap:.0f} Elo")

    # Rank by base model Elo (lower = better)
    ranked_by_base = sorted(
        [r for r in runs if r.base_model_elo is not None],
        key=lambda x: x.base_model_elo or 9999,
    )

    print("\nBy Base Model Elo (Lower = League Beats It More):")
    for i, r in enumerate(ranked_by_base, 1):
        print(f"  {i}. {r.short_name}: {r.base_model_elo:.0f} Elo")

    # Rank by total fixed reference drop
    ranked_by_drop = sorted(
        runs,
        key=lambda x: x.fixed_ref_total_drop,
        reverse=True,
    )

    print("\nBy Total Fixed Reference Elo Drop:")
    print("  Sum of (1000 - final_elo) for all baseline bots")
    for i, r in enumerate(ranked_by_drop, 1):
        print(f"  {i}. {r.short_name}: {r.fixed_ref_total_drop:.0f} total drop")


def print_verdict(runs: list[RunMetrics]) -> None:
    """Print analysis verdict."""
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    # Find best by Elo gap
    best_gap = max((r for r in runs if r.elo_gap), key=lambda x: x.elo_gap or 0, default=None)
    if best_gap:
        print(f"\nBest by Elo Gap: {best_gap.short_name}")
        print(f"  - Best checkpoint: {best_gap.best_checkpoint_elo:.0f} Elo")
        print(f"  - Base model: {best_gap.base_model_elo:.0f} Elo")
        print(f"  - Gap: +{best_gap.elo_gap:.0f}")

    # Stability check
    unstable = [r for r in runs if r.kl_max and r.kl_max > 10]
    if unstable:
        print("\nStability Warning:")
        for r in unstable:
            print(f"  - {r.short_name}: KL max = {r.kl_max:.1f} (>10 = unstable)")

    # Check if any run clearly dominates
    if best_gap:
        others = [r for r in runs if r != best_gap and r.elo_gap]
        if others:
            runner_up = max(others, key=lambda x: x.elo_gap or 0)
            margin = (best_gap.elo_gap or 0) - (runner_up.elo_gap or 0)
            if margin > 20:
                print(f"\nClear Winner: {best_gap.short_name} (+{margin:.0f} Elo margin)")
            else:
                print(
                    f"\nClose Race: {best_gap.short_name} vs {runner_up.short_name} ({margin:.0f} Elo margin)"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze sweep runs using fixed reference metrics")
    parser.add_argument(
        "runs",
        nargs="*",
        help="Run names to compare",
    )
    parser.add_argument(
        "--sweep",
        type=str,
        help="Sweep prefix to find runs (e.g., 'stability-ablation')",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="diplomacy-grpo",
        help="WandB project name",
    )

    args = parser.parse_args()

    # Get run names
    if args.sweep:
        print(f"Finding runs matching prefix: {args.sweep}")
        run_names = find_sweep_runs(args.sweep, args.project)
        if not run_names:
            print(f"No runs found matching prefix: {args.sweep}")
            sys.exit(1)
        print(f"Found {len(run_names)} runs")
    elif args.runs:
        run_names = args.runs
    else:
        parser.print_help()
        sys.exit(1)

    # Fetch metrics for all runs
    print("\nFetching metrics...")
    runs = []
    for name in run_names:
        metrics = fetch_run_metrics(name, args.project)
        if metrics:
            runs.append(metrics)

    if not runs:
        print("No valid runs found")
        sys.exit(1)

    # Print analysis
    print_comparison_table(runs)
    print_ranking(runs)
    print_verdict(runs)


if __name__ == "__main__":
    main()
