#!/usr/bin/env python3
"""
Elo trajectory analysis helper for experiment-analysis skill.

Usage:
    uv run python .claude/skills/experiment-analysis/analyze_elo.py <run-name>

Example:
    uv run python .claude/skills/experiment-analysis/analyze_elo.py grpo-20251222-191408
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import wandb


@dataclass
class AgentEloStats:
    """Statistics for a single agent's Elo trajectory."""

    name: str
    initial_elo: float
    peak_elo: float
    peak_step: int
    final_elo: float
    is_checkpoint: bool

    @property
    def trend(self) -> str:
        """Classify the Elo trend."""
        delta = self.final_elo - self.peak_elo
        if delta > -5:
            return "Near peak"
        elif delta > -20:
            return "Slight decline"
        else:
            return "Declining"

    @property
    def change(self) -> float:
        """Change from initial to final."""
        return self.final_elo - self.initial_elo


def get_elo_metrics(run_name: str, project: str = "diplomacy-grpo") -> dict:
    """Fetch all elo/* metrics from WandB."""
    api = wandb.Api()

    # Try to find run by name
    runs = api.runs(project, filters={"displayName": run_name})
    runs_list = list(runs)

    if not runs_list:
        # Try by run ID
        try:
            run = api.run(f"{project}/{run_name}")
        except Exception:
            print(f"Could not find run: {run_name}")
            sys.exit(1)
    else:
        run = runs_list[0]

    print(f"Found run: {run.name} ({run.id})")

    # Get history
    history = run.history(x_axis="_step")

    if history.empty:
        print("No history data found")
        sys.exit(1)

    # Filter for elo/* columns
    elo_cols = [col for col in history.columns if col.startswith("elo/")]

    if not elo_cols:
        print("No elo/* metrics found in run")
        sys.exit(1)

    # Build metrics dict: agent_name -> list of (step, elo) tuples
    metrics: dict[str, list[tuple[int, float]]] = {}

    for col in elo_cols:
        # Extract agent name from metric name (e.g., "elo/adapter_v10" -> "adapter_v10")
        agent_name = col.replace("elo/", "").replace("_", "/", 1)  # Restore run-name/adapter format

        values = []
        for idx, row in history.iterrows():
            step = int(row.get("_step", idx))
            elo = row.get(col)
            if elo is not None and not (isinstance(elo, float) and elo != elo):  # Not NaN
                values.append((step, float(elo)))

        if values:
            metrics[agent_name] = sorted(values, key=lambda x: x[0])

    return metrics


def analyze_agent(name: str, data: list[tuple[int, float]]) -> AgentEloStats:
    """Analyze a single agent's Elo trajectory."""
    # Identify if checkpoint or baseline
    baselines = {
        "base_model",
        "chaos_bot",
        "defensive_bot",
        "territorial_bot",
        "coordinated_bot",
        "random_bot",
    }
    is_checkpoint = name not in baselines and "adapter" in name.lower()

    # Get initial, peak, and final
    initial_elo = data[0][1]
    peak_elo = max(elo for _, elo in data)
    peak_step = next(step for step, elo in data if elo == peak_elo)
    final_elo = data[-1][1]

    return AgentEloStats(
        name=name,
        initial_elo=initial_elo,
        peak_elo=peak_elo,
        peak_step=peak_step,
        final_elo=final_elo,
        is_checkpoint=is_checkpoint,
    )


def print_table(headers: list[str], rows: list[list[str]], title: str) -> None:
    """Print a formatted table."""
    # Calculate column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    # Print
    print(f"\n{title}")
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, widths, strict=False))
    print(header_line)
    print("-" * len(header_line))
    for row in rows:
        print(" | ".join(cell.ljust(w) for cell, w in zip(row, widths, strict=False)))


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: uv run python .claude/skills/experiment-analysis/analyze_elo.py <run-name>")
        print(
            "Example: uv run python .claude/skills/experiment-analysis/analyze_elo.py grpo-20251222-191408"
        )
        sys.exit(1)

    run_name = sys.argv[1]

    print(f"=== Elo Analysis: {run_name} ===")

    # Fetch metrics
    metrics = get_elo_metrics(run_name)

    # Analyze each agent
    stats = [analyze_agent(name, data) for name, data in metrics.items()]

    # Separate checkpoints and baselines
    checkpoints = sorted([s for s in stats if s.is_checkpoint], key=lambda x: x.name)
    baselines = sorted([s for s in stats if not s.is_checkpoint], key=lambda x: x.name)

    # Print checkpoint table
    if checkpoints:
        rows = [
            [s.name, f"{s.peak_elo:.1f}", str(s.peak_step), f"{s.final_elo:.1f}", s.trend]
            for s in checkpoints
        ]
        print_table(
            ["Name", "Peak Elo", "Peak Step", "Final Elo", "Trend"],
            rows,
            "Checkpoints:",
        )

    # Print baseline table
    if baselines:
        rows = [
            [s.name, f"{s.initial_elo:.1f}", f"{s.final_elo:.1f}", f"{s.change:+.1f}"]
            for s in baselines
        ]
        print_table(
            ["Name", "Initial", "Final", "Change"],
            rows,
            "Baselines:",
        )

    # Summary
    print("\nSummary:")

    if checkpoints:
        best = max(checkpoints, key=lambda x: x.peak_elo)
        print(f"- Best checkpoint: {best.name} ({best.peak_elo:.1f} Elo)")

        # Find base_model for comparison
        base_model = next((s for s in baselines if s.name == "base_model"), None)
        if base_model:
            improvement = best.peak_elo - base_model.initial_elo
            print(f"- Improvement over base_model: {improvement:+.1f} Elo")

    # Baseline exploitation
    bot_loss = sum(s.change for s in baselines if s.change < 0)
    if bot_loss < 0:
        print(f"- Baseline exploitation: {abs(bot_loss):.1f} total Elo lost by bots")

    # Learning signal
    if checkpoints:
        declining = sum(1 for s in checkpoints if s.trend == "Declining")
        total = len(checkpoints)
        if declining > total / 2:
            print("- Learning signal: POSITIVE (older checkpoints declining)")
        elif declining > 0:
            print(f"- Learning signal: MODERATE ({declining}/{total} checkpoints declining)")
        else:
            print("- Learning signal: WEAK (no declining checkpoints)")


if __name__ == "__main__":
    main()
