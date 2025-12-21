"""
Fixed Benchmark Suite for Absolute Skill Measurement.

This module defines a static set of benchmark agents that never change,
providing consistent reference points for measuring absolute skill improvement
across training runs.

Unlike the PFSP system which creates a closed feedback loop (training only
against similar-skill peers), benchmarks provide external reference points:
- Trivial baselines (random_bot, chaos_bot) - should beat 95%+
- Stronger heuristic baselines (defensive, territorial, coordinated)
- Frozen checkpoints from previous training runs

Usage:
    from src.league.benchmarks import BENCHMARK_SUITE, BenchmarkAgent

    for benchmark in BENCHMARK_SUITE:
        results = await run_benchmark_games(challenger, benchmark)
        print(f"vs {benchmark.name}: {results.win_rate:.1%}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class BenchmarkTier(Enum):
    """Difficulty tiers for benchmark agents."""

    TRIVIAL = "trivial"  # Should beat 95%+ (random, chaos)
    HEURISTIC = "heuristic"  # Should beat 70%+ (defensive, territorial)
    CHECKPOINT = "checkpoint"  # Frozen checkpoints from training


@dataclass
class BenchmarkAgent:
    """
    A frozen benchmark agent for consistent evaluation.

    Attributes:
        name: Unique identifier for this benchmark
        path: Adapter path (None for baselines, relative path for checkpoints)
        description: Human-readable description
        tier: Difficulty tier
        expected_winrate_floor: Minimum expected win rate (used for pass/fail)
    """

    name: str
    path: str | None
    description: str
    tier: BenchmarkTier
    expected_winrate_floor: float = 0.5

    def is_baseline(self) -> bool:
        """Check if this is a baseline bot (no adapter)."""
        return self.path is None or self.path in (
            "random_bot",
            "chaos_bot",
            "defensive_bot",
            "territorial_bot",
            "coordinated_bot",
        )


# ============================================================================
# BENCHMARK SUITE DEFINITION
# ============================================================================

# Trivial baselines - should beat these 95%+ immediately
TRIVIAL_BENCHMARKS = [
    BenchmarkAgent(
        name="random_bot",
        path="random_bot",
        description="Uniform random move selection",
        tier=BenchmarkTier.TRIVIAL,
        expected_winrate_floor=0.95,
    ),
    BenchmarkAgent(
        name="chaos_bot",
        path="chaos_bot",
        description="Aggressive random (prefers moves over holds)",
        tier=BenchmarkTier.TRIVIAL,
        expected_winrate_floor=0.90,
    ),
]

# Stronger heuristic baselines - harder to beat
HEURISTIC_BENCHMARKS = [
    BenchmarkAgent(
        name="defensive_bot",
        path="defensive_bot",
        description="Prefers supports and holds (cautious play)",
        tier=BenchmarkTier.HEURISTIC,
        expected_winrate_floor=0.70,
    ),
    BenchmarkAgent(
        name="territorial_bot",
        path="territorial_bot",
        description="Prioritizes neutral center acquisition",
        tier=BenchmarkTier.HEURISTIC,
        expected_winrate_floor=0.70,
    ),
    BenchmarkAgent(
        name="coordinated_bot",
        path="coordinated_bot",
        description="Attempts to coordinate supports between units",
        tier=BenchmarkTier.HEURISTIC,
        expected_winrate_floor=0.65,
    ),
]

# Frozen checkpoints - reference points from previous training
# These paths are relative to /data/models/ on the Modal volume
CHECKPOINT_BENCHMARKS = [
    BenchmarkAgent(
        name="frozen_v10",
        path="benchmarks/frozen_v10",
        description="Early checkpoint (10 steps) - represents novice play",
        tier=BenchmarkTier.CHECKPOINT,
        expected_winrate_floor=0.70,
    ),
    BenchmarkAgent(
        name="frozen_v50",
        path="benchmarks/frozen_v50",
        description="Mid-training checkpoint (50 steps) - represents competent play",
        tier=BenchmarkTier.CHECKPOINT,
        expected_winrate_floor=0.50,
    ),
    BenchmarkAgent(
        name="frozen_v150",
        path="benchmarks/frozen_v150",
        description="Best checkpoint from grpo-20251219-143727 - target to beat",
        tier=BenchmarkTier.CHECKPOINT,
        expected_winrate_floor=0.30,
    ),
]

# Full benchmark suite
BENCHMARK_SUITE: list[BenchmarkAgent] = (
    TRIVIAL_BENCHMARKS + HEURISTIC_BENCHMARKS + CHECKPOINT_BENCHMARKS
)

# Quick benchmark suite (for frequent evaluation during training)
QUICK_BENCHMARK_SUITE: list[BenchmarkAgent] = [
    TRIVIAL_BENCHMARKS[0],  # random_bot
    HEURISTIC_BENCHMARKS[0],  # defensive_bot
    CHECKPOINT_BENCHMARKS[-1],  # frozen_v150 (best checkpoint)
]


def get_benchmark_by_name(name: str) -> BenchmarkAgent | None:
    """Get a benchmark agent by name."""
    for benchmark in BENCHMARK_SUITE:
        if benchmark.name == name:
            return benchmark
    return None


def get_benchmarks_by_tier(tier: BenchmarkTier) -> list[BenchmarkAgent]:
    """Get all benchmarks in a specific tier."""
    return [b for b in BENCHMARK_SUITE if b.tier == tier]


@dataclass
class BenchmarkResult:
    """Result from evaluating against a single benchmark."""

    benchmark_name: str
    games_played: int
    wins: int
    win_rate: float
    avg_score: float
    avg_benchmark_score: float
    meets_floor: bool


@dataclass
class BenchmarkSuiteResult:
    """Aggregate results from full benchmark suite evaluation."""

    challenger_path: str
    results: dict[str, BenchmarkResult] = field(default_factory=dict)
    total_games: int = 0
    overall_win_rate: float = 0.0
    benchmarks_passed: int = 0
    benchmarks_total: int = 0

    def compute_overall_score(self) -> float:
        """
        Compute an overall benchmark score (0-100).

        Weighted average of win rates, with higher weight for harder tiers.
        """
        if not self.results:
            return 0.0

        weights = {
            BenchmarkTier.TRIVIAL: 1.0,
            BenchmarkTier.HEURISTIC: 2.0,
            BenchmarkTier.CHECKPOINT: 3.0,
        }

        total_weight = 0.0
        weighted_sum = 0.0

        for name, result in self.results.items():
            benchmark = get_benchmark_by_name(name)
            if benchmark:
                weight = weights.get(benchmark.tier, 1.0)
                weighted_sum += result.win_rate * weight
                total_weight += weight

        return (weighted_sum / total_weight * 100) if total_weight > 0 else 0.0

    def to_wandb_dict(self) -> dict:
        """Format results for WandB logging."""
        metrics = {
            "benchmark/overall_score": self.compute_overall_score(),
            "benchmark/games_played": self.total_games,
            "benchmark/benchmarks_passed": self.benchmarks_passed,
            "benchmark/benchmarks_total": self.benchmarks_total,
        }

        for name, result in self.results.items():
            safe_name = name.replace("/", "_")
            metrics[f"benchmark/vs_{safe_name}_winrate"] = result.win_rate
            metrics[f"benchmark/vs_{safe_name}_avg_score"] = result.avg_score
            metrics[f"benchmark/vs_{safe_name}_passed"] = 1 if result.meets_floor else 0

        return metrics
