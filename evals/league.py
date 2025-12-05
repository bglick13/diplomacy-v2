"""
League evaluation for comparing checkpoints against multiple opponents.

This module provides:
- Running a checkpoint against a battery of opponents
- Comparing multiple checkpoints head-to-head
- ELO-style rating computation (optional)
- Summary statistics and WandB logging
"""

from dataclasses import dataclass, field
from typing import Any

from evals.evaluator import EvalConfig, EvalResult, OpponentType, run_eval_game


@dataclass
class LeagueConfig:
    """Configuration for a league evaluation."""

    # Checkpoint(s) to evaluate
    checkpoints: list[str]  # Paths relative to /data/models

    # Opponents to play against
    opponents: list[OpponentType] = field(
        default_factory=lambda: [OpponentType.RANDOM, OpponentType.CHAOS]
    )

    # Optional: other checkpoints to compare against
    opponent_checkpoints: list[str] = field(default_factory=list)

    # Game settings
    games_per_matchup: int = 5
    max_years: int = 10

    # Powers the checkpoint plays (None = FRANCE only for consistency)
    eval_powers: list[str] | None = None

    # Model settings
    model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    compact_prompts: bool = True

    # Visualization
    visualize: bool = True
    visualize_sample_rate: float = 0.2


@dataclass
class MatchupResult:
    """Results from a checkpoint vs opponent matchup."""

    checkpoint: str
    opponent_type: str  # "random", "chaos", or checkpoint path
    eval_result: EvalResult

    @property
    def win_rate(self) -> float:
        return self.eval_result.win_rate

    @property
    def survival_rate(self) -> float:
        return self.eval_result.survival_rate

    @property
    def avg_centers(self) -> float:
        return self.eval_result.avg_centers


@dataclass
class LeagueResult:
    """Aggregated results from a league evaluation."""

    config: LeagueConfig
    matchup_results: list[MatchupResult]
    total_duration_s: float = 0.0

    def get_checkpoint_summary(self, checkpoint: str) -> dict[str, Any]:
        """Get summary metrics for a specific checkpoint."""
        results = [m for m in self.matchup_results if m.checkpoint == checkpoint]
        if not results:
            return {}

        return {
            "checkpoint": checkpoint,
            "matchups": len(results),
            "avg_win_rate": sum(m.win_rate for m in results) / len(results),
            "avg_survival_rate": sum(m.survival_rate for m in results) / len(results),
            "avg_centers": sum(m.avg_centers for m in results) / len(results),
            "vs_random": next((m for m in results if m.opponent_type == "random"), None),
            "vs_chaos": next((m for m in results if m.opponent_type == "chaos"), None),
        }

    def to_wandb_table(self) -> list[dict[str, Any]]:
        """Convert to format suitable for WandB table logging."""
        rows = []
        for m in self.matchup_results:
            rows.append(
                {
                    "checkpoint": m.checkpoint,
                    "opponent": m.opponent_type,
                    "win_rate": m.win_rate,
                    "survival_rate": m.survival_rate,
                    "avg_centers": m.avg_centers,
                    "games_played": m.eval_result.num_games,
                    "avg_game_length": m.eval_result.avg_game_length,
                }
            )
        return rows

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "League Evaluation Results",
            "=" * 60,
            f"Checkpoints: {len(self.config.checkpoints)}",
            f"Opponents: {[o.value for o in self.config.opponents]}",
            f"Games per matchup: {self.config.games_per_matchup}",
            "",
            "Results by Checkpoint:",
            "-" * 40,
        ]

        for ckpt in self.config.checkpoints:
            summary = self.get_checkpoint_summary(ckpt)
            if summary:
                lines.append(f"\n{ckpt}:")
                lines.append(f"  Win Rate:      {summary['avg_win_rate']:.1%}")
                lines.append(f"  Survival Rate: {summary['avg_survival_rate']:.1%}")
                lines.append(f"  Avg Centers:   {summary['avg_centers']:.1f}")

        lines.extend(["", "=" * 60, f"Total Duration: {self.total_duration_s:.1f}s"])

        return "\n".join(lines)


async def run_league_evaluation(
    config: LeagueConfig,
    inference_engine: Any,  # Modal InferenceEngine
    logger: Any = None,
) -> LeagueResult:
    """
    Run a full league evaluation.

    Args:
        config: League configuration
        inference_engine: Modal InferenceEngine for checkpoint inference
        logger: Optional logger for progress updates

    Returns:
        LeagueResult with all matchup results
    """
    import time

    start_time = time.time()
    matchup_results = []

    log = logger.info if logger else print

    log(f"Starting league evaluation with {len(config.checkpoints)} checkpoint(s)")

    for checkpoint in config.checkpoints:
        log(f"\n📊 Evaluating checkpoint: {checkpoint}")

        # Run against each opponent type
        for opponent_type in config.opponents:
            log(f"  vs {opponent_type.value}...")

            eval_config = EvalConfig(
                checkpoint_path=checkpoint,
                eval_powers=config.eval_powers,
                opponent_type=opponent_type,
                max_years=config.max_years,
                num_games=config.games_per_matchup,
                visualize=config.visualize,
                visualize_sample_rate=config.visualize_sample_rate,
                model_id=config.model_id,
                compact_prompts=config.compact_prompts,
            )

            # Run games
            game_results = []
            for i in range(config.games_per_matchup):
                result = await run_eval_game(eval_config, i, inference_engine)
                game_results.append(result)
                log(
                    f"    Game {i + 1}/{config.games_per_matchup}: {result.checkpoint_avg_centers:.1f} avg centers"
                )

            # Collect visualization paths
            vis_paths = [g.visualization_path for g in game_results if g.visualization_path]

            eval_result = EvalResult(
                config=eval_config,
                game_results=game_results,
                visualization_paths=vis_paths,
            )

            matchup_results.append(
                MatchupResult(
                    checkpoint=checkpoint,
                    opponent_type=opponent_type.value,
                    eval_result=eval_result,
                )
            )

            log(
                f"    Win rate: {eval_result.win_rate:.1%}, Survival: {eval_result.survival_rate:.1%}"
            )

        # Run against opponent checkpoints
        for opp_checkpoint in config.opponent_checkpoints:
            if opp_checkpoint == checkpoint:
                continue  # Skip self-play (handled separately if needed)

            log(f"  vs checkpoint: {opp_checkpoint}...")

            eval_config = EvalConfig(
                checkpoint_path=checkpoint,
                eval_powers=config.eval_powers,
                opponent_type=OpponentType.CHECKPOINT,
                opponent_checkpoint=opp_checkpoint,
                max_years=config.max_years,
                num_games=config.games_per_matchup,
                visualize=config.visualize,
                visualize_sample_rate=config.visualize_sample_rate,
                model_id=config.model_id,
                compact_prompts=config.compact_prompts,
            )

            # For checkpoint vs checkpoint, we need special handling
            # (both powers use inference - not implemented yet)
            log("    ⚠️ Checkpoint vs Checkpoint not fully implemented yet")

    total_duration = time.time() - start_time

    return LeagueResult(
        config=config,
        matchup_results=matchup_results,
        total_duration_s=total_duration,
    )


def log_to_wandb(
    result: LeagueResult,
    run_name: str | None = None,
    project: str = "diplomacy-grpo",
) -> None:
    """
    Log league evaluation results to WandB.

    Args:
        result: League evaluation results
        run_name: Optional WandB run name
        project: WandB project name
    """
    import wandb

    # Initialize WandB run
    wandb.init(
        project=project,
        name=run_name or f"eval-{result.config.checkpoints[0].split('/')[-1]}",
        tags=["evaluation", "league"],
        config={
            "checkpoints": result.config.checkpoints,
            "opponents": [o.value for o in result.config.opponents],
            "games_per_matchup": result.config.games_per_matchup,
            "max_years": result.config.max_years,
        },
    )

    # Log summary metrics
    for checkpoint in result.config.checkpoints:
        summary = result.get_checkpoint_summary(checkpoint)
        if summary:
            checkpoint_name = checkpoint.split("/")[-1]
            wandb.log(
                {
                    f"eval/{checkpoint_name}/avg_win_rate": summary["avg_win_rate"],
                    f"eval/{checkpoint_name}/avg_survival_rate": summary["avg_survival_rate"],
                    f"eval/{checkpoint_name}/avg_centers": summary["avg_centers"],
                }
            )

    # Log results table
    table = wandb.Table(
        columns=[
            "checkpoint",
            "opponent",
            "win_rate",
            "survival_rate",
            "avg_centers",
            "games_played",
        ]
    )
    for row in result.to_wandb_table():
        table.add_data(
            row["checkpoint"],
            row["opponent"],
            row["win_rate"],
            row["survival_rate"],
            row["avg_centers"],
            row["games_played"],
        )
    wandb.log({"eval/results_table": table})

    # Log HTML visualizations as artifacts
    for matchup in result.matchup_results:
        for vis_path in matchup.eval_result.visualization_paths:
            if vis_path:
                artifact_name = (
                    f"game-replay-{matchup.checkpoint.split('/')[-1]}-vs-{matchup.opponent_type}"
                )
                artifact = wandb.Artifact(artifact_name, type="visualization")
                artifact.add_file(vis_path)
                wandb.log_artifact(artifact)

                # Also log as HTML directly for inline viewing
                wandb.log(
                    {f"eval/replay/{matchup.opponent_type}": wandb.Html(open(vis_path).read())}
                )

    wandb.finish()
