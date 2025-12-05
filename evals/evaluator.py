"""
Core evaluation logic for Diplomacy GRPO.

This module provides the infrastructure to evaluate trained checkpoints
against various opponents and generate metrics + visualizations.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class OpponentType(str, Enum):
    """Types of opponents for evaluation."""

    RANDOM = "random"
    CHAOS = "chaos"
    CHECKPOINT = "checkpoint"  # Another trained checkpoint
    SELF = "self"  # Self-play (all powers use the same checkpoint)


@dataclass
class EvalConfig:
    """Configuration for a single evaluation game."""

    # Checkpoint to evaluate (path relative to /data/models)
    checkpoint_path: str

    # Which power(s) the checkpoint plays (None = all powers)
    eval_powers: list[str] | None = None

    # Opponent configuration
    opponent_type: OpponentType = OpponentType.RANDOM
    opponent_checkpoint: str | None = None  # For CHECKPOINT opponent type

    # Game settings
    max_years: int = 10  # Max game length
    num_games: int = 10  # Number of games to play

    # Visualization
    visualize: bool = True
    visualize_sample_rate: float = 0.2  # Fraction of games to visualize

    # Model settings
    model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    compact_prompts: bool = True


@dataclass
class PowerResult:
    """Results for a single power in a game."""

    power_name: str
    is_checkpoint: bool  # Whether this power used the checkpoint
    final_centers: int
    final_units: int
    survived: bool
    won: bool  # 18+ centers or last standing
    elimination_year: int | None = None


@dataclass
class GameResult:
    """Results from a single evaluation game."""

    game_id: str
    num_years: int
    power_results: list[PowerResult]
    visualization_path: str | None = None

    # Aggregated metrics
    @property
    def checkpoint_powers(self) -> list[PowerResult]:
        return [p for p in self.power_results if p.is_checkpoint]

    @property
    def opponent_powers(self) -> list[PowerResult]:
        return [p for p in self.power_results if not p.is_checkpoint]

    @property
    def checkpoint_win_count(self) -> int:
        return sum(1 for p in self.checkpoint_powers if p.won)

    @property
    def checkpoint_survival_count(self) -> int:
        return sum(1 for p in self.checkpoint_powers if p.survived)

    @property
    def checkpoint_avg_centers(self) -> float:
        centers = [p.final_centers for p in self.checkpoint_powers]
        return sum(centers) / len(centers) if centers else 0


@dataclass
class EvalResult:
    """Aggregated results from an evaluation run."""

    config: EvalConfig
    game_results: list[GameResult]
    visualization_paths: list[str] = field(default_factory=list)

    # Timing
    total_duration_s: float = 0.0

    @property
    def num_games(self) -> int:
        return len(self.game_results)

    @property
    def win_rate(self) -> float:
        """Win rate of checkpoint powers."""
        total_wins = sum(g.checkpoint_win_count for g in self.game_results)
        total_powers = sum(len(g.checkpoint_powers) for g in self.game_results)
        return total_wins / total_powers if total_powers > 0 else 0

    @property
    def survival_rate(self) -> float:
        """Survival rate of checkpoint powers."""
        total_survived = sum(g.checkpoint_survival_count for g in self.game_results)
        total_powers = sum(len(g.checkpoint_powers) for g in self.game_results)
        return total_survived / total_powers if total_powers > 0 else 0

    @property
    def avg_centers(self) -> float:
        """Average final supply center count for checkpoint powers."""
        all_centers = []
        for g in self.game_results:
            all_centers.extend(p.final_centers for p in g.checkpoint_powers)
        return sum(all_centers) / len(all_centers) if all_centers else 0

    @property
    def avg_game_length(self) -> float:
        """Average game length in years."""
        lengths = [g.num_years for g in self.game_results]
        return sum(lengths) / len(lengths) if lengths else 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for serialization."""
        return {
            "config": {
                "checkpoint_path": self.config.checkpoint_path,
                "opponent_type": self.config.opponent_type.value,
                "opponent_checkpoint": self.config.opponent_checkpoint,
                "num_games": self.config.num_games,
                "max_years": self.config.max_years,
            },
            "metrics": {
                "win_rate": self.win_rate,
                "survival_rate": self.survival_rate,
                "avg_centers": self.avg_centers,
                "avg_game_length": self.avg_game_length,
                "num_games_completed": self.num_games,
            },
            "visualization_paths": self.visualization_paths,
            "total_duration_s": self.total_duration_s,
        }

    def summary(self) -> str:
        """Human-readable summary."""
        return f"""
Evaluation Results: {self.config.checkpoint_path}
{"=" * 50}
Opponent: {self.config.opponent_type.value}
Games: {self.num_games}

Metrics:
  Win Rate:      {self.win_rate:.1%}
  Survival Rate: {self.survival_rate:.1%}
  Avg Centers:   {self.avg_centers:.1f}
  Avg Game Len:  {self.avg_game_length:.1f} years

Duration: {self.total_duration_s:.1f}s
{"=" * 50}
"""


async def run_eval_game(
    config: EvalConfig,
    game_index: int,
    inference_engine: Any,  # Modal InferenceEngine class
) -> GameResult:
    """
    Run a single evaluation game.

    Args:
        config: Evaluation configuration
        game_index: Index of this game (for deterministic seeding)
        inference_engine: Modal InferenceEngine instance for checkpoint inference

    Returns:
        GameResult with metrics and optional visualization
    """
    import random

    from src.agents import LLMAgent, PromptConfig
    from src.agents.baselines import ChaosBot, RandomBot
    from src.engine.wrapper import DiplomacyWrapper
    from src.utils.parsing import extract_orders
    from src.utils.vis import GameVisualizer

    # Seed for reproducibility
    random.seed(42 + game_index)

    # Initialize game
    game = DiplomacyWrapper(horizon=config.max_years * 2)  # 2 phases per year
    game_id = game.game.game_id

    # Determine which powers use checkpoint vs opponent
    all_powers = list(game.game.powers.keys())
    if config.eval_powers:
        checkpoint_powers = set(config.eval_powers)
    else:
        # Default: checkpoint plays FRANCE (for consistency)
        checkpoint_powers = {"FRANCE"}

    # Initialize opponent agent
    if config.opponent_type == OpponentType.RANDOM:
        opponent_agent = RandomBot()
    elif config.opponent_type == OpponentType.CHAOS:
        opponent_agent = ChaosBot()
    else:
        # For CHECKPOINT/SELF types, we use inference (handled separately)
        opponent_agent = RandomBot()  # Fallback

    # Initialize LLM agent for prompt building
    prompt_config = PromptConfig(compact_mode=config.compact_prompts)
    llm_agent = LLMAgent(config=prompt_config)

    # Visualization
    should_visualize = config.visualize and (random.random() < config.visualize_sample_rate)
    vis = GameVisualizer() if should_visualize else None
    if vis:
        vis.capture_turn(game.game, "Game Start")

    # Track eliminations
    elimination_years: dict[str, int] = {}

    # Game loop
    while not game.is_done() and game.get_year() <= 1900 + config.max_years:
        current_year = game.get_year()
        all_orders = []

        # Get orders for checkpoint powers (batched inference)
        checkpoint_prompts = []
        checkpoint_valid_moves = []
        checkpoint_power_names = []

        for power_name in all_powers:
            power = game.game.powers[power_name]

            # Skip eliminated powers
            if len(power.units) == 0 and len(power.centers) == 0:
                if power_name not in elimination_years:
                    elimination_years[power_name] = current_year
                continue

            if power_name in checkpoint_powers:
                # Use checkpoint for this power
                prompt, valid_moves = llm_agent.build_prompt(game, power_name)
                checkpoint_prompts.append(prompt)
                checkpoint_valid_moves.append(valid_moves)
                checkpoint_power_names.append(power_name)
            else:
                # Use opponent agent
                orders = opponent_agent.get_orders(game, power_name)
                all_orders.extend(orders)

        # Batch inference for checkpoint powers
        if checkpoint_prompts:
            responses = await inference_engine.generate.remote.aio(
                prompts=checkpoint_prompts,
                valid_moves=checkpoint_valid_moves,
                lora_name=config.checkpoint_path,
            )

            for response, valid_moves in zip(responses, checkpoint_valid_moves, strict=True):
                orders = extract_orders(response)
                # Fallback if no orders extracted
                if not orders:
                    # Use first valid move for each unit
                    orders = [moves[0] for moves in valid_moves.values() if moves]
                all_orders.extend(orders)

        # Execute orders
        game.step(all_orders)

        # Capture visualization
        if vis:
            phase = game.get_current_phase()
            vis.capture_turn(game.game, f"Year {current_year} - {phase}")

    # Collect results
    power_results = []
    for power_name in all_powers:
        power = game.game.powers[power_name]
        n_centers = len(power.centers)
        n_units = len(power.units)
        survived = n_centers > 0 or n_units > 0
        won = n_centers >= 18  # Standard victory condition

        power_results.append(
            PowerResult(
                power_name=power_name,
                is_checkpoint=power_name in checkpoint_powers,
                final_centers=n_centers,
                final_units=n_units,
                survived=survived,
                won=won,
                elimination_year=elimination_years.get(power_name),
            )
        )

    # Save visualization if enabled
    vis_path = None
    if vis:
        vis_path = f"/data/evals/{game_id}.html"
        vis.save_html(vis_path)

    return GameResult(
        game_id=game_id,
        num_years=game.get_year() - 1901,
        power_results=power_results,
        visualization_path=vis_path,
    )
