# src/utils/config.py
"""
Centralized configuration for Diplomacy GRPO experiments.

This module provides:
- ExperimentConfig: Pydantic model for all training/experiment settings
- EvalConfig: Pydantic model for evaluation settings
- add_config_args: Helper to auto-generate argparse arguments from Pydantic models
- config_from_args: Helper to create config from parsed arguments
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Any, Literal, get_args, get_origin

from pydantic import BaseModel, Field, model_validator

if TYPE_CHECKING:
    pass

ProfilingMode = Literal["rollout", "trainer", "e2e"]


class ExperimentConfig(BaseModel):
    """
    Global configuration for a GRPO training run.
    Passed between Trainer, Inference, and Rollout workers.

    All fields have sensible defaults and can be overridden via CLI arguments.
    Use `add_config_args()` to auto-generate argparse arguments from this model.
    """

    # =========================================================================
    # Experiment Metadata
    # =========================================================================
    run_name: str = Field(
        default="diplomacy-grpo-v1",
        description="Name for this training run (used in WandB and checkpoints)",
    )
    seed: int = Field(default=42, description="Random seed for reproducibility")
    experiment_tag: str | None = Field(
        default=None,
        description="Tag for grouping related runs in WandB (e.g., 'power-laws')",
    )
    wandb_project: str = Field(default="diplomacy-grpo", description="WandB project name")

    # =========================================================================
    # Model Settings
    # =========================================================================
    base_model_id: str = Field(
        default="Qwen/Qwen2.5-7B-Instruct",
        description="HuggingFace model ID for base model",
    )
    lora_rank: int = Field(default=16, description="LoRA adapter rank")

    # =========================================================================
    # Rollout/Environment Settings
    # =========================================================================
    rollout_horizon_years: int = Field(
        default=4, description="Default/short rollout horizon in game years"
    )
    rollout_long_horizon_years: int = Field(
        default=6, description="Long horizon for a subset of rollouts"
    )
    rollout_long_horizon_chance: float = Field(
        default=0.2,
        description="Probability of using long horizon (0.0-1.0). 0.2 = 20% long, 80% short",
    )
    rollout_visualize_chance: float = Field(
        default=0.0,
        description="Probability of generating visualization for a rollout (0.0-1.0)",
    )
    trajectory_sample_rate: float = Field(
        default=0.01,
        description="Fraction of trajectories to log to Weave for debugging (0.01 = 1%)",
    )
    rollout_no_warmup_chance: float = Field(
        default=0.2,
        description="Probability of skipping warmup phase in rollout (0.0-1.0)",
    )

    # =========================================================================
    # Reward / Scoring Settings
    # =========================================================================
    win_bonus: float = Field(
        default=50.0,
        description=(
            "Bonus reward for the sole leader when they have >= winner_threshold_sc supply centers. "
            "Creates pressure to WIN outright, breaking cooperative stalemate equilibria. "
            "Note: Scaled by final_reward_weight (0.2), so 50 * 0.2 = 10 effective bonus."
        ),
    )
    winner_threshold_sc: int = Field(
        default=5,
        description="Minimum supply centers required to be eligible for win bonus",
    )
    step_reward_weight: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description=(
            "Weight for per-step reward shaping (board state delta). "
            "Higher values emphasize immediate consequences of each decision."
        ),
    )
    final_reward_weight: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description=(
            "Weight for final game outcome in reward. "
            "Higher values emphasize winning the overall game vs immediate gains."
        ),
    )
    step_dislodgment_weight: float = Field(
        default=0.5,
        ge=0.0,
        description="Weight for dislodgment signals in step scoring (+/- per dislodged unit)",
    )
    step_territory_weight: float = Field(
        default=0.2,
        ge=0.0,
        description="Weight for territory expansion in step scoring (+ per new province)",
    )
    step_threat_weight: float = Field(
        default=0.3,
        ge=0.0,
        description="Weight for SC threat penalty in step scoring (- per threatened SC)",
    )
    step_forward_weight: float = Field(
        default=0.1,
        ge=0.0,
        description="Weight for forward unit positioning in step scoring (+ per unit outside home)",
    )

    # =========================================================================
    # League Training Settings
    # =========================================================================
    # NOTE: League training collects trajectories only for the "hero" power,
    # reducing trajectories per rollout by 7x compared to self-play.
    #
    # Recommended compensations:
    #   - Increase num_groups_per_step from 8 to 48-56 (6-7x more rollouts)
    #   - OR lower learning_rate from 1e-5 to 2e-6 (5x lower)
    #   - OR increase samples_per_group from 8 to 24 (3x more forks)
    # =========================================================================
    league_training: bool = Field(
        default=True,
        description="Enable league training with PFSP opponent sampling",
    )
    league_registry_path: str | None = Field(
        default=None,
        description="Path to league registry JSON file. Defaults to /data/league_{run_name}.json",
    )
    league_inherit_from: str | None = Field(
        default=None,
        description="Run name to inherit league opponents from (for curriculum). Copies checkpoints as opponents.",
    )
    checkpoint_every_n_steps: int = Field(
        default=10,
        description="Save checkpoint to league every N steps (for recent curriculum)",
    )
    elo_eval_every_n_steps: int = Field(
        default=50,
        description="Run async Elo evaluation every N steps for monitoring (0 to disable). "
        "Note: Primary Elo updates now come from rollouts, not evaluation.",
    )
    elo_eval_games_per_opponent: int = Field(
        default=2,
        description="Games per gatekeeper during Elo evaluation",
    )
    rollout_elo_k_factor: float = Field(
        default=16.0,
        description="K-factor for Elo updates from rollouts. Lower = more stable ratings. "
        "Default 16 (half of standard 32) since updates happen every step.",
    )
    pfsp_self_play_weight: float = Field(
        default=0.30,
        description="PFSP: Weight for self-play (current policy)",
    )
    pfsp_peer_weight: float = Field(
        default=0.30,
        description="PFSP: Weight for peer opponents (similar Elo)",
    )
    pfsp_exploitable_weight: float = Field(
        default=0.35,
        description="PFSP: Weight for exploitable opponents (weaker)",
    )
    pfsp_baseline_weight: float = Field(
        default=0.05,
        description="PFSP: Weight for baseline opponents (bots)",
    )

    # =========================================================================
    # Checkpointing & Resume
    # =========================================================================
    save_state_every_n_steps: int = Field(
        default=10,
        description="Save full training state (optimizer, step) every N steps for resume capability",
    )
    resume_from_run: str | None = Field(
        default=None,
        description="Run name to resume from (e.g., 'grpo-20251206-133229'). Will load latest checkpoint.",
    )
    resume_from_step: int | None = Field(
        default=None,
        description="Specific step to resume from (defaults to latest if not specified)",
    )
    disable_auto_resume: bool = Field(
        default=False,
        description="Disable auto-resume from crash. If True, always starts fresh even if checkpoints exist.",
    )

    # =========================================================================
    # Training Loop
    # =========================================================================
    total_steps: int = Field(default=250, description="Total number of training steps")
    num_groups_per_step: int = Field(
        default=16,
        description="Number of rollout groups per step (G in GRPO). I.e., how many different game states to start from",
    )
    samples_per_group: int = Field(
        default=8,
        description="Number of trajectory samples per group (N in GRPO). I.e., how many different forked games to simulate for each starting state",
    )
    buffer_depth: int = Field(
        default=3,
        description="Number of rollout batches to keep in flight ahead of trainer (1-2 recommended after fork sync)",
    )
    max_concurrent_containers: int = Field(
        default=100,
        description=(
            "Maximum number of concurrent Modal containers for rollouts. "
            "Modal has hard limits (~250-500 depending on plan). "
            "Set this below the limit to leave headroom for inference containers."
        ),
    )

    # =========================================================================
    # Optimizer Settings
    # =========================================================================
    learning_rate: float = Field(default=5e-6, description="Learning rate for AdamW optimizer")
    max_grad_norm: float = Field(default=5.0, description="Maximum gradient norm for clipping")
    chunk_size: int = Field(
        default=8, ge=1, description="Mini-batch size for gradient accumulation (must be >= 1)"
    )

    # =========================================================================
    # KL Penalty / GRPO Stability Settings
    # =========================================================================
    kl_beta: float = Field(
        default=0.0,
        description="KL penalty coefficient. Set to 0 to disable KL regularization entirely.",
    )
    kl_beta_warmup_steps: int = Field(
        default=20,
        ge=0,
        description=(
            "Number of steps to linearly warmup KL beta from 0 to kl_beta. "
            "Allows policy to explore freely in early training before constraining."
        ),
    )
    kl_target: float | None = Field(
        default=None,
        description=(
            "Target KL divergence for adaptive KL control. If set, beta is adjusted "
            "each step to achieve this target. Typical values: 0.01-0.1. "
            "None disables adaptive KL (uses fixed/warmup beta)."
        ),
    )
    kl_horizon: int = Field(
        default=10,
        ge=1,
        description="Number of steps over which to smooth KL for adaptive control.",
    )
    kl_beta_min: float = Field(
        default=0.001,
        gt=0,
        description="Minimum KL beta when using adaptive control.",
    )
    kl_beta_max: float = Field(
        default=0.5,
        gt=0,
        description="Maximum KL beta when using adaptive control.",
    )

    # =========================================================================
    # Advantage Processing Settings
    # =========================================================================
    advantage_clip: float | None = Field(
        default=10,
        description=(
            "Clip advantages to [-clip, +clip] after normalization. "
            "Prevents extreme gradients from outlier rewards. "
            "Typical values: 4.0-10.0. None disables clipping."
        ),
    )
    advantage_min_std: float = Field(
        default=1e-8,
        gt=0,
        description=(
            "Minimum std for advantage normalization. Groups with lower std are skipped. "
            "Higher values (e.g., 0.1) skip more low-signal groups."
        ),
    )

    # =========================================================================
    # Inference Settings
    # =========================================================================
    max_new_tokens: int = Field(default=120, description="Maximum tokens to generate per inference")
    temperature: float = Field(default=0.8, description="Sampling temperature for generation")
    compact_prompts: bool = Field(
        default=True, description="Use compact prompt format (reduces token count)"
    )
    # Leave this off for now - it seems to tank extraction rates. TODO: investigate.
    prefix_cache_optimized: bool = Field(
        default=True,
        description="Optimize prompt structure for vLLM prefix caching",
    )
    show_valid_moves: bool = Field(
        default=False,
        description=(
            "Include valid moves list in prompts. When False, prompts only show unit "
            "positions and rely on the logits processor for move validity. "
            "Dramatically reduces tokens late-game (~94% savings with 10+ units) "
            "but may affect strategic decision quality. A/B test recommended."
        ),
    )
    show_board_context: bool = Field(
        default=True,
        description=(
            "Include board context (supply centers, opponent positions, power rankings) when "
            "show_valid_moves=False. Provides strategic awareness for decision making."
        ),
    )
    show_map_windows: bool = Field(
        default=True,
        description=(
            "Include compact per-unit map windows (adjacent tiles + nearby threats) when "
            "show_valid_moves=False. Adds a few tokens per unit to provide strategic context."
        ),
    )
    show_action_counts: bool = Field(
        default=True,
        description=(
            "Show action count and types per unit (e.g., '15 moves | move,support') when "
            "show_valid_moves=False. Helps model understand action space size without listing all moves. "
            "Adds ~10-15 tokens per unit."
        ),
    )
    compute_ref_logprobs_in_rollout: bool = Field(
        default=False,
        description=(
            "Compute reference (base model) logprobs during rollouts. "
            "If True: Rollouts slower, training faster (no ref forward pass). "
            "If False: Rollouts faster, training does ref forward pass. "
            "Default False is optimal when rollouts are the bottleneck (trainer data-starved)."
        ),
    )

    # =========================================================================
    # Profiling / Instrumentation
    # =========================================================================
    profiling_mode: ProfilingMode | None = Field(
        default=None,
        description="Enable profiling: 'rollout', 'trainer', or 'e2e'",
    )
    profile_run_name: str | None = Field(
        default=None, description="Custom name for profiling output files"
    )
    profiling_trace_steps: int = Field(
        default=3, description="Number of steps to capture in profiler trace"
    )

    # =========================================================================
    # Computed Properties
    # =========================================================================
    @property
    def batch_size(self) -> int:
        """Total batch size per training step."""
        return self.num_groups_per_step * self.samples_per_group

    @property
    def expected_horizon_years(self) -> float:
        """Expected rollout horizon accounting for variable horizons."""
        return (
            self.rollout_horizon_years * (1 - self.rollout_long_horizon_chance)
            + self.rollout_long_horizon_years * self.rollout_long_horizon_chance
        )

    @property
    def simulated_years_per_step(self) -> int:
        """Calculate expected simulated years per training step."""
        return int(self.num_groups_per_step * self.samples_per_group * self.expected_horizon_years)

    @property
    def total_simulated_years(self) -> int:
        """Calculate total simulated years for the full training run."""
        return self.simulated_years_per_step * self.total_steps

    @model_validator(mode="after")
    def validate_pfsp_weights(self) -> ExperimentConfig:
        """Validate that PFSP weights sum to 1.0."""
        total = (
            self.pfsp_self_play_weight
            + self.pfsp_peer_weight
            + self.pfsp_exploitable_weight
            + self.pfsp_baseline_weight
        )
        if not (0.99 <= total <= 1.01):  # Allow small floating point tolerance
            raise ValueError(
                f"PFSP weights must sum to 1.0, got {total:.4f}. "
                f"(self_play={self.pfsp_self_play_weight}, peer={self.pfsp_peer_weight}, "
                f"exploitable={self.pfsp_exploitable_weight}, baseline={self.pfsp_baseline_weight})"
            )
        return self


class EvalConfig(BaseModel):
    """
    Configuration for checkpoint evaluation.

    Separate from ExperimentConfig to keep concerns clear.
    """

    # =========================================================================
    # Checkpoint Settings
    # =========================================================================
    checkpoint_path: str = Field(..., description="Path to checkpoint relative to /data/models")
    base_model_id: str = Field(
        default="Qwen/Qwen2.5-7B-Instruct",
        description="HuggingFace model ID for base model",
    )

    # =========================================================================
    # Evaluation Settings
    # =========================================================================
    opponents: list[str] = Field(
        default=["random", "chaos"],
        description="Opponent types to evaluate against",
    )
    games_per_opponent: int = Field(default=10, description="Number of games per opponent type")
    max_years: int = Field(default=10, description="Maximum game length in years")
    eval_powers: list[str] = Field(
        default=["FRANCE"],
        description="Which powers use the checkpoint",
    )

    # =========================================================================
    # Visualization Settings
    # =========================================================================
    visualize: bool = Field(default=True, description="Generate HTML visualizations of games")
    visualize_sample_rate: float = Field(
        default=0.3, description="Fraction of games to visualize (0.0-1.0)"
    )

    # =========================================================================
    # Logging Settings
    # =========================================================================
    log_to_wandb: bool = Field(default=True, description="Log results to WandB")
    wandb_run_name: str | None = Field(default=None, description="Custom WandB run name")
    wandb_project: str = Field(default="diplomacy-grpo", description="WandB project name")


# =============================================================================
# CLI Argument Helpers
# =============================================================================


def _get_field_type_name(field_type: Any) -> str:
    """Get a human-readable name for a field type."""
    origin = get_origin(field_type)
    if origin is Literal:
        choices = get_args(field_type)
        return f"one of {choices}"
    if origin is list:
        inner = get_args(field_type)
        if inner:
            return f"list of {inner[0].__name__}"
        return "list"
    if hasattr(field_type, "__name__"):
        return field_type.__name__
    return str(field_type)


def add_config_args(
    parser: argparse.ArgumentParser,
    config_class: type[BaseModel],
    exclude: set[str] | None = None,
    prefix: str = "",
) -> None:
    """
    Auto-generate argparse arguments from a Pydantic model.

    This ensures CLI arguments stay in sync with the config model.
    Arguments are named with dashes (e.g., --learning-rate) but stored
    with underscores in the namespace to match Pydantic field names.

    Args:
        parser: ArgumentParser to add arguments to
        config_class: Pydantic model class to generate args from
        exclude: Set of field names to skip (e.g., computed properties)
        prefix: Optional prefix for argument names

    Example:
        >>> parser = argparse.ArgumentParser()
        >>> add_config_args(parser, ExperimentConfig, exclude={"batch_size"})
        >>> args = parser.parse_args(["--total-steps", "100"])
        >>> cfg = config_from_args(args, ExperimentConfig)
    """
    exclude = exclude or set()

    for field_name, field_info in config_class.model_fields.items():
        if field_name in exclude:
            continue

        # Convert underscore to dash for CLI (e.g., learning_rate -> learning-rate)
        arg_name = f"--{prefix}{field_name.replace('_', '-')}"

        # Get field type and default
        field_type = field_info.annotation
        default = field_info.default
        description = field_info.description or ""

        # Handle Optional types (Union[X, None])
        origin = get_origin(field_type)
        is_optional = origin is type(None) or (
            hasattr(origin, "__origin__") and type(None) in get_args(field_type)
        )

        # Extract inner type from Optional
        if origin is type(str | None) or (is_optional and origin is not None):
            args = get_args(field_type)
            inner_types = [t for t in args if t is not type(None)]
            if inner_types:
                field_type = inner_types[0]
                origin = get_origin(field_type)

        # Build argument kwargs
        kwargs: dict[str, Any] = {
            "help": f"{description} (default: {default})",
            "dest": field_name,  # Store with underscore name
        }

        # Handle different types
        if field_type is bool:
            # Booleans get --flag and --no-flag
            if default is True:
                parser.add_argument(
                    f"--no-{prefix}{field_name.replace('_', '-')}",
                    action="store_false",
                    dest=field_name,
                    help=f"Disable: {description}",
                )
            else:
                parser.add_argument(
                    arg_name,
                    action="store_true",
                    dest=field_name,
                    help=description,
                )
            continue

        elif origin is Literal:
            # Literal types become choices
            choices = get_args(field_type)
            kwargs["choices"] = choices
            kwargs["type"] = str
            kwargs["default"] = default

        elif origin is list:
            # List types use nargs
            inner_types = get_args(field_type)
            inner_type = inner_types[0] if inner_types else str
            kwargs["nargs"] = "+"
            kwargs["type"] = inner_type
            kwargs["default"] = default

        elif field_type in (int, float, str):
            kwargs["type"] = field_type
            kwargs["default"] = default

        else:
            # Fallback for complex types - treat as string
            kwargs["type"] = str
            kwargs["default"] = str(default) if default is not None else None

        parser.add_argument(arg_name, **kwargs)


def config_from_args(
    args: argparse.Namespace,
    config_class: type[BaseModel],
    extra_values: dict[str, Any] | None = None,
) -> BaseModel:
    """
    Create a Pydantic config from parsed argparse namespace.

    Args:
        args: Parsed argument namespace
        config_class: Pydantic model class to instantiate
        extra_values: Additional values to include (override args)

    Returns:
        Instantiated config model

    Example:
        >>> args = parser.parse_args(["--total-steps", "100"])
        >>> cfg = config_from_args(args, ExperimentConfig)
        >>> print(cfg.total_steps)
        100
    """
    # Convert namespace to dict, filtering out None values for optional fields
    values = {
        k: v for k, v in vars(args).items() if v is not None and k in config_class.model_fields
    }

    # Apply extra values
    if extra_values:
        values.update(extra_values)

    return config_class(**values)


def create_experiment_parser(
    description: str = "Run Diplomacy GRPO experiment",
    exclude: set[str] | None = None,
) -> argparse.ArgumentParser:
    """
    Create a fully-configured argument parser for experiments.

    Convenience function that creates a parser with all ExperimentConfig fields.

    Args:
        description: Parser description
        exclude: Fields to exclude from CLI

    Returns:
        Configured ArgumentParser

    Example:
        >>> parser = create_experiment_parser("My training script")
        >>> parser.add_argument("--my-custom-arg", type=str)
        >>> args = parser.parse_args()
        >>> cfg = config_from_args(args, ExperimentConfig)
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_config_args(parser, ExperimentConfig, exclude=exclude)
    return parser


def create_eval_parser(
    description: str = "Evaluate Diplomacy GRPO checkpoint",
    exclude: set[str] | None = None,
) -> argparse.ArgumentParser:
    """
    Create a fully-configured argument parser for evaluation.

    Args:
        description: Parser description
        exclude: Fields to exclude from CLI

    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_config_args(parser, EvalConfig, exclude=exclude)
    return parser
