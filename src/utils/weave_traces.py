"""
Weave trajectory tracing for Diplomacy GRPO.

This module provides structured logging of sample trajectories to WandB Weave
for debugging, analysis, and quality assessment.

Usage:
    from src.utils.weave_traces import init_weave, log_trajectory, TrajectoryTrace

    # Initialize once per run
    init_weave("diplomacy-grpo")

    # Log individual trajectories (sampled based on config.trajectory_sample_rate)
    if random.random() < cfg.trajectory_sample_rate:
        trace = TrajectoryTrace(
            game_id=game_id,
            power=power,
            year=year,
            phase=phase,
            prompt=prompt,
            completion=completion,
            reward=reward,
            orders_expected=expected_count,
            orders_extracted=len(orders),
            extraction_status="full",
        )
        log_trajectory(trace)

Weave Features:
    - Trace Explorer: Visual UI to browse all logged trajectories
    - Filtering: Filter by power, reward range, extraction status
    - Versioning: Automatic versioning of trace schemas
    - API Access: Query traces programmatically via Weave API
"""

from __future__ import annotations

import weave
from pydantic import BaseModel, Field

# Track initialization state
_weave_initialized: bool = False
_weave_project: str | None = None


def init_weave(project: str = "diplomacy-grpo") -> None:
    """
    Initialize Weave for trajectory logging.

    Should be called once at the start of a training/rollout run.
    Safe to call multiple times - will only initialize once.

    Requires WANDB_API_KEY environment variable to be set.

    Args:
        project: Weave project name (appears in WandB UI)
    """
    global _weave_initialized, _weave_project

    if _weave_initialized:
        return

    import os

    # Check for WandB credentials
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError(
            "WANDB_API_KEY not set. Weave requires WandB authentication. "
            "Ensure wandb-secret is attached to the Modal function."
        )

    weave.init(project)
    _weave_initialized = True
    _weave_project = project
    print(f"✅ Weave initialized for project: {project}")


def is_weave_initialized() -> bool:
    """Check if Weave has been initialized."""
    return _weave_initialized


class TrajectoryTrace(BaseModel):
    """
    Trajectory data model for Weave tracing.

    Captures all relevant information about a single trajectory for
    debugging and quality analysis.
    """

    # Identification
    game_id: str = Field(description="Unique game identifier")
    rollout_id: str = Field(default="", description="Rollout batch identifier")
    run_name: str = Field(default="", description="Training run name")

    # Game state
    power: str = Field(description="Power playing (e.g., FRANCE, ENGLAND)")
    year: int = Field(description="Game year (1901, 1902, etc.)")
    phase: str = Field(description="Game phase (SPRING, FALL)")

    # LLM interaction
    prompt: str = Field(description="Full prompt sent to LLM")
    completion: str = Field(description="LLM response/completion")

    # Reward and scoring
    reward: float = Field(description="Final reward for this trajectory")
    advantage: float | None = Field(
        default=None, description="Computed advantage (after normalization)"
    )

    # Order extraction
    orders_expected: int = Field(description="Number of orders expected (units owned)")
    orders_extracted: int = Field(description="Number of orders successfully extracted")
    extraction_status: str = Field(description="Extraction result: 'full', 'partial', or 'empty'")
    extracted_orders: list[str] = Field(
        default_factory=list, description="List of extracted order strings"
    )

    # Metadata
    horizon_type: str = Field(default="short", description="Rollout horizon type (short/long)")
    opponent_adapters: list[str] = Field(
        default_factory=list, description="LoRA adapters used by opponents"
    )
    hero_adapter: str | None = Field(default=None, description="LoRA adapter used by hero")

    # Token counts (for cost/efficiency analysis)
    prompt_tokens: int = Field(default=0, description="Number of tokens in prompt")
    completion_tokens: int = Field(default=0, description="Number of tokens in completion")


@weave.op()
def log_trajectory(trajectory: TrajectoryTrace) -> TrajectoryTrace:
    """
    Log a trajectory to Weave.

    This function is decorated with @weave.op() which automatically:
    - Records the call to Weave
    - Captures inputs and outputs
    - Enables filtering and querying via Weave UI

    Args:
        trajectory: TrajectoryTrace object with all trajectory data

    Returns:
        The input trajectory (for chaining)
    """
    return trajectory


@weave.op()
def log_trajectories_batch(trajectories: list[TrajectoryTrace]) -> list[TrajectoryTrace]:
    """
    Log multiple trajectories to Weave in a single operation.

    More efficient than logging individually when you have multiple
    trajectories to log at once.

    Args:
        trajectories: List of TrajectoryTrace objects

    Returns:
        The input trajectories (for chaining)
    """
    return trajectories


class TrajectoryBatchSummary(BaseModel):
    """Summary statistics for a batch of trajectories."""

    batch_size: int = Field(description="Number of trajectories in batch")
    run_name: str = Field(description="Training run name")
    step: int = Field(default=0, description="Training step number")

    # Extraction stats
    full_extraction_count: int = Field(default=0, description="Trajectories with full extraction")
    partial_extraction_count: int = Field(
        default=0, description="Trajectories with partial extraction"
    )
    empty_extraction_count: int = Field(
        default=0, description="Trajectories with no orders extracted"
    )
    extraction_rate: float = Field(default=0.0, description="Rate of successful extractions")

    # Reward stats
    reward_mean: float = Field(default=0.0, description="Mean reward in batch")
    reward_std: float = Field(default=0.0, description="Std dev of rewards")
    reward_min: float = Field(default=0.0, description="Minimum reward")
    reward_max: float = Field(default=0.0, description="Maximum reward")

    # Power distribution
    power_counts: dict[str, int] = Field(
        default_factory=dict, description="Count of trajectories per power"
    )


@weave.op()
def log_batch_summary(summary: TrajectoryBatchSummary) -> TrajectoryBatchSummary:
    """
    Log a batch summary to Weave.

    Useful for tracking high-level stats without logging every trajectory.

    Args:
        summary: TrajectoryBatchSummary with aggregate statistics

    Returns:
        The input summary (for chaining)
    """
    return summary


class ISOutlierTrace(BaseModel):
    """
    Importance Sampling outlier for debugging vLLM-HuggingFace mismatch.

    These samples have extreme IS ratios (> threshold), which cause training
    instability when not properly capped/masked.
    """

    # Identification
    run_name: str = Field(default="", description="Training run name")
    step: int = Field(default=0, description="Training step when outlier was detected")

    # The actual completion data
    prompt: str = Field(description="Full prompt (truncated to 500 chars)")
    completion: str = Field(description="LLM completion (truncated to 500 chars)")

    # IS ratio diagnostics
    is_ratio: float = Field(description="Raw IS ratio (π_new / π_old) before capping")
    logprob_diff: float = Field(
        description="Sequence-level log_prob difference (HF - vLLM) in nats"
    )
    completion_logprob: float = Field(description="HuggingFace computed log_prob")
    rollout_logprob: float = Field(description="vLLM computed log_prob (from rollout)")
    num_tokens: int = Field(description="Number of completion tokens")

    # Context
    group_id: str = Field(description="GRPO group ID")
    reward: float = Field(description="Reward for this trajectory")
    advantage: float = Field(description="Computed advantage")


@weave.op()
def log_is_outlier(outlier: ISOutlierTrace) -> ISOutlierTrace:
    """
    Log an IS outlier to Weave for debugging.

    IS outliers are samples where vLLM and HuggingFace compute significantly
    different log probabilities, causing extreme importance sampling ratios.

    Args:
        outlier: ISOutlierTrace with outlier details

    Returns:
        The input outlier (for chaining)
    """
    return outlier


@weave.op()
def log_is_outliers_batch(outliers: list[ISOutlierTrace]) -> list[ISOutlierTrace]:
    """
    Log multiple IS outliers to Weave in a single operation.

    Args:
        outliers: List of ISOutlierTrace objects

    Returns:
        The input outliers (for chaining)
    """
    return outliers


def create_trajectory_from_rollout_data(
    data: dict,
    game_id: str,
    rollout_id: str,
    run_name: str,
    power: str,
    year: int,
    phase: str,
    reward: float,
    expected_orders: int,
    extracted_orders: list[str],
    extraction_status: str,
    horizon_type: str = "short",
    hero_adapter: str | None = None,
    opponent_adapters: list[str] | None = None,
) -> TrajectoryTrace:
    """
    Create a TrajectoryTrace from rollout data.

    Convenience function to construct TrajectoryTrace from the data
    available during rollout processing.

    Args:
        data: Dict containing 'prompt', 'completion', and optionally token IDs
        game_id: Unique game identifier
        rollout_id: Rollout batch identifier
        run_name: Training run name
        power: Power being played
        year: Game year
        phase: Game phase
        reward: Final reward
        expected_orders: Number of orders expected
        extracted_orders: List of extracted order strings
        extraction_status: "full", "partial", or "empty"
        horizon_type: Rollout horizon type
        hero_adapter: LoRA adapter for hero (if any)
        opponent_adapters: List of opponent LoRA adapters

    Returns:
        Populated TrajectoryTrace object
    """
    prompt = data.get("prompt", "")
    completion = data.get("completion", "")
    prompt_tokens = len(data.get("prompt_token_ids", []))
    completion_tokens = len(data.get("completion_token_ids", []))

    return TrajectoryTrace(
        game_id=game_id,
        rollout_id=rollout_id,
        run_name=run_name,
        power=power,
        year=year,
        phase=phase,
        prompt=prompt,
        completion=completion,
        reward=reward,
        orders_expected=expected_orders,
        orders_extracted=len(extracted_orders),
        extraction_status=extraction_status,
        extracted_orders=extracted_orders,
        horizon_type=horizon_type,
        hero_adapter=hero_adapter,
        opponent_adapters=opponent_adapters or [],
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
