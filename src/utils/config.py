# src/config.py
from typing import Literal

from pydantic import BaseModel

ProfilingMode = Literal["rollout", "trainer", "e2e"]


class ExperimentConfig(BaseModel):
    """
    Global configuration for a GRPO training run.
    Passed between Trainer, Inference, and Rollout workers.
    """

    # Experiment Metadata
    run_name: str = "diplomacy-grpo-v1"
    seed: int = 42
    experiment_tag: str | None = None  # Tag for grouping related runs (e.g., "power-laws")

    # Model Settings
    base_model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    lora_rank: int = 16

    # Environment Settings
    rollout_horizon_years: int = 2
    rollout_visualize_chance: float = 0

    # Training Loop
    total_steps: int = 10
    num_groups_per_step: int = 8  # G in GRPO
    samples_per_group: int = 8  # N in GRPO

    # Inference Settings
    max_new_tokens: int = 256
    temperature: float = 0.8
    compact_prompts: bool = False

    # Profiling / instrumentation
    profiling_mode: ProfilingMode | None = None
    profile_run_name: str | None = None
    profiling_trace_steps: int = 3

    @property
    def batch_size(self) -> int:
        return self.num_groups_per_step * self.samples_per_group

    @property
    def simulated_years_per_step(self) -> int:
        """Calculate total simulated years per training step."""
        return self.num_groups_per_step * self.samples_per_group * self.rollout_horizon_years

    @property
    def total_simulated_years(self) -> int:
        """Calculate total simulated years for the full training run."""
        return self.simulated_years_per_step * self.total_steps
