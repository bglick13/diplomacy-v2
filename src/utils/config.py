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

    # Model Settings
    base_model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    lora_rank: int = 16

    # Environment Settings
    rollout_horizon_years: int = 2
    rollout_visualize_chance: float = 0.0
    enable_rollout_replays: bool = False
    use_state_cache: bool = False
    compact_prompts: bool = False  # Minify prompts to reduce token count ~30-40%

    # Training Loop
    total_steps: int = 10
    num_groups_per_step: int = 8  # G in GRPO
    samples_per_group: int = 8  # N in GRPO

    # Inference Settings
    max_new_tokens: int = 256
    temperature: float = 0.8

    # Profiling / instrumentation
    profiling_mode: ProfilingMode | None = None
    profile_run_name: str | None = None
    profiling_trace_steps: int = 3

    @property
    def batch_size(self) -> int:
        return self.num_groups_per_step * self.samples_per_group
