# src/config.py
from pydantic import BaseModel


class ExperimentConfig(BaseModel):
    """
    Global configuration for a GRPO training run.
    Passed between Trainer, Inference, and Rollout workers.
    """

    # Experiment Metadata
    run_name: str = "diplomacy-grpo-v1"
    seed: int = 42

    # Model Settings
    base_model_id: str = "mistralai/Mistral-7B-v0.1"
    lora_rank: int = 16

    # Environment Settings
    rollout_horizon_years: int = 2
    rollout_visualize_chance: float = 1.0

    # Training Loop
    total_steps: int = 100
    num_groups_per_step: int = 8  # G in GRPO
    samples_per_group: int = 8  # N in GRPO

    # Inference Settings
    max_new_tokens: int = 256
    temperature: float = 0.8

    @property
    def batch_size(self) -> int:
        return self.num_groups_per_step * self.samples_per_group
