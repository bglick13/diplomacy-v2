from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


@dataclass
class TrajectoryStats:
    """Statistics about processed trajectories for logging."""

    total_trajectories: int = 0
    total_groups: int = 0
    skipped_single_sample_groups: int = 0
    skipped_empty_groups: int = 0

    # Reward statistics (before normalization)
    reward_mean: float = 0.0
    reward_std: float = 0.0
    reward_min: float = 0.0
    reward_max: float = 0.0

    # Per-group statistics
    group_sizes: list[int] = field(default_factory=list)
    group_reward_stds: list[float] = field(default_factory=list)

    # Token statistics
    total_tokens: int = 0
    avg_completion_tokens: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for logging."""
        return {
            "total_trajectories": self.total_trajectories,
            "total_groups": self.total_groups,
            "skipped_single_sample_groups": self.skipped_single_sample_groups,
            "skipped_empty_groups": self.skipped_empty_groups,
            "reward_mean": self.reward_mean,
            "reward_std": self.reward_std,
            "reward_min": self.reward_min,
            "reward_max": self.reward_max,
            "avg_group_size": np.mean(self.group_sizes) if self.group_sizes else 0,
            "avg_group_reward_std": np.mean(self.group_reward_stds)
            if self.group_reward_stds
            else 0,
            "total_tokens": self.total_tokens,
            "avg_completion_tokens": self.avg_completion_tokens,
        }


def process_trajectories(
    trajectories: list[dict],
    tokenizer,
    min_group_size: int = 2,
) -> tuple[list[dict], TrajectoryStats]:
    """
    Takes raw rollouts and prepares tensors for the training loop.

    Steps:
    1. Groups trajectories by 'group_id'
    2. Calculates normalized advantages within each group
    3. Tokenizes prompt + completion
    4. Creates labels with prompt tokens masked (-100)

    Args:
        trajectories: List of trajectory dicts with keys:
            - 'prompt': str
            - 'completion': str
            - 'reward': float
            - 'group_id': str
        tokenizer: HuggingFace tokenizer
        min_group_size: Minimum samples per group for valid advantage computation.
            Groups smaller than this are skipped (can't compute meaningful std).

    Returns:
        Tuple of (processed_batch, stats) where processed_batch is list of dicts
        with 'input_ids', 'attention_mask', 'labels', 'advantages'.
    """
    stats = TrajectoryStats()

    if not trajectories:
        return [], stats

    # Collect all rewards for global stats
    all_rewards = [t["reward"] for t in trajectories]
    stats.reward_mean = float(np.mean(all_rewards))
    stats.reward_std = float(np.std(all_rewards))
    stats.reward_min = float(np.min(all_rewards))
    stats.reward_max = float(np.max(all_rewards))

    # Group trajectories by group_id
    groups: dict[str, list[dict]] = defaultdict(list)
    for t in trajectories:
        groups[t["group_id"]].append(t)

    stats.total_groups = len(groups)

    processed_batch = []
    total_completion_tokens = 0

    for group_id, items in groups.items():
        # Skip empty groups (shouldn't happen but defensive)
        if not items:
            stats.skipped_empty_groups += 1
            continue

        # Skip groups with too few samples - can't compute meaningful advantage
        if len(items) < min_group_size:
            stats.skipped_single_sample_groups += 1
            continue

        stats.group_sizes.append(len(items))

        # Calculate Group Statistics
        rewards = np.array([x["reward"] for x in items])
        mean_r = rewards.mean()
        std_r = rewards.std()

        stats.group_reward_stds.append(float(std_r))

        # Handle edge case: all rewards identical (std=0)
        # In this case, all advantages should be 0 (no signal)
        if std_r < 1e-8:
            std_r = 1.0  # Avoid division by zero, advantages will all be ~0

        # Normalize Advantages within group
        for item in items:
            advantage = (item["reward"] - mean_r) / std_r

            if all(k in item for k in ("input_ids", "attention_mask", "labels")):
                input_ids = torch.tensor(item["input_ids"], dtype=torch.long)
                attention_mask = torch.tensor(
                    item["attention_mask"], dtype=torch.long
                )
                labels = torch.tensor(item["labels"], dtype=torch.long)
            else:
                # Tokenize: Prompt + Completion together
                text = item["prompt"] + item["completion"]
                enc = tokenizer(
                    text, return_tensors="pt", truncation=True, max_length=2048
                )
                input_ids = enc.input_ids[0]
                attention_mask = enc.attention_mask[0]

                # Create Labels (Mask the prompt tokens with -100)
                # Encode prompt separately to find where completion starts
                prompt_tokens = tokenizer.encode(
                    item["prompt"],
                    add_special_tokens=False,
                    truncation=True,
                    max_length=1536,
                )
                prompt_len = len(prompt_tokens)

                labels = input_ids.clone()
                labels[:prompt_len] = -100  # Mask prompt

            # Count completion tokens (non-masked labels)
            completion_tokens = (labels != -100).sum().item()
            total_completion_tokens += completion_tokens

            processed_batch.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                    "advantages": float(advantage),
                    "group_id": group_id,  # Keep for debugging
                    "reward": item["reward"],  # Keep for debugging
                }
            )
            if "reference_logprob" in item:
                processed_batch[-1]["reference_logprob"] = float(
                    item["reference_logprob"]
                )

    stats.total_trajectories = len(processed_batch)
    stats.total_tokens = total_completion_tokens
    stats.avg_completion_tokens = (
        total_completion_tokens / len(processed_batch) if processed_batch else 0.0
    )

    return processed_batch, stats
