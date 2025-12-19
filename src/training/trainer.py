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

    # Pre-tokenized data stats
    pretokenized_count: int = 0
    fallback_tokenized_count: int = 0

    # Advantage statistics (after normalization)
    advantage_mean: float = 0.0
    advantage_std: float = 0.0
    advantage_min: float = 0.0
    advantage_max: float = 0.0
    advantages_clipped: int = 0  # Count of advantages that were clipped

    # Skip diagnostics
    skipped_zero_variance_groups: int = 0  # Groups skipped due to low variance
    total_samples_skipped: int = 0  # Total samples lost due to all skipping
    effective_batch_size: int = 0  # Final batch size after all skipping
    skip_rate: float = 0.0  # Fraction of input trajectories skipped

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
            "pretokenized_count": self.pretokenized_count,
            "fallback_tokenized_count": self.fallback_tokenized_count,
            # Advantage statistics
            "advantage_mean": self.advantage_mean,
            "advantage_std": self.advantage_std,
            "advantage_min": self.advantage_min,
            "advantage_max": self.advantage_max,
            "advantages_clipped": self.advantages_clipped,
            # Skip diagnostics
            "skipped_zero_variance_groups": self.skipped_zero_variance_groups,
            "total_samples_skipped": self.total_samples_skipped,
            "effective_batch_size": self.effective_batch_size,
            "skip_rate": self.skip_rate,
        }


def process_trajectories(
    trajectories: list[dict],
    tokenizer,
    min_group_size: int = 2,
    verbose: bool = False,
    advantage_clip: float | None = None,
    advantage_min_std: float = 1e-8,
) -> tuple[list[dict], TrajectoryStats]:
    """
    Takes raw rollouts and prepares tensors for the training loop.

    Steps:
    1. Groups trajectories by 'group_id'
    2. Calculates normalized advantages within each group
    3. Uses pre-tokenized data if available, otherwise tokenizes prompt + completion
    4. Creates labels with prompt tokens masked (-100)

    Args:
        trajectories: List of trajectory dicts with keys:
            - 'prompt': str
            - 'completion': str
            - 'reward': float
            - 'group_id': str
            - 'prompt_token_ids': list[int] (optional, for pre-tokenized data)
            - 'completion_token_ids': list[int] (optional, for pre-tokenized data)
            - 'completion_logprobs': list[float] (optional, for reference logprobs)
        tokenizer: HuggingFace tokenizer
        min_group_size: Minimum samples per group for valid advantage computation.
            Groups smaller than this are skipped (can't compute meaningful std).
            Recommended: 3-4 for stable advantages (2 can have high variance).
        advantage_clip: If set, clip advantages to [-clip, +clip] to prevent
            extreme gradients from outliers.
        advantage_min_std: Minimum std for advantage normalization. Groups with
            std below this are skipped as they provide no gradient signal.

    Returns:
        Tuple of (processed_batch, stats) where processed_batch is list of dicts
        with 'input_ids', 'attention_mask', 'labels', 'advantages'.
        If pre-tokenized data with logprobs is available, also includes 'ref_logprobs'.
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
    all_advantages: list[float] = []  # Track all advantages for stats
    input_trajectory_count = len(trajectories)  # For skip rate calculation

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

        # Handle edge case: all rewards identical or near-identical (low std)
        # Skip groups with low variance - they provide no gradient signal
        if std_r < advantage_min_std:
            stats.skipped_zero_variance_groups += 1
            continue

        # Normalize Advantages within group
        for item in items:
            advantage = (item["reward"] - mean_r) / std_r

            # Clip advantages if configured (prevents extreme gradients from outliers)
            if advantage_clip is not None:
                clipped_advantage = max(-advantage_clip, min(advantage_clip, advantage))
                if clipped_advantage != advantage:
                    stats.advantages_clipped += 1
                advantage = clipped_advantage

            all_advantages.append(advantage)

            # Check if pre-tokenized data is available
            prompt_token_ids = item.get("prompt_token_ids", [])
            completion_token_ids = item.get("completion_token_ids", [])
            completion_logprobs = item.get("completion_logprobs", [])

            if prompt_token_ids and completion_token_ids:
                # Use pre-tokenized data (skip tokenization!)
                stats.pretokenized_count += 1

                # Concatenate prompt + completion token IDs
                full_token_ids = prompt_token_ids + completion_token_ids
                input_ids = torch.tensor(full_token_ids, dtype=torch.long)
                attention_mask = torch.ones_like(input_ids)

                prompt_len = len(prompt_token_ids)

                # Create labels with prompt masked
                labels = input_ids.clone()
                labels[:prompt_len] = -100

                # Process reference logprobs if available
                # IMPORTANT: Only use if ALL trajectories in the batch will have them
                # Otherwise, the loss function will mix cached and computed ref logprobs
                ref_logprobs = None
                if completion_logprobs and len(completion_logprobs) == len(completion_token_ids):
                    # Sum completion logprobs to get sequence-level logprob
                    ref_logprobs = sum(completion_logprobs)
            else:
                # Fallback: tokenize prompt + completion (old path)
                stats.fallback_tokenized_count += 1

                text = item["prompt"] + item["completion"]
                enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
                input_ids = enc.input_ids[0]
                attention_mask = enc.attention_mask[0]

                # Encode prompt separately to find where completion starts
                prompt_tokens = tokenizer.encode(
                    item["prompt"],
                    add_special_tokens=False,
                    truncation=True,
                    max_length=1536,
                )
                prompt_len = len(prompt_tokens)

                labels = input_ids.clone()
                labels[:prompt_len] = -100

                ref_logprobs = None  # No pre-computed reference logprobs

            # Count completion tokens (non-masked labels)
            completion_tokens = (labels != -100).sum().item()
            total_completion_tokens += completion_tokens

            processed_item = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "advantages": float(advantage),
                "group_id": group_id,  # Keep for debugging
                "reward": item["reward"],  # Keep for debugging
            }

            # Add reference logprobs if available (enables skipping reference forward pass)
            if ref_logprobs is not None:
                processed_item["ref_logprobs"] = ref_logprobs

            processed_batch.append(processed_item)

    stats.total_trajectories = len(processed_batch)
    stats.total_tokens = int(total_completion_tokens)
    stats.avg_completion_tokens = (
        total_completion_tokens / len(processed_batch) if processed_batch else 0.0
    )

    # Global batch normalization (REINFORCE++ style)
    # After group-level normalization, we apply global normalization across the entire batch.
    # This eliminates the bias from group-level normalization that only vanishes as batch size → ∞.
    # Reference: https://arxiv.org/html/2501.03262
    if all_advantages and len(all_advantages) > 1:
        adv_array = np.array(all_advantages)
        global_mean = float(adv_array.mean())
        global_std = float(adv_array.std())

        if global_std > 1e-8:  # Avoid division by zero
            # Re-normalize all advantages with global stats
            for i, item in enumerate(processed_batch):
                old_adv = item["advantages"]
                new_adv = (old_adv - global_mean) / global_std
                item["advantages"] = new_adv
                all_advantages[i] = new_adv

    # Compute advantage statistics (now on globally normalized advantages)
    if all_advantages:
        adv_array = np.array(all_advantages)
        stats.advantage_mean = float(adv_array.mean())
        stats.advantage_std = float(adv_array.std())
        stats.advantage_min = float(adv_array.min())
        stats.advantage_max = float(adv_array.max())

    # Compute skip diagnostics
    stats.effective_batch_size = len(processed_batch)
    stats.total_samples_skipped = input_trajectory_count - len(processed_batch)
    stats.skip_rate = (
        stats.total_samples_skipped / input_trajectory_count if input_trajectory_count > 0 else 0.0
    )

    # Log warnings for edge cases
    if verbose:
        if stats.skipped_single_sample_groups > 0:
            print(
                f"⚠️ Skipped {stats.skipped_single_sample_groups} groups "
                f"(fewer than {min_group_size} samples)"
            )
        if stats.skipped_zero_variance_groups > 0:
            print(
                f"⚠️ Skipped {stats.skipped_zero_variance_groups} groups "
                f"(std < {advantage_min_std}, no gradient signal)"
            )
        if stats.fallback_tokenized_count > 0 and stats.pretokenized_count > 0:
            print(
                f"⚠️ Mixed tokenization: {stats.pretokenized_count} pre-tokenized, "
                f"{stats.fallback_tokenized_count} fallback tokenized"
            )

    # PRE-VALIDATION: Check for mixed ref_logprobs consistency
    # This catches the issue early, before the batch reaches the loss function.
    # Mixed batches (some with ref_logprobs, some without) create biased KL estimates.
    if processed_batch:
        has_ref = sum(1 for item in processed_batch if "ref_logprobs" in item)
        has_no_ref = len(processed_batch) - has_ref

        if has_ref > 0 and has_no_ref > 0:
            # Mixed batch detected - strip ref_logprobs to force consistent behavior
            # This ensures all items use the reference forward pass instead of
            # mixing cached and computed ref logprobs (which would bias KL estimates).
            if verbose:
                print(
                    f"⚠️ Mixed ref_logprobs in batch: {has_ref} with, {has_no_ref} without. "
                    "Stripping all ref_logprobs for consistency."
                )
            for item in processed_batch:
                item.pop("ref_logprobs", None)

    return processed_batch, stats
