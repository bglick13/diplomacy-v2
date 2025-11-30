from collections import defaultdict

import numpy as np


def process_trajectories(trajectories, tokenizer):
    """
    Takes raw rollouts and prepares tensors for the training loop.
    1. Groups by 'group_id'
    2. Calculates Advantages
    3. Tokenizes
    """
    # Grouping
    groups = defaultdict(list)
    for t in trajectories:
        groups[t["group_id"]].append(t)

    processed_batch = []

    for group_id, items in groups.items():
        # Calculate Group Stats
        rewards = [x["reward"] for x in items]
        mean_r = np.mean(rewards)
        std_r = np.std(rewards) + 1e-4

        # Normalize Advantages
        for item in items:
            advantage = (item["reward"] - mean_r) / std_r

            # Tokenize: Prompt + Completion
            # We assume prompt is already formatted string
            text = item["prompt"] + item["completion"]

            enc = tokenizer(text, return_tensors="pt")
            input_ids = enc.input_ids[0]
            attention_mask = enc.attention_mask[0]

            # Create Labels (Mask the prompt part)
            # Find where completion starts.
            # Heuristic: Encode prompt separately to find length.
            prompt_len = len(tokenizer.encode(item["prompt"], add_special_tokens=False))
            labels = input_ids.clone()
            labels[:prompt_len] = -100  # Mask prompt

            processed_batch.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                    "advantages": advantage,
                }
            )

    return processed_batch
