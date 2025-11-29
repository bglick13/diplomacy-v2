"""
Critical Nuances for the Coding Agent
When you provide this code, add these Integration Notes so the agent knows how to hook it up:

Tokenizer "Space" Sensitivity:

Models like Llama/Mistral treat " Paris" (with space) and "Paris" (no space) as different tokens.

Instruction: Ensure the valid_moves strings match the exact spacing used in the prompt. If the prompt is <orders>\n, the first order probably doesn't have a leading space, but subsequent lines might depending on how the newline token behaves.

Safe Bet: Normalize the prompt to ensure no trailing spaces, and encode valid moves without leading spaces.

State Management in vLLM:

Standard LogitsProcessor in HuggingFace is stateful.

In vLLM, you often pass the processor class and the engine instantiates it.

Workaround: For your Modal InferenceEngine, you will likely effectively re-create this processor object for every request (because valid_moves changes every turn).

Performance: Building the Trie takes ~5ms for 100 moves. This is negligible compared to the network call. It is safe to rebuild per request.

The "Stop" Token:

The code above allows \n to reset the Trie to the root (allowing multiple orders).

You must ensure the GRPO training loop treats </orders> or a specific stop sequence as the signal to stop generating.
"""

from typing import Dict, List

import torch
from transformers import PreTrainedTokenizer


class TokenTrieNode:
    """A node in the prefix tree of valid token sequences."""

    def __init__(self):
        self.children: Dict[int, TokenTrieNode] = {}
        self.is_end_of_move: bool = False

    def add_sequence(self, token_ids: List[int]):
        node = self
        for token_id in token_ids:
            if token_id not in node.children:
                node.children[token_id] = TokenTrieNode()
            node = node.children[token_id]
        node.is_end_of_move = True


class DiplomacyLogitsProcessor:
    """
    Custom LogitsProcessor for vLLM to enforce valid Diplomacy orders.
    Refactored to re-walk the Trie on every call to support stateless validation.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        valid_moves_dict: Dict[str, List[str]],
        device: str = "cuda",
    ):
        self.tokenizer = tokenizer
        self.device = device

        self.all_valid_orders = []
        for unit, moves in valid_moves_dict.items():
            self.all_valid_orders.extend(moves)

        # Add the "newline" token as a valid separator between moves
        # Note: We take the last token of the encoded newline to handle potential tokenizer artifacts
        self.newline_token_id = tokenizer.encode("\n", add_special_tokens=False)[-1]
        self.eos_token_id = tokenizer.eos_token_id

        # Build the Trie
        self.root = TokenTrieNode()
        self._build_trie()

    def _build_trie(self):
        for order_str in self.all_valid_orders:
            token_ids = self.tokenizer.encode(order_str, add_special_tokens=False)
            self.root.add_sequence(token_ids)

    def __call__(self, input_ids: List[int], scores: torch.Tensor) -> torch.Tensor:
        """
        Walks the Trie based on the full input_ids history to find the valid next tokens.
        """
        # Always start walking from the root
        current_node = self.root

        # Walk the history
        for token_id in input_ids:
            # 1. Check for Reset (Newline)
            # If we hit a newline and we were at a valid end of move, we reset to root
            # to allow the model to start generating a NEW move.
            if token_id == self.newline_token_id and current_node.is_end_of_move:
                current_node = self.root
                continue

            # 2. Walk down the tree
            if token_id in current_node.children:
                current_node = current_node.children[token_id]
            else:
                # We are off the valid path.
                # In strict generation, this shouldn't happen if previous tokens were masked correctly.
                # If we are lost, we stay 'lost' (node with no children) which results in -inf mask
                # except for potentially EOS if we want to fail gracefully.
                # For this implementation, we break the loop and current_node will likely have no children.
                # Creates a "dead end".
                current_node = TokenTrieNode()  # Empty dummy node
                break

        # Get valid next tokens from the node we landed on
        valid_next_tokens = list(current_node.children.keys())

        # Allow Newline or EOS if we are at the end of a valid move
        if current_node.is_end_of_move:
            valid_next_tokens.append(self.newline_token_id)
            valid_next_tokens.append(self.eos_token_id)

        # Apply Mask
        mask = torch.full_like(scores, float("-inf"))

        if valid_next_tokens:
            mask[valid_next_tokens] = 0
            scores = scores + mask
        else:
            # Dead end?
            # If we are truly stuck, allow EOS to prevent infinite loops of garbage
            scores[self.eos_token_id] = 0

        return scores
