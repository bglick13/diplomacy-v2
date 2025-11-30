"""
Diplomacy Logits Processor for vLLM v1 Engine

This module implements a custom batch-level logits processor conforming to the
vLLM v1 LogitsProcessor API. It constrains token generation to valid Diplomacy
orders using a Trie (prefix tree) data structure.

Usage:
    1. Load the processor at engine initialization via logits_processors arg
    2. Pass valid_moves_dict per-request via SamplingParams.extra_args:

       SamplingParams(
           extra_args={"valid_moves_dict": {"A PAR": ["A PAR - BUR", "A PAR - MAR"]}}
       )

Critical Nuances:
- Tokenizer "Space" Sensitivity: Models treat " Paris" and "Paris" as different
  tokens. Ensure valid_moves strings match the exact spacing in the prompt.
- The Trie is rebuilt per-request since valid_moves changes every turn.
- Newline token resets the Trie to allow multiple orders per generation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from vllm.v1.sample.logits_processor import LogitsProcessor

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.sampling_params import SamplingParams
    from vllm.v1.sample.logits_processor import BatchUpdate


class TokenTrieNode:
    """A node in the prefix tree of valid token sequences."""

    __slots__ = ("children", "is_end_of_move")

    def __init__(self):
        self.children: dict[int, TokenTrieNode] = {}
        self.is_end_of_move: bool = False

    def add_sequence(self, token_ids: list[int]):
        node = self
        for token_id in token_ids:
            if token_id not in node.children:
                node.children[token_id] = TokenTrieNode()
            node = node.children[token_id]
        node.is_end_of_move = True


class RequestTrieState:
    """Tracks Trie state for a single request."""

    __slots__ = ("root", "newline_token_id", "eos_token_id", "output_token_ids")

    def __init__(
        self,
        root: TokenTrieNode,
        newline_token_id: int,
        eos_token_id: int,
        output_token_ids: list[int],
    ):
        self.root = root
        self.newline_token_id = newline_token_id
        self.eos_token_id = eos_token_id
        self.output_token_ids = output_token_ids


class DiplomacyLogitsProcessor(LogitsProcessor):
    """
    Batch-level LogitsProcessor for vLLM v1 to enforce valid Diplomacy orders.

    Subclasses vllm.v1.sample.logits_processor.LogitsProcessor and implements:
    - validate_params(cls, params): Validates SamplingParams
    - __init__(self, vllm_config, device, is_pin_memory): Initializes processor
    - apply(self, logits): Transforms batch logits tensor
    - is_argmax_invariant(self): Returns False (we modify argmax)
    - update_state(self, batch_update): Handles batch state changes
    """

    @classmethod
    def validate_params(cls, sampling_params: "SamplingParams") -> None:
        """Validate that extra_args contains valid_moves_dict if present."""
        if sampling_params.extra_args is None:
            return

        valid_moves_dict = sampling_params.extra_args.get("valid_moves_dict")
        if valid_moves_dict is None:
            return

        if not isinstance(valid_moves_dict, dict):
            raise ValueError(
                f"valid_moves_dict must be a dict, got {type(valid_moves_dict)}"
            )

        for unit, moves in valid_moves_dict.items():
            if not isinstance(unit, str):
                raise ValueError(f"Unit key must be str, got {type(unit)}")
            if not isinstance(moves, list):
                raise ValueError(f"Moves must be list, got {type(moves)}")
            for move in moves:
                if not isinstance(move, str):
                    raise ValueError(f"Each move must be str, got {type(move)}")

    def __init__(
        self,
        vllm_config: "VllmConfig",
        device: torch.device,
        is_pin_memory: bool,
    ):
        """Initialize the batch-level logits processor."""
        from transformers import AutoTokenizer

        self.device = device
        self.is_pin_memory = is_pin_memory

        # Load tokenizer from model config
        model_name = vllm_config.model_config.model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Cache special token IDs
        self.newline_token_id = self.tokenizer.encode("\n", add_special_tokens=False)[
            -1
        ]
        self.eos_token_id = self.tokenizer.eos_token_id

        # Per-request state: index -> RequestTrieState
        self.req_states: dict[int, RequestTrieState] = {}

    def is_argmax_invariant(self) -> bool:
        """
        Returns False because this processor modifies which token has highest logit.
        We mask invalid tokens with -inf, fundamentally changing the distribution.
        """
        return False

    def _build_trie(self, valid_moves_dict: dict[str, list[str]]) -> TokenTrieNode:
        """Build a Trie from valid moves dictionary."""
        root = TokenTrieNode()

        for moves in valid_moves_dict.values():
            for move in moves:
                token_ids = self.tokenizer.encode(move, add_special_tokens=False)
                root.add_sequence(token_ids)

        return root

    def _walk_trie(self, state: RequestTrieState) -> tuple[list[int], bool]:
        """
        Walk the Trie based on output_token_ids to find valid next tokens.

        Returns:
            (valid_token_ids, is_at_end_of_move)
        """
        current_node = state.root

        for token_id in state.output_token_ids:
            # Check for reset (newline after valid move end)
            if token_id == state.newline_token_id and current_node.is_end_of_move:
                current_node = state.root
                continue

            # Walk down the tree
            if token_id in current_node.children:
                current_node = current_node.children[token_id]
            else:
                # Dead end - off the valid path
                return [], False

        valid_tokens = list(current_node.children.keys())
        return valid_tokens, current_node.is_end_of_move

    def update_state(self, batch_update: "BatchUpdate | None") -> None:
        """
        Update internal state based on batch changes.

        Processes: removes, adds, then moves (in that order per vLLM spec).
        """
        from vllm.v1.sample.logits_processor import MoveDirectionality

        if batch_update is None:
            return

        # Process removed requests
        for index in batch_update.removed:
            self.req_states.pop(index, None)

        # Process added requests
        # AddedRequest = (index, params, prompt_tok_ids, output_tok_ids)
        for index, params, _prompt_token_ids, output_token_ids in batch_update.added:
            if params is None or params.extra_args is None:
                self.req_states.pop(index, None)
                continue

            valid_moves_dict = params.extra_args.get("valid_moves_dict")
            if valid_moves_dict is None:
                self.req_states.pop(index, None)
                continue

            # Build Trie for this request
            root = self._build_trie(valid_moves_dict)

            # IMPORTANT: output_token_ids is a REFERENCE to the running output
            # tokens list. Via this reference we always see the latest tokens.
            self.req_states[index] = RequestTrieState(
                root=root,
                newline_token_id=self.newline_token_id,
                eos_token_id=self.eos_token_id or 0,
                output_token_ids=output_token_ids,
            )

        # Process moves (both unidirectional and swaps)
        for src_idx, dst_idx, directionality in batch_update.moved:
            src_state = self.req_states.pop(src_idx, None)
            dst_state = self.req_states.pop(dst_idx, None)

            if src_state is not None:
                self.req_states[dst_idx] = src_state

            if directionality == MoveDirectionality.SWAP and dst_state is not None:
                self.req_states[src_idx] = dst_state

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply Diplomacy order constraints to a batch of logits.

        Args:
            logits: Tensor of shape (num_requests, vocab_size)

        Returns:
            Modified logits tensor with invalid tokens masked to -inf
        """
        if not self.req_states:
            return logits

        # Process each request that has Trie state
        for idx, state in self.req_states.items():
            if idx >= logits.shape[0]:
                continue

            valid_tokens, is_at_end = self._walk_trie(state)

            # Build allowed token set
            allowed_tokens = set(valid_tokens)
            if is_at_end:
                allowed_tokens.add(state.newline_token_id)
                allowed_tokens.add(state.eos_token_id)

            # Fallback: if no valid tokens, allow EOS to prevent infinite loops
            if not allowed_tokens:
                allowed_tokens.add(state.eos_token_id)

            # Create mask and apply
            mask = torch.full((logits.shape[1],), float("-inf"), device=logits.device)
            allowed_list = list(allowed_tokens)
            mask[allowed_list] = 0

            logits[idx] = logits[idx] + mask

        return logits


# =============================================================================
# Legacy Per-Request Processor (for testing and non-vLLM usage)
# =============================================================================


class LegacyDiplomacyLogitsProcessor:
    """
    Legacy per-request LogitsProcessor for testing and non-vLLM usage.

    This maintains the original callable interface:
        processor(input_ids: list[int], scores: Tensor) -> Tensor
    """

    def __init__(
        self,
        tokenizer: Any,
        valid_moves_dict: dict[str, list[str]],
        device: str = "cuda",
    ):
        self.tokenizer = tokenizer
        self.device = device

        self.all_valid_orders: list[str] = []
        for moves in valid_moves_dict.values():
            self.all_valid_orders.extend(moves)

        self.newline_token_id = tokenizer.encode("\n", add_special_tokens=False)[-1]
        self.eos_token_id = tokenizer.eos_token_id

        # Build the Trie
        self.root = TokenTrieNode()
        self._build_trie()

    def _build_trie(self):
        for order_str in self.all_valid_orders:
            token_ids = self.tokenizer.encode(order_str, add_special_tokens=False)
            self.root.add_sequence(token_ids)

    def __call__(self, input_ids: list[int], scores: torch.Tensor) -> torch.Tensor:
        """
        Walks the Trie based on input_ids history to find valid next tokens.
        """
        current_node = self.root

        for token_id in input_ids:
            if token_id == self.newline_token_id and current_node.is_end_of_move:
                current_node = self.root
                continue

            if token_id in current_node.children:
                current_node = current_node.children[token_id]
            else:
                current_node = TokenTrieNode()  # Dead end
                break

        valid_next_tokens = list(current_node.children.keys())

        if current_node.is_end_of_move:
            valid_next_tokens.append(self.newline_token_id)
            valid_next_tokens.append(self.eos_token_id)

        mask = torch.full_like(scores, float("-inf"))

        if valid_next_tokens:
            mask[valid_next_tokens] = 0
            scores = scores + mask
        else:
            scores = torch.full_like(scores, float("-inf"))
            scores[self.eos_token_id] = 0

        return scores
