"""
Diplomacy Logits Processor for vLLM v1 Engine (Optimized for Performance)

Key optimizations:
1. Incremental tag detection - only scan new tokens, not entire history
2. Cached trie position - don't re-walk from root each call
3. Stateful tracking - O(1) per token instead of O(n)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

try:
    from vllm.v1.sample.logits_processor import LogitsProcessor
except ImportError:
    # vLLM not available (e.g., in CI test environment)
    LogitsProcessor = object  # type: ignore[assignment, misc]

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.sampling_params import SamplingParams
    from vllm.v1.sample.logits_processor import BatchUpdate


class TokenTrieNode:
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


class RequestState:
    """
    Cached state for a single request. Enables O(1) per-token processing.

    State machine:
    - DORMANT: Before <orders> tag, allow all tokens
    - ACTIVE: Inside <orders> block, constrain via trie
    - DONE: After </orders> tag, allow all tokens
    """

    __slots__ = (
        "root",
        "newline_token_id",
        "eos_token_id",
        "output_token_ids",
        "start_tag_ids",
        "end_tag_ids",
        # Cached state
        "last_processed_len",  # How many tokens we've already processed
        "orders_start_idx",  # Index where <orders> starts (-1 if not found)
        "is_done",  # True if </orders> found
        "current_node",  # Current position in trie
        "tag_match_progress",  # Partial match progress for tags
    )

    def __init__(
        self,
        root: TokenTrieNode,
        newline_token_id: int,
        eos_token_id: int,
        output_token_ids: list[int],
        start_tag_ids: list[int],
        end_tag_ids: list[int],
    ):
        self.root = root
        self.newline_token_id = newline_token_id
        self.eos_token_id = eos_token_id
        self.output_token_ids = output_token_ids
        self.start_tag_ids = start_tag_ids
        self.end_tag_ids = end_tag_ids

        # Initialize cached state
        self.last_processed_len = 0
        self.orders_start_idx = -1
        self.is_done = False
        self.current_node = root
        self.tag_match_progress = 0  # For incremental tag matching


class DiplomacyLogitsProcessor(LogitsProcessor):  # type: ignore[misc]
    @classmethod
    def validate_params(cls, sampling_params: SamplingParams) -> None:
        if sampling_params.extra_args is None:
            return
        valid_moves_dict = sampling_params.extra_args.get("valid_moves_dict")
        if valid_moves_dict is None:
            return
        if not isinstance(valid_moves_dict, dict):
            raise ValueError("valid_moves_dict must be a dict")
        # Validate inner types
        for unit_key, moves in valid_moves_dict.items():
            if not isinstance(unit_key, str):
                raise ValueError(f"Unit key must be str, got {type(unit_key).__name__}")
            if not isinstance(moves, list):
                raise ValueError(f"Moves must be list, got {type(moves).__name__}")
            for move in moves:
                if not isinstance(move, str):
                    raise ValueError(f"Each move must be str, got {type(move).__name__}")

    def __init__(self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool):
        from transformers import AutoTokenizer

        self.device = device
        self.is_pin_memory = is_pin_memory

        model_name = vllm_config.model_config.model  # type: ignore[attr-defined]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Cache Special Tokens
        self.newline_token_id = self.tokenizer.encode("\n", add_special_tokens=False)[-1]
        self.eos_token_id = self.tokenizer.eos_token_id

        # Cache Tag Sequences for Fast Matching
        self.start_tag_ids = self.tokenizer.encode("<orders>", add_special_tokens=False)
        self.end_tag_ids = self.tokenizer.encode("</orders>", add_special_tokens=False)

        self.req_states: dict[int, RequestState] = {}

    def is_argmax_invariant(self) -> bool:
        return False

    def _build_trie(self, valid_moves_dict: dict[str, list[str]]) -> TokenTrieNode:
        root = TokenTrieNode()
        for moves in valid_moves_dict.values():
            for move in moves:
                token_ids = self.tokenizer.encode(move, add_special_tokens=False)
                root.add_sequence(token_ids)
        return root

    def _find_sequence_at(self, tokens: list[int], needle: list[int], start: int) -> int:
        """Check if needle appears at exactly position start in tokens."""
        if start < 0 or start + len(needle) > len(tokens):
            return -1
        for i, n_tok in enumerate(needle):
            if tokens[start + i] != n_tok:
                return -1
        return start

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        from vllm.v1.sample.logits_processor import MoveDirectionality

        if batch_update is None:
            return

        for index in batch_update.removed:
            self.req_states.pop(index, None)

        for index, params, _, output_token_ids in batch_update.added:
            if params is None or params.extra_args is None:
                self.req_states.pop(index, None)
                continue

            valid_moves_dict = params.extra_args.get("valid_moves_dict")
            if not valid_moves_dict:
                self.req_states.pop(index, None)
                continue

            # Check if we should start in ACTIVE mode (prompt already contains <orders>)
            start_active = params.extra_args.get("start_active", False)

            root = self._build_trie(valid_moves_dict)
            state = RequestState(
                root=root,
                newline_token_id=self.newline_token_id,
                eos_token_id=self.eos_token_id or 0,
                output_token_ids=output_token_ids,
                start_tag_ids=self.start_tag_ids,
                end_tag_ids=self.end_tag_ids,
            )

            # If prompt ends with <orders>, start in ACTIVE mode immediately
            if start_active:
                state.orders_start_idx = 0  # Mark as found
                state.current_node = root  # Ready for trie walk

            self.req_states[index] = state

        for src, dst, directionality in batch_update.moved:
            src_state = self.req_states.pop(src, None)
            dst_state = self.req_states.pop(dst, None)
            if src_state is not None:
                self.req_states[dst] = src_state
            if directionality == MoveDirectionality.SWAP and dst_state is not None:
                self.req_states[src] = dst_state

    def _update_request_state(self, state: RequestState) -> None:
        """
        Incrementally update cached state based on new tokens.
        Only processes tokens since last_processed_len.
        """
        tokens = state.output_token_ids
        current_len = len(tokens)

        if current_len <= state.last_processed_len:
            return  # No new tokens

        # Process only new tokens
        for i in range(state.last_processed_len, current_len):
            token = tokens[i]

            # STATE: Looking for <orders>
            if state.orders_start_idx == -1:
                # Check if this token continues or starts the <orders> tag
                if token == state.start_tag_ids[state.tag_match_progress]:
                    state.tag_match_progress += 1
                    if state.tag_match_progress == len(state.start_tag_ids):
                        # Found complete <orders> tag!
                        state.orders_start_idx = i - len(state.start_tag_ids) + 1
                        state.tag_match_progress = 0  # Reset for </orders> search
                        state.current_node = state.root  # Ready for trie walk
                else:
                    # Reset progress (but check if current token starts the tag)
                    state.tag_match_progress = 0
                    if token == state.start_tag_ids[0]:
                        state.tag_match_progress = 1

            # STATE: Inside <orders>, looking for </orders>
            elif not state.is_done:
                # Check for </orders> closing tag
                if token == state.end_tag_ids[state.tag_match_progress]:
                    state.tag_match_progress += 1
                    if state.tag_match_progress == len(state.end_tag_ids):
                        state.is_done = True
                else:
                    # Reset progress (check if current token starts </orders>)
                    state.tag_match_progress = 0
                    if token == state.end_tag_ids[0]:
                        state.tag_match_progress = 1

                    # Update trie position for this token
                    if token == state.newline_token_id:
                        if state.current_node.is_end_of_move:
                            state.current_node = state.root  # New line, reset trie
                        # else: invalid state, but let trie handle it
                    elif token in state.current_node.children:
                        state.current_node = state.current_node.children[token]
                    else:
                        # Dead end - stay at current node (will mask in apply)
                        pass

        state.last_processed_len = current_len

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.req_states:
            return logits

        for idx, state in self.req_states.items():
            if idx >= logits.shape[0]:
                continue

            # Incrementally update state based on new tokens
            self._update_request_state(state)

            # DORMANT: No <orders> tag yet - allow free generation
            if state.orders_start_idx == -1:
                continue

            # DONE: After </orders> - allow free generation
            if state.is_done:
                continue

            # ACTIVE: Inside <orders> block - apply trie constraint
            current_node = state.current_node

            # Check if we're in the middle of generating </orders> tag
            if state.tag_match_progress > 0 and state.end_tag_ids:
                # We've started </orders>, only allow the next token in the sequence
                next_tag_token = state.end_tag_ids[state.tag_match_progress]
                mask = torch.full((logits.shape[1],), float("-inf"), device=logits.device)
                mask[next_tag_token] = 0
                logits[idx] = logits[idx] + mask
                continue

            # Determine valid next tokens from trie
            valid_tokens = list(current_node.children.keys())
            is_at_end = current_node.is_end_of_move
            is_at_root = current_node is state.root

            # Build allowed set
            allowed = set(valid_tokens)

            # At root: allow newlines (formatting) and closing tag start
            if is_at_root:
                allowed.add(state.newline_token_id)
                if state.end_tag_ids:
                    allowed.add(state.end_tag_ids[0])

            # At end of move: allow newline (to start next move) or close
            if is_at_end:
                allowed.add(state.newline_token_id)
                allowed.add(state.eos_token_id)
                if state.end_tag_ids:
                    allowed.add(state.end_tag_ids[0])

            # Safety: If dead end, allow escape via EOS or closing tag
            if not allowed:
                allowed.add(state.eos_token_id)
                if state.end_tag_ids:
                    allowed.add(state.end_tag_ids[0])

            # Apply mask efficiently
            mask = torch.full((logits.shape[1],), float("-inf"), device=logits.device)
            mask[list(allowed)] = 0
            logits[idx] = logits[idx] + mask

        return logits
