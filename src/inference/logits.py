"""
Diplomacy Logits Processor for vLLM v1 Engine

Fix: Option A (robust tag detection)
- Detect <orders> and </orders> by incremental *text* matching (not token-id sequences),
  which avoids BPE "boundary merge" bugs like ">\n" being a single token.

Also includes practical improvements:
- Trie includes both `move` and `move + "\n"` variants to tolerate BPE merges at line ends.
- Newline handling uses decoded text (`"\n" in piece`) instead of relying solely on a
  single newline token id.
- Prevents duplicate orders per unit: after each order is emitted, the trie is rebuilt
  excluding moves for that unit.

Notes:
- Constraints begin on the first token *after* <orders> is detected.
- When we see a partial "</orders>" suffix, we force-complete the remainder by
  encoding the remaining string and masking logits to that sequence.
- Unit identifiers are extracted from completed moves (e.g., "A PAR - BUR" -> "A PAR")
  and tracked to prevent reuse.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

try:
    from vllm.v1.sample.logits_processor import LogitsProcessor
except ImportError:  # pragma: no cover - fallback for environments without vLLM
    LogitsProcessor = object  # type: ignore[assignment, misc]

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.sampling_params import SamplingParams
    from vllm.v1.sample.logits_processor import BatchUpdate


# -------------------------
# Trie for valid move tokens
# -------------------------


class TokenTrieNode:
    __slots__ = ("children", "is_end_of_move")

    def __init__(self):
        self.children: dict[int, TokenTrieNode] = {}
        self.is_end_of_move: bool = False

    def add_sequence(self, token_ids: list[int]) -> None:
        node = self
        for token_id in token_ids:
            child = node.children.get(token_id)
            if child is None:
                child = TokenTrieNode()
                node.children[token_id] = child
            node = child
        node.is_end_of_move = True


# -------------------------
# Per-request cached state
# -------------------------


class RequestState:
    __slots__ = (
        "root",
        "output_token_ids",
        "eos_token_id",
        "newline_token_id",
        # streaming decode / tag detection
        "last_processed_len",
        "recent_text",
        "recent_text_lower",
        "in_orders",
        "is_done",
        # trie cursor
        "current_node",
        # line counting
        "expected_orders",
        "emitted_orders",
        # end-tag forcing
        "force_end_ids",
        "force_end_pos",
        # track used units
        "valid_moves_dict",
        "current_line_text",
        "used_units",
    )

    def __init__(
        self,
        *,
        root: TokenTrieNode,
        output_token_ids: list[int],
        eos_token_id: int,
        newline_token_id: int,
        expected_orders: int,
        valid_moves_dict: dict[str, list[str]],
    ):
        self.root = root
        self.output_token_ids = output_token_ids
        self.eos_token_id = eos_token_id
        self.newline_token_id = newline_token_id

        self.last_processed_len = 0
        self.recent_text = ""  # rolling buffer of decoded text
        self.recent_text_lower = ""
        self.in_orders = False
        self.is_done = False

        self.current_node = root

        # Stop-gap to avoid runaway generations: close after expected_orders lines
        self.expected_orders = max(expected_orders, 0)
        self.emitted_orders = 0

        self.force_end_ids: list[int] | None = None
        self.force_end_pos: int = 0

        # Track which units have been used
        self.valid_moves_dict = valid_moves_dict
        self.current_line_text = ""
        self.used_units: set[str] = set()


# -------------------------
# Logits Processor
# -------------------------


class DiplomacyLogitsProcessor(LogitsProcessor):  # type: ignore[misc]
    START_TAG = "<orders>"
    END_TAG = "</orders>"
    START_TAG_LOWER = START_TAG.lower()
    END_TAG_LOWER = END_TAG.lower()

    # keep only the last N chars of decoded output (enough to catch tags)
    RECENT_TEXT_MAX_CHARS = 256

    @staticmethod
    def _extract_unit_from_move(move: str) -> str | None:
        """Extract unit identifier from a move string (e.g., 'A PAR - BUR' -> 'A PAR')"""
        parts = move.strip().split()
        if len(parts) >= 2:
            return f"{parts[0]} {parts[1]}"
        return None

    @classmethod
    def validate_params(cls, sampling_params: SamplingParams) -> None:
        if sampling_params.extra_args is None:
            return
        valid_moves_dict = sampling_params.extra_args.get("valid_moves_dict")
        if valid_moves_dict is None:
            return
        # Tolerate legacy list-wrapped payloads (e.g., [ {unit: [moves]} ])
        if isinstance(valid_moves_dict, list):
            if valid_moves_dict and isinstance(valid_moves_dict[0], dict):
                valid_moves_dict = valid_moves_dict[0]
            else:
                raise ValueError("valid_moves_dict must be a dict")
        if not isinstance(valid_moves_dict, dict):
            raise ValueError("valid_moves_dict must be a dict")
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

        # Best-effort newline + EOS ids
        # (newline id may not cover all tokenizer-specific newline variants,
        #  but we also detect "\n" in decoded pieces.)
        nl_ids = self.tokenizer.encode("\n", add_special_tokens=False)
        self.newline_token_id = nl_ids[-1] if nl_ids else 0

        self.eos_token_id = self.tokenizer.eos_token_id
        if self.eos_token_id is None:
            # keep an int for masking escape; you may prefer to omit EOS if None
            self.eos_token_id = 0

        # Used only as a *hint* for allowing the model to start the closing tag
        self.end_tag_ids = self.tokenizer.encode(self.END_TAG, add_special_tokens=False)
        self.end_tag_start_id = self.end_tag_ids[0] if self.end_tag_ids else None

        self.req_states: dict[int, RequestState] = {}

    def is_argmax_invariant(self) -> bool:
        return False

    def _build_trie(
        self, valid_moves_dict: dict[str, list[str]], exclude_units: set[str] | None = None
    ) -> TokenTrieNode:
        """
        Build a trie over token ids for valid moves.

        Important: include `move + "\n"` to tolerate BPE merges like "BUR\n" as one token.

        Args:
            valid_moves_dict: Dictionary mapping unit identifiers to their valid moves
            exclude_units: Set of unit identifiers to exclude from the trie
        """
        root = TokenTrieNode()
        exclude = exclude_units or set()

        for unit, moves in valid_moves_dict.items():
            if unit in exclude:
                continue

            for move in moves:
                # Base move
                ids = self.tokenizer.encode(move, add_special_tokens=False)
                if ids:
                    root.add_sequence(ids)

                # Move + newline (common formatting)
                ids_nl = self.tokenizer.encode(move + "\n", add_special_tokens=False)
                if ids_nl:
                    root.add_sequence(ids_nl)

                # Optional: tolerate CRLF (rare, but cheap)
                ids_crlf = self.tokenizer.encode(move + "\r\n", add_special_tokens=False)
                if ids_crlf:
                    root.add_sequence(ids_crlf)
        return root

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

            if isinstance(valid_moves_dict, list):
                if valid_moves_dict and isinstance(valid_moves_dict[0], dict):
                    valid_moves_dict = valid_moves_dict[0]
                else:
                    self.req_states.pop(index, None)
                    continue

            if not isinstance(valid_moves_dict, dict):
                self.req_states.pop(index, None)
                continue

            start_active = bool(params.extra_args.get("start_active", False))
            expected_orders = len(valid_moves_dict)

            root = self._build_trie(valid_moves_dict)
            state = RequestState(
                root=root,
                output_token_ids=output_token_ids,
                eos_token_id=int(self.eos_token_id),
                newline_token_id=int(self.newline_token_id),
                expected_orders=expected_orders,
                valid_moves_dict=valid_moves_dict,
            )

            if start_active:
                # If prompt already contains <orders>, begin constraining immediately.
                state.in_orders = True
                state.current_node = state.root

            self.req_states[index] = state

        for src, dst, directionality in batch_update.moved:
            src_state = self.req_states.pop(src, None)
            dst_state = self.req_states.pop(dst, None)
            if src_state is not None:
                self.req_states[dst] = src_state
            if directionality == MoveDirectionality.SWAP and dst_state is not None:
                self.req_states[src] = dst_state

    def _append_recent_text(self, state: RequestState, piece: str) -> None:
        if not piece:
            return
        state.recent_text += piece
        state.recent_text_lower += piece.lower()
        if len(state.recent_text) > self.RECENT_TEXT_MAX_CHARS:
            state.recent_text = state.recent_text[-self.RECENT_TEXT_MAX_CHARS :]
            state.recent_text_lower = state.recent_text_lower[-self.RECENT_TEXT_MAX_CHARS :]

    def _longest_suffix_prefix_lower(self, text_lower: str, target_lower: str) -> int:
        max_k = min(len(text_lower), len(target_lower))
        for k in range(max_k, 0, -1):
            if text_lower.endswith(target_lower[:k]):
                return k
        return 0

    def _is_tag_boundary(self, text: str, tag_index: int) -> bool:
        i = tag_index - 1
        while i >= 0 and text[i] in (" ", "\t", "\r"):
            i -= 1
        if i < 0:
            return True
        prev_char = text[i]
        return prev_char in {"\n", ">"}

    def _restrict_logits_to_ids(self, logits_row: torch.Tensor, ids: set[int] | list[int]) -> None:
        if not ids:
            return
        allowed = list(ids)
        values = logits_row[allowed].clone()
        logits_row.fill_(float("-inf"))
        logits_row[allowed] = values

    def _decode_token(self, token_id: int) -> str:
        # decode one token; keep special tokens; avoid cleanup to preserve exact chars
        try:
            return self.tokenizer.decode(
                [token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
        except TypeError:
            # some tokenizers don’t accept these kwargs
            return self.tokenizer.decode([token_id])

    def _update_request_state(self, state: RequestState) -> None:
        """
        Incrementally:
        - decode only new tokens into a rolling text buffer
        - detect <orders> / </orders> by substring search (robust to BPE merges)
        - maintain trie cursor while inside <orders>
        - if "</orders>" is partially started, force-complete it
        """
        tokens = state.output_token_ids
        cur_len = len(tokens)
        if cur_len <= state.last_processed_len:
            return

        for i in range(state.last_processed_len, cur_len):
            tok = tokens[i]
            piece = self._decode_token(tok)
            self._append_recent_text(state, piece)

            # If we are forcing the remainder of </orders>, advance force pointer
            if state.force_end_ids is not None:
                # Masking should have ensured this matches, but be defensive.
                if (
                    state.force_end_pos < len(state.force_end_ids)
                    and tok == state.force_end_ids[state.force_end_pos]
                ):
                    state.force_end_pos += 1
                else:
                    # Something unexpected happened; stop forcing and continue normally.
                    state.force_end_ids = None
                    state.force_end_pos = 0

                if state.force_end_ids is not None and state.force_end_pos >= len(
                    state.force_end_ids
                ):
                    state.force_end_ids = None
                    state.force_end_pos = 0
                    state.is_done = True
                continue

            # Detect entering orders (one-time)
            if not state.in_orders:
                start_idx = state.recent_text_lower.rfind(self.START_TAG_LOWER)
                if start_idx != -1 and self._is_tag_boundary(state.recent_text, start_idx):
                    state.in_orders = True
                    state.current_node = state.root
                    state.current_line_text = ""  # Reset line tracking

                    # Optional: trim buffer to after the start tag to reduce accidental matches
                    # and make suffix-prefix checks less noisy.
                    trim_idx = start_idx + len(self.START_TAG)
                    state.recent_text = state.recent_text[trim_idx:]
                    state.recent_text_lower = state.recent_text_lower[trim_idx:]
                continue

            # If already in orders, detect completion
            if not state.is_done and state.recent_text_lower.rfind(self.END_TAG_LOWER) != -1:
                state.is_done = True
                continue

            if state.is_done:
                continue

            # If the decoded text ends with a partial prefix of END_TAG, force completion.
            # Example: recent_text endswith("</ord") => force "ers>" next.
            k = self._longest_suffix_prefix_lower(state.recent_text_lower, self.END_TAG_LOWER)
            if 0 < k < len(self.END_TAG):
                remaining = self.END_TAG[k:]
                force_ids = self.tokenizer.encode(remaining, add_special_tokens=False)
                if force_ids:
                    state.force_end_ids = force_ids
                    state.force_end_pos = 0
                    continue

            # Track current line text for unit extraction
            if state.in_orders and not state.is_done:
                state.current_line_text += piece

            # Maintain trie cursor using token ids (works with move+"\n" variants too)
            node = state.current_node
            nxt = node.children.get(tok)
            if nxt is not None:
                state.current_node = nxt
            else:
                # dead end: keep node (apply() will mask to escape options)
                pass

            # If the token's decoded text contains a newline and we are at a move end,
            # extract the unit and rebuild the trie excluding it.
            if "\n" in piece and state.current_node.is_end_of_move:
                state.emitted_orders += 1

                # Extract unit from completed line and mark as used
                unit = self._extract_unit_from_move(state.current_line_text)
                if unit and unit in state.valid_moves_dict:
                    state.used_units.add(unit)
                    # Rebuild trie excluding used units
                    state.root = self._build_trie(state.valid_moves_dict, state.used_units)

                # Reset for next line
                state.current_node = state.root
                state.current_line_text = ""

        state.last_processed_len = cur_len

    # -------------------------
    # Logits masking
    # -------------------------

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.req_states:
            return logits

        for idx, state in self.req_states.items():
            if idx >= logits.shape[0]:
                continue

            # Update cached state from new tokens
            self._update_request_state(state)

            # Not inside <orders> yet => allow free generation
            if not state.in_orders:
                continue

            # Finished </orders> => allow free generation
            if state.is_done:
                self._restrict_logits_to_ids(logits[idx], [state.eos_token_id])
                continue

            # If we've emitted enough orders, force close tag
            if (
                state.force_end_ids is None
                and state.expected_orders > 0
                and state.emitted_orders >= state.expected_orders
            ):
                if self.end_tag_ids:
                    state.force_end_ids = self.end_tag_ids
                    state.force_end_pos = 0
                else:
                    # Fallback: only allow EOS
                    self._restrict_logits_to_ids(logits[idx], [state.eos_token_id])
                    continue

            # Forcing remainder of </orders>
            if state.force_end_ids is not None:
                next_id = state.force_end_ids[state.force_end_pos]
                self._restrict_logits_to_ids(logits[idx], [next_id])
                continue

            # ACTIVE constraints via trie
            node = state.current_node
            allowed = set(node.children.keys())

            # If at end-of-move, allow newline (to start next order) and close tag start
            if node.is_end_of_move:
                allowed.add(state.newline_token_id)
                allowed.add(state.eos_token_id)
                if self.end_tag_start_id is not None:
                    allowed.add(self.end_tag_start_id)

            # If at root, allow optional formatting newlines and close tag start
            if node is state.root:
                allowed.add(state.newline_token_id)
                if self.end_tag_start_id is not None:
                    allowed.add(self.end_tag_start_id)

            # Safety escape if dead end
            if not allowed:
                allowed.add(state.eos_token_id)
                if self.end_tag_start_id is not None:
                    allowed.add(self.end_tag_start_id)

            self._restrict_logits_to_ids(logits[idx], allowed)

        return logits
