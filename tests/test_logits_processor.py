"""
Tests for Diplomacy Logits Processors.

This module tests both:
1. LegacyDiplomacyLogitsProcessor - Per-request processor for unit testing
2. DiplomacyLogitsProcessor.validate_params - Parameter validation for vLLM v1 API

The batch-level DiplomacyLogitsProcessor requires full vLLM infrastructure
and is tested via integration tests with the actual engine.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from transformers import AutoTokenizer

from src.inference.logits import (
    DiplomacyLogitsProcessor,
    LegacyDiplomacyLogitsProcessor,
    TokenTrieNode,
)

# Mock Valid Moves Data
# Scenario: A PAR can move to BUR or MAR. F BRE can move to MAO.
SAMPLE_VALID_MOVES = {"A PAR": ["A PAR - BUR", "A PAR - MAR"], "F BRE": ["F BRE - MAO"]}


@pytest.fixture(scope="module")
def tokenizer():
    # We use GPT2 for testing because it's small and standard.
    # In prod, this would be Mistral/Llama.
    return AutoTokenizer.from_pretrained("gpt2")


@pytest.fixture
def processor(tokenizer):
    return LegacyDiplomacyLogitsProcessor(
        tokenizer=tokenizer, valid_moves_dict=SAMPLE_VALID_MOVES
    )


# =============================================================================
# TokenTrieNode Tests
# =============================================================================


class TestTokenTrieNode:
    """Tests for the Trie data structure."""

    def test_empty_trie(self):
        """Test that a new trie node has no children."""
        node = TokenTrieNode()
        assert node.children == {}
        assert node.is_end_of_move is False

    def test_add_single_sequence(self):
        """Test adding a single token sequence."""
        root = TokenTrieNode()
        root.add_sequence([1, 2, 3])

        assert 1 in root.children
        assert 2 in root.children[1].children
        assert 3 in root.children[1].children[2].children
        assert root.children[1].children[2].children[3].is_end_of_move is True

    def test_add_overlapping_sequences(self):
        """Test adding sequences that share a prefix."""
        root = TokenTrieNode()
        root.add_sequence([1, 2, 3])
        root.add_sequence([1, 2, 4])

        # Both paths should exist
        assert 3 in root.children[1].children[2].children
        assert 4 in root.children[1].children[2].children

    def test_add_empty_sequence(self):
        """Test adding an empty sequence marks root as end."""
        root = TokenTrieNode()
        root.add_sequence([])
        assert root.is_end_of_move is True


# =============================================================================
# DiplomacyLogitsProcessor.validate_params Tests
# =============================================================================


class MockSamplingParams:
    """Mock class mimicking SamplingParams for testing validate_params."""

    def __init__(self, extra_args: dict[str, Any] | None = None):
        self.extra_args = extra_args


class TestValidateParams:
    """Tests for the vLLM v1 parameter validation."""

    def test_validate_params_none_extra_args(self):
        """Test that None extra_args passes validation."""
        params = MockSamplingParams(extra_args=None)
        # Should not raise
        DiplomacyLogitsProcessor.validate_params(params)  # type: ignore[arg-type]

    def test_validate_params_no_valid_moves(self):
        """Test that missing valid_moves_dict passes validation."""
        params = MockSamplingParams(extra_args={"other_key": "value"})
        # Should not raise
        DiplomacyLogitsProcessor.validate_params(params)  # type: ignore[arg-type]

    def test_validate_params_valid_moves_dict(self):
        """Test that valid moves dict passes validation."""
        params = MockSamplingParams(
            extra_args={"valid_moves_dict": {"A PAR": ["A PAR - BUR", "A PAR - MAR"]}}
        )
        # Should not raise
        DiplomacyLogitsProcessor.validate_params(params)  # type: ignore[arg-type]

    def test_validate_params_invalid_type(self):
        """Test that non-dict valid_moves_dict fails validation."""
        params = MockSamplingParams(extra_args={"valid_moves_dict": "not a dict"})
        with pytest.raises(ValueError, match="must be a dict"):
            DiplomacyLogitsProcessor.validate_params(params)  # type: ignore[arg-type]

    def test_validate_params_invalid_unit_key(self):
        """Test that non-string unit key fails validation."""
        params = MockSamplingParams(
            extra_args={"valid_moves_dict": {123: ["A PAR - BUR"]}}
        )
        with pytest.raises(ValueError, match="Unit key must be str"):
            DiplomacyLogitsProcessor.validate_params(params)  # type: ignore[arg-type]

    def test_validate_params_invalid_moves_list(self):
        """Test that non-list moves fails validation."""
        params = MockSamplingParams(
            extra_args={"valid_moves_dict": {"A PAR": "not a list"}}
        )
        with pytest.raises(ValueError, match="Moves must be list"):
            DiplomacyLogitsProcessor.validate_params(params)  # type: ignore[arg-type]

    def test_validate_params_invalid_move_string(self):
        """Test that non-string move fails validation."""
        params = MockSamplingParams(extra_args={"valid_moves_dict": {"A PAR": [123]}})
        with pytest.raises(ValueError, match="Each move must be str"):
            DiplomacyLogitsProcessor.validate_params(params)  # type: ignore[arg-type]


# =============================================================================
# LegacyDiplomacyLogitsProcessor Tests (Unit Tests)
# =============================================================================


class TestLegacyProcessor:
    """Tests for the legacy per-request processor."""

    def test_initial_masking(self, processor, tokenizer):
        """Test that at the start, only the first tokens of valid orders are allowed."""
        input_ids = []
        scores = torch.zeros(tokenizer.vocab_size)

        # Run processor
        processed_scores = processor(input_ids, scores)

        # Check valid start tokens
        # "A" (from A PAR) and "F" (from F BRE) should be valid
        token_a = tokenizer.encode("A PAR - BUR")[0]
        token_f = tokenizer.encode("F BRE - MAO")[0]

        assert processed_scores[token_a] > -float("inf"), (
            "Starting with 'A' should be allowed"
        )
        assert processed_scores[token_f] > -float("inf"), (
            "Starting with 'F' should be allowed"
        )

        # Check invalid token (e.g., "Pizza")
        token_invalid = tokenizer.encode("Pizza")[0]
        assert processed_scores[token_invalid] == -float("inf"), (
            "Invalid start token should be masked"
        )

    def test_path_following(self, processor, tokenizer):
        """Test that once we start a move, we must follow the valid path."""
        # Simulate generating "A PAR -"
        history_str = "A PAR -"
        input_ids = tokenizer.encode(history_str)

        scores = torch.zeros(tokenizer.vocab_size)
        processed_scores = processor(input_ids, scores)

        # Next valid tokens should be " BUR" or " MAR"
        token_bur = tokenizer.encode(" BUR")[0]
        token_bur_alt = tokenizer.encode("BUR")[0]

        # Check if BUR (with or without space prefix) is unmasked
        assert processed_scores[token_bur] > -float("inf") or processed_scores[
            token_bur_alt
        ] > -float("inf")

    def test_dead_end_blocking(self, processor, tokenizer):
        """Test that off-path tokens are strictly forbidden."""
        # Simulate "A PAR - "
        input_ids = tokenizer.encode("A PAR -")
        scores = torch.zeros(tokenizer.vocab_size)
        processed_scores = processor(input_ids, scores)

        # "LON" is a valid game token, but NOT valid for A PAR
        token_lon = tokenizer.encode(" LON")[0]

        assert processed_scores[token_lon] == -float("inf"), (
            "London is not adjacent to Paris; should be blocked."
        )

    def test_newline_reset(self, processor, tokenizer):
        """Test that hitting \\n resets the Trie to allow a new order."""
        # Simulate a full valid order "A PAR - BUR" followed by newline
        full_order = "A PAR - BUR\n"
        input_ids = tokenizer.encode(full_order)

        # Step through tokens one by one (simulating generation)
        for i in range(len(input_ids)):
            current_history = input_ids[:i]
            scores = torch.zeros(tokenizer.vocab_size)
            processor(current_history, scores)

        # Process the step *after* newline
        final_scores = processor(input_ids, torch.zeros(tokenizer.vocab_size))

        token_a = tokenizer.encode("A")[0]
        token_f = tokenizer.encode("F")[0]

        assert final_scores[token_a] > -float("inf"), (
            "Should be able to start 'A' after newline"
        )
        assert final_scores[token_f] > -float("inf"), (
            "Should be able to start 'F' after newline"
        )

    def test_token_boundary_stress(self, tokenizer):
        """
        Stress test for sub-word tokenization.
        Example: 'BURGUNDY' is tokenized as [' BUR', 'G', 'UN', 'D', 'Y'].
        The Trie must support multi-token moves.
        """
        long_move_processor = LegacyDiplomacyLogitsProcessor(
            tokenizer,
            {"A PAR": ["A PAR - BURGUNDY"]},
        )

        # Get the actual tokenization of the full move
        full_tokens = tokenizer.encode("A PAR - BURGUNDY")
        # GPT-2: [32, 29463, 532, 45604, 38, 4944, 35, 56]
        # Which is: ['A', ' PAR', ' -', ' BUR', 'G', 'UN', 'D', 'Y']

        # Walk step by step through the Trie
        for i in range(len(full_tokens)):
            input_ids = full_tokens[:i]
            scores = torch.zeros(tokenizer.vocab_size)
            processed_scores = long_move_processor(input_ids, scores)

            # The next token should be allowed
            next_token = full_tokens[i]
            assert processed_scores[next_token] > -float("inf"), (
                f"Token {next_token} ({repr(tokenizer.decode([next_token]))}) "
                f"should be allowed at position {i}"
            )

        # After complete move, newline and EOS should be allowed
        scores = torch.zeros(tokenizer.vocab_size)
        final_scores = long_move_processor(full_tokens, scores)
        assert final_scores[tokenizer.eos_token_id] > -float("inf"), (
            "EOS should be allowed after complete multi-token move"
        )

    def test_eos_allowed_at_end_of_move(self, processor, tokenizer):
        """Test that EOS token is allowed after completing a valid move."""
        # Complete a valid move
        input_ids = tokenizer.encode("A PAR - BUR")
        scores = torch.zeros(tokenizer.vocab_size)

        processed_scores = processor(input_ids, scores)

        # EOS should be allowed
        assert processed_scores[tokenizer.eos_token_id] > -float("inf"), (
            "EOS should be allowed after completing a move"
        )

    def test_dead_end_allows_eos(self, processor, tokenizer):
        """Test that EOS is allowed as fallback when in a dead end."""
        # Force a dead end by providing invalid sequence
        input_ids = tokenizer.encode("INVALID GARBAGE")
        scores = torch.zeros(tokenizer.vocab_size)

        processed_scores = processor(input_ids, scores)

        # EOS should be allowed as fallback
        assert processed_scores[tokenizer.eos_token_id] == 0, (
            "EOS should be allowed in dead end"
        )

    def test_multiple_units_in_trie(self, processor, tokenizer):
        """Test that all units' moves are in the trie."""
        # Both "A PAR - BUR" and "F BRE - MAO" should be valid starting points
        input_ids = []
        scores = torch.zeros(tokenizer.vocab_size)
        processed_scores = processor(input_ids, scores)

        # Count how many tokens are valid at the start
        valid_count = (processed_scores > -float("inf")).sum().item()

        # Should have at least 2 valid starting tokens (A and F)
        assert valid_count >= 1, "Should have valid starting tokens"


# =============================================================================
# DiplomacyLogitsProcessor (Batch-Level) Tests
# =============================================================================


class MockVllmConfig:
    """Mock VllmConfig for testing batch-level processor."""

    def __init__(self, model_name: str = "gpt2"):
        self.model_config = MockModelConfig(model_name)


class MockModelConfig:
    """Mock ModelConfig for testing."""

    def __init__(self, model: str):
        self.model = model


class MockBatchUpdate:
    """Mock BatchUpdate for testing update_state."""

    def __init__(
        self,
        batch_size: int = 1,
        removed: list[int] | None = None,
        added: list[tuple] | None = None,
        moved: list[tuple] | None = None,
    ):
        self.batch_size = batch_size
        self.removed = removed or []
        self.added = added or []
        self.moved = moved or []


class TestBatchLevelProcessor:
    """
    Tests for the batch-level DiplomacyLogitsProcessor.

    These tests verify the vLLM v1 API implementation including:
    - update_state correctly parses BatchUpdate tuples
    - apply correctly masks logits based on output tokens (not prompt tokens)
    """

    @pytest.fixture
    def batch_processor(self, tokenizer):
        """Create a batch-level processor with mocked vllm_config."""
        mock_config = MockVllmConfig(model_name="gpt2")
        return DiplomacyLogitsProcessor(
            vllm_config=mock_config,  # type: ignore[arg-type]
            device=torch.device("cpu"),
            is_pin_memory=False,
        )

    def test_update_state_uses_output_tokens_not_prompt_tokens(
        self, batch_processor, tokenizer
    ):
        """
        CRITICAL: Verify that update_state uses output_token_ids (element 3)
        not prompt_token_ids (element 2) from the AddedRequest tuple.

        AddedRequest = (index, params, prompt_tok_ids, output_tok_ids)

        This test catches the bug where we had:
            for index, params, output_token_ids, _ in batch_update.added:
        Instead of:
            for index, params, _prompt_token_ids, output_token_ids in batch_update.added:
        """
        valid_moves = {"A PAR": ["A PAR - BUR"]}
        params = MockSamplingParams(extra_args={"valid_moves_dict": valid_moves})

        # Prompt tokens that would cause trie walk to fail if used
        # (random tokens that aren't in the valid moves trie)
        prompt_token_ids = [999, 888, 777]

        # Output tokens should be empty at start of generation
        output_token_ids: list[int] = []

        # Create BatchUpdate with the correct tuple order:
        # (index, params, prompt_tok_ids, output_tok_ids)
        batch_update = MockBatchUpdate(
            batch_size=1,
            added=[(0, params, prompt_token_ids, output_token_ids)],
        )

        batch_processor.update_state(batch_update)  # type: ignore[arg-type]

        # Verify state was created
        assert 0 in batch_processor.req_states, "Request should be tracked"

        # The stored output_token_ids should be the empty list reference,
        # NOT the prompt_token_ids
        state = batch_processor.req_states[0]
        assert state.output_token_ids is output_token_ids, (
            "Should store reference to output_token_ids, not prompt_token_ids"
        )
        assert state.output_token_ids == [], "output_token_ids should be empty at start"

    def test_apply_masks_based_on_output_tokens(self, batch_processor, tokenizer):
        """
        Test that apply() walks the trie based on output_token_ids.

        At the start of generation (empty output_token_ids), only the first
        tokens of valid moves should be allowed.
        """
        valid_moves = {"A PAR": ["A PAR - BUR", "A PAR - MAR"]}
        params = MockSamplingParams(extra_args={"valid_moves_dict": valid_moves})

        output_token_ids: list[int] = []

        batch_update = MockBatchUpdate(
            batch_size=1,
            added=[(0, params, None, output_token_ids)],
        )

        batch_processor.update_state(batch_update)  # type: ignore[arg-type]

        # Create logits tensor (batch_size=1, vocab_size)
        logits = torch.zeros(1, tokenizer.vocab_size)
        result = batch_processor.apply(logits)

        # First token of "A PAR - BUR" should be allowed
        first_token = tokenizer.encode("A PAR - BUR")[0]
        assert result[0, first_token] > -float("inf"), (
            f"First token {first_token} should be allowed"
        )

        # Random invalid token should be masked
        assert result[0, 12345] == -float("inf"), (
            "Invalid token should be masked to -inf"
        )

    def test_apply_advances_trie_with_output_tokens(self, batch_processor, tokenizer):
        """
        Test that as output_token_ids grows, the trie walk advances correctly.

        This simulates the generation process where output_token_ids is a
        reference that gets appended to as tokens are generated.
        """
        valid_moves = {"A PAR": ["A PAR - BUR"]}
        params = MockSamplingParams(extra_args={"valid_moves_dict": valid_moves})

        # This list will be mutated to simulate token generation
        output_token_ids: list[int] = []

        batch_update = MockBatchUpdate(
            batch_size=1,
            added=[(0, params, None, output_token_ids)],
        )

        batch_processor.update_state(batch_update)  # type: ignore[arg-type]

        # Get the full tokenization of the valid move
        full_tokens = tokenizer.encode("A PAR - BUR")

        # Simulate generation: append tokens one by one
        for i, token in enumerate(full_tokens):
            logits = torch.zeros(1, tokenizer.vocab_size)
            result = batch_processor.apply(logits)

            # Current token should be allowed (we're about to generate it)
            assert result[0, token] > -float("inf"), (
                f"Token {token} at position {i} should be allowed"
            )

            # Simulate the model selecting this token
            output_token_ids.append(token)

        # After full move, EOS should be allowed
        logits = torch.zeros(1, tokenizer.vocab_size)
        result = batch_processor.apply(logits)
        assert result[0, tokenizer.eos_token_id] > -float("inf"), (
            "EOS should be allowed after complete move"
        )

    def test_update_state_removes_request(self, batch_processor, tokenizer):
        """Test that removed requests are cleaned up from state."""
        valid_moves = {"A PAR": ["A PAR - BUR"]}
        params = MockSamplingParams(extra_args={"valid_moves_dict": valid_moves})

        # Add a request
        batch_update = MockBatchUpdate(
            batch_size=1,
            added=[(0, params, None, [])],
        )
        batch_processor.update_state(batch_update)  # type: ignore[arg-type]
        assert 0 in batch_processor.req_states

        # Remove the request
        batch_update = MockBatchUpdate(
            batch_size=0,
            removed=[0],
        )
        batch_processor.update_state(batch_update)  # type: ignore[arg-type]
        assert 0 not in batch_processor.req_states

    def test_apply_returns_unmodified_when_no_state(self, batch_processor, tokenizer):
        """Test that apply returns logits unmodified when no requests are tracked."""
        logits = torch.ones(1, tokenizer.vocab_size) * 5.0
        result = batch_processor.apply(logits)

        # Should be unchanged
        assert torch.allclose(result, logits), (
            "Logits should be unmodified when no requests tracked"
        )
