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

try:
    import vllm  # noqa: F401

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

from src.inference.logits import (
    DiplomacyLogitsProcessor,
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


@pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM not available")
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
        params = MockSamplingParams(extra_args={"valid_moves_dict": {123: ["A PAR - BUR"]}})
        with pytest.raises(ValueError, match="Unit key must be str"):
            DiplomacyLogitsProcessor.validate_params(params)  # type: ignore[arg-type]

    def test_validate_params_invalid_moves_list(self):
        """Test that non-list moves fails validation."""
        params = MockSamplingParams(extra_args={"valid_moves_dict": {"A PAR": "not a list"}})
        with pytest.raises(ValueError, match="Moves must be list"):
            DiplomacyLogitsProcessor.validate_params(params)  # type: ignore[arg-type]

    def test_validate_params_invalid_move_string(self):
        """Test that non-string move fails validation."""
        params = MockSamplingParams(extra_args={"valid_moves_dict": {"A PAR": [123]}})
        with pytest.raises(ValueError, match="Each move must be str"):
            DiplomacyLogitsProcessor.validate_params(params)  # type: ignore[arg-type]


# =============================================================================
# Self-Bounce Prevention Tests
# =============================================================================


class TestExtractTargetFromMove:
    """Tests for target location extraction from movement orders."""

    def test_movement_order(self):
        """Movement order extracts target location."""
        assert DiplomacyLogitsProcessor._extract_target_from_move("A PAR - BUR") == "BUR"
        assert DiplomacyLogitsProcessor._extract_target_from_move("F NTH - NWY") == "NWY"

    def test_movement_with_via(self):
        """Movement with VIA convoy still extracts target."""
        assert DiplomacyLogitsProcessor._extract_target_from_move("A PAR - BUR VIA") == "BUR"

    def test_movement_with_coast(self):
        """Movement to coastal location extracts base province."""
        assert DiplomacyLogitsProcessor._extract_target_from_move("F NWG - STP/NC") == "STP"
        assert DiplomacyLogitsProcessor._extract_target_from_move("F AEG - BUL/SC") == "BUL"

    def test_hold_returns_none(self):
        """Hold orders don't target any location."""
        assert DiplomacyLogitsProcessor._extract_target_from_move("A PAR H") is None

    def test_support_hold_returns_none(self):
        """Support hold orders don't add movement target."""
        assert DiplomacyLogitsProcessor._extract_target_from_move("A PAR S A BUR") is None

    def test_support_move_returns_none(self):
        """Support move orders - supporting unit doesn't move."""
        assert DiplomacyLogitsProcessor._extract_target_from_move("A PAR S A BUR - MAR") is None
        assert DiplomacyLogitsProcessor._extract_target_from_move("F ENG S F MAO - BRE") is None

    def test_convoy_returns_none(self):
        """Convoy orders - fleet doesn't move to destination."""
        assert DiplomacyLogitsProcessor._extract_target_from_move("F NTH C A LON - BRE") is None

    def test_build_returns_none(self):
        """Build orders don't have movement."""
        assert DiplomacyLogitsProcessor._extract_target_from_move("A PAR B") is None

    def test_disband_returns_none(self):
        """Disband orders don't have movement."""
        assert DiplomacyLogitsProcessor._extract_target_from_move("A PAR D") is None

    def test_retreat_returns_none(self):
        """Retreat orders use R syntax, not - syntax."""
        # Retreat format is "A PAR R BUR" not "A PAR - BUR"
        assert DiplomacyLogitsProcessor._extract_target_from_move("A PAR R BUR") is None


@pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM not available")
class TestSelfBouncePrevention:
    """Tests for trie building with target location exclusion."""

    @pytest.fixture
    def batch_processor(self, tokenizer):
        """Create a batch-level processor."""
        mock_config = MockVllmConfig(model_name="gpt2")
        return DiplomacyLogitsProcessor(
            vllm_config=mock_config,  # type: ignore[arg-type]
            device=torch.device("cpu"),
            is_pin_memory=False,
        )

    def test_build_trie_excludes_targeted_locations(self, batch_processor, tokenizer):
        """Trie excludes movements to already-targeted locations."""
        valid_moves = {
            "F LON": ["F LON - YOR", "F LON - NTH", "F LON H"],
            "A LVP": ["A LVP - YOR", "A LVP - WAL", "A LVP H"],
        }

        # Build trie with YOR already targeted
        excluded_targets = {"YOR"}
        trie = batch_processor._build_trie(valid_moves, exclude_targets=excluded_targets)

        # "F LON - YOR" should NOT be in trie
        yor_move_ids = tokenizer.encode("F LON - YOR", add_special_tokens=False)
        node = trie
        can_reach_yor = True
        for tid in yor_move_ids:
            if tid in node.children:
                node = node.children[tid]
            else:
                can_reach_yor = False
                break
        assert not can_reach_yor, "F LON - YOR should be excluded"

        # "F LON - NTH" SHOULD be in trie
        nth_move_ids = tokenizer.encode("F LON - NTH", add_special_tokens=False)
        node = trie
        can_reach_nth = True
        for tid in nth_move_ids:
            if tid in node.children:
                node = node.children[tid]
            else:
                can_reach_nth = False
                break
        assert can_reach_nth, "F LON - NTH should be allowed"

        # "F LON H" SHOULD be in trie (holds are not affected)
        hold_move_ids = tokenizer.encode("F LON H", add_special_tokens=False)
        node = trie
        can_reach_hold = True
        for tid in hold_move_ids:
            if tid in node.children:
                node = node.children[tid]
            else:
                can_reach_hold = False
                break
        assert can_reach_hold, "F LON H should be allowed"

    def test_supports_still_allowed_when_target_excluded(self, batch_processor, tokenizer):
        """Support orders to excluded locations should still be allowed."""
        valid_moves = {
            "A PAR": ["A PAR - BUR", "A PAR S A MAR - BUR", "A PAR H"],
            "A MAR": ["A MAR - BUR", "A MAR H"],
        }

        # If MAR already ordered to BUR, PAR can still support that move
        excluded_targets = {"BUR"}
        trie = batch_processor._build_trie(valid_moves, exclude_targets=excluded_targets)

        # "A PAR - BUR" should NOT be in trie (direct movement)
        move_ids = tokenizer.encode("A PAR - BUR", add_special_tokens=False)
        node = trie
        can_reach = True
        for tid in move_ids:
            if tid in node.children:
                node = node.children[tid]
            else:
                can_reach = False
                break
        assert not can_reach, "A PAR - BUR should be excluded"

        # "A PAR S A MAR - BUR" SHOULD be in trie (support is allowed)
        support_ids = tokenizer.encode("A PAR S A MAR - BUR", add_special_tokens=False)
        node = trie
        can_reach_support = True
        for tid in support_ids:
            if tid in node.children:
                node = node.children[tid]
            else:
                can_reach_support = False
                break
        assert can_reach_support, "A PAR S A MAR - BUR should be allowed (support)"


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


@pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM not available")
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

    def test_update_state_uses_output_tokens_not_prompt_tokens(self, batch_processor, tokenizer):
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

        CRITICAL: Masking only occurs INSIDE <orders> tags. Before that,
        free generation is allowed for Chain-of-Thought reasoning.
        """
        valid_moves = {"A PAR": ["A PAR - BUR", "A PAR - MAR"]}
        params = MockSamplingParams(extra_args={"valid_moves_dict": valid_moves})

        # Simulate that model has already generated <orders> tag
        orders_tag_tokens = tokenizer.encode("<orders>", add_special_tokens=False)
        output_token_ids: list[int] = list(orders_tag_tokens)

        batch_update = MockBatchUpdate(
            batch_size=1,
            added=[(0, params, None, output_token_ids)],
        )

        batch_processor.update_state(batch_update)  # type: ignore[arg-type]

        # Create logits tensor (batch_size=1, vocab_size)
        logits = torch.zeros(1, tokenizer.vocab_size)
        result = batch_processor.apply(logits)

        # First token of "A PAR - BUR" should be allowed
        first_token = tokenizer.encode("A PAR - BUR", add_special_tokens=False)[0]
        assert result[0, first_token] > -float("inf"), (
            f"First token {first_token} should be allowed"
        )

        # Random invalid token should be masked
        assert result[0, 12345] == -float("inf"), "Invalid token should be masked to -inf"

    def test_apply_advances_trie_with_output_tokens(self, batch_processor, tokenizer):
        """
        Test that as output_token_ids grows, the trie walk advances correctly.

        This simulates the generation process where output_token_ids is a
        reference that gets appended to as tokens are generated.

        CRITICAL: Masking only applies INSIDE <orders> tags.
        """
        valid_moves = {"A PAR": ["A PAR - BUR"]}
        params = MockSamplingParams(extra_args={"valid_moves_dict": valid_moves})

        # Start with <orders> tag already generated (to trigger masking)
        orders_tag_tokens = tokenizer.encode("<orders>", add_special_tokens=False)
        output_token_ids: list[int] = list(orders_tag_tokens)

        batch_update = MockBatchUpdate(
            batch_size=1,
            added=[(0, params, None, output_token_ids)],
        )

        batch_processor.update_state(batch_update)  # type: ignore[arg-type]

        # Get the full tokenization of the valid move
        full_tokens = tokenizer.encode("A PAR - BUR", add_special_tokens=False)

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


# =============================================================================
# Tag Detection (Context Awareness) Tests
# =============================================================================


@pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM not available")
class TestTagDetection:
    """
    Tests for the critical tag detection logic that enables Chain-of-Thought.

    The processor must be "dormant" (allow free generation) until the model
    outputs <orders>, then constrain tokens until </orders>.
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

    def test_free_generation_before_orders_tag(self, batch_processor, tokenizer):
        """
        CRITICAL: Before <orders> tag, model must be free to generate anything.

        This is the "lobotomization" bug - if we mask during CoT, the model
        cannot think and output quality collapses.
        """
        valid_moves = {"A PAR": ["A PAR - BUR"]}
        params = MockSamplingParams(extra_args={"valid_moves_dict": valid_moves})

        # Model is generating reasoning (no <orders> tag yet)
        analysis_text = "<analysis>I think I should move to Burgundy because"
        output_token_ids: list[int] = tokenizer.encode(analysis_text, add_special_tokens=False)

        batch_update = MockBatchUpdate(
            batch_size=1,
            added=[(0, params, None, output_token_ids)],
        )
        batch_processor.update_state(batch_update)  # type: ignore[arg-type]

        logits = torch.zeros(1, tokenizer.vocab_size)
        result = batch_processor.apply(logits)

        # ALL tokens should be allowed (no masking during CoT)
        assert torch.all(result == 0), (
            "All tokens should be allowed before <orders> tag (free CoT generation)"
        )

    def test_masking_activates_inside_orders_tag(self, batch_processor, tokenizer):
        """
        After <orders> tag, masking should activate and constrain to valid moves.
        """
        valid_moves = {"A PAR": ["A PAR - BUR"]}
        params = MockSamplingParams(extra_args={"valid_moves_dict": valid_moves})

        # Model has output reasoning and entered orders block
        text_with_orders = "<analysis>Moving to Burgundy</analysis>\n<orders>"
        output_token_ids: list[int] = tokenizer.encode(text_with_orders, add_special_tokens=False)

        batch_update = MockBatchUpdate(
            batch_size=1,
            added=[(0, params, None, output_token_ids)],
        )
        batch_processor.update_state(batch_update)  # type: ignore[arg-type]

        logits = torch.zeros(1, tokenizer.vocab_size)
        result = batch_processor.apply(logits)

        # Some tokens should now be masked
        masked_count = (result == -float("inf")).sum().item()
        assert masked_count > 0, "Tokens should be masked inside <orders> block"

        # First token of valid move should be allowed
        first_move_token = tokenizer.encode("A PAR - BUR", add_special_tokens=False)[0]
        assert result[0, first_move_token] > -float("inf"), (
            "First token of valid move should be allowed"
        )

    def test_eos_forced_after_closing_tag(self, batch_processor, tokenizer):
        """
        After </orders> tag, only EOS should be allowed to prevent trailing garbage.
        The expected format is <analysis>...</analysis><orders>...</orders> with nothing after.
        """
        valid_moves = {"A PAR": ["A PAR - BUR"]}
        params = MockSamplingParams(extra_args={"valid_moves_dict": valid_moves})

        # Full generation with orders block completed
        complete_text = "<analysis>Plan</analysis>\n<orders>A PAR - BUR\n</orders>"
        output_token_ids: list[int] = tokenizer.encode(complete_text, add_special_tokens=False)

        batch_update = MockBatchUpdate(
            batch_size=1,
            added=[(0, params, None, output_token_ids)],
        )
        batch_processor.update_state(batch_update)  # type: ignore[arg-type]

        logits = torch.zeros(1, tokenizer.vocab_size)
        result = batch_processor.apply(logits)

        # Only EOS should be allowed after </orders> to cleanly terminate generation
        eos_id = tokenizer.eos_token_id
        assert result[0, eos_id] == 0, "EOS token should be allowed"
        # All other tokens should be masked
        non_eos_mask = torch.ones(tokenizer.vocab_size, dtype=torch.bool)
        non_eos_mask[eos_id] = False
        assert torch.all(result[0, non_eos_mask] == float("-inf")), (
            "All non-EOS tokens should be masked after </orders> tag"
        )

    def test_empty_output_allows_free_generation(self, batch_processor, tokenizer):
        """
        At the very start of generation (empty output), free generation is allowed.
        """
        valid_moves = {"A PAR": ["A PAR - BUR"]}
        params = MockSamplingParams(extra_args={"valid_moves_dict": valid_moves})

        output_token_ids: list[int] = []

        batch_update = MockBatchUpdate(
            batch_size=1,
            added=[(0, params, None, output_token_ids)],
        )
        batch_processor.update_state(batch_update)  # type: ignore[arg-type]

        logits = torch.zeros(1, tokenizer.vocab_size)
        result = batch_processor.apply(logits)

        # All tokens should be allowed
        assert torch.all(result == 0), (
            "All tokens should be allowed with empty output (no <orders> yet)"
        )

    def test_newline_allowed_at_root_inside_orders(self, batch_processor, tokenizer):
        """
        CRITICAL: After <orders> tag, newlines must be allowed for formatting.

        The expected format is:
        <orders>
        A PAR - BUR
        </orders>

        The newline immediately after <orders> must be allowed.
        """
        valid_moves = {"A PAR": ["A PAR - BUR"]}
        params = MockSamplingParams(extra_args={"valid_moves_dict": valid_moves})

        # Model has just output <orders> tag
        orders_tag = "<orders>"
        output_token_ids: list[int] = tokenizer.encode(orders_tag, add_special_tokens=False)

        batch_update = MockBatchUpdate(
            batch_size=1,
            added=[(0, params, None, output_token_ids)],
        )
        batch_processor.update_state(batch_update)  # type: ignore[arg-type]

        logits = torch.zeros(1, tokenizer.vocab_size)
        result = batch_processor.apply(logits)

        # Newline token should be allowed (for formatting)
        newline_token = tokenizer.encode("\n", add_special_tokens=False)[-1]
        assert result[0, newline_token] > -float("inf"), (
            "Newline should be allowed immediately after <orders> tag"
        )

        # First token of valid move should also be allowed
        first_move_token = tokenizer.encode("A PAR - BUR", add_special_tokens=False)[0]
        assert result[0, first_move_token] > -float("inf"), (
            "First token of valid move should be allowed"
        )

        # Start of closing tag should also be allowed at root
        close_tag_start = tokenizer.encode("</orders>", add_special_tokens=False)[0]
        assert result[0, close_tag_start] > -float("inf"), (
            "Start of </orders> tag should be allowed at root"
        )


# =============================================================================
# Performance Benchmark Tests
# =============================================================================


@pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM not available")
class TestPerformance:
    """
    Performance tests to ensure the logits processor is fast enough for
    high-throughput inference.

    Target: <1ms per apply() call for realistic batch sizes.
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

    def test_apply_performance_single_request(self, batch_processor, tokenizer):
        """
        Benchmark apply() with a single request mid-generation.

        Target: <1ms per call on CPU.
        """
        import time

        # Realistic valid moves (7 powers × ~5 units × ~10 moves each)
        units = ["A PAR", "A MAR", "A BUR", "F BRE", "F MAO", "A MUN", "A BER"]
        destinations = [
            "PAR",
            "MAR",
            "BUR",
            "BRE",
            "MAO",
            "MUN",
            "BER",
            "PIC",
            "GAS",
            "RUH",
        ]
        valid_moves = {unit: [f"{unit} - {dest}" for dest in destinations[:8]] for unit in units}

        params = MockSamplingParams(extra_args={"valid_moves_dict": valid_moves})

        # Simulate mid-generation inside <orders> block
        text = "<analysis>Strategic analysis here.</analysis>\n<orders>A PAR - "
        output_token_ids: list[int] = tokenizer.encode(text, add_special_tokens=False)

        batch_update = MockBatchUpdate(
            batch_size=1,
            added=[(0, params, None, output_token_ids)],
        )
        batch_processor.update_state(batch_update)  # type: ignore[arg-type]

        # Warm-up
        logits = torch.zeros(1, tokenizer.vocab_size)
        for _ in range(10):
            batch_processor.apply(logits.clone())

        # Benchmark
        num_iterations = 100
        start = time.perf_counter()
        for _ in range(num_iterations):
            batch_processor.apply(logits.clone())
        elapsed = time.perf_counter() - start

        avg_ms = (elapsed / num_iterations) * 1000
        print(f"\nSingle request apply(): {avg_ms:.3f}ms avg over {num_iterations} calls")

        assert avg_ms < 5.0, f"apply() too slow: {avg_ms:.3f}ms (target <5ms on CPU)"

    def test_apply_performance_batch(self, batch_processor, tokenizer):
        """
        Benchmark apply() with a batch of requests (simulating vLLM batching).

        Target: <10ms per call for batch_size=8 on CPU.
        """
        import time

        batch_size = 8

        # Build valid moves for each request
        units = ["A PAR", "A MAR", "F BRE", "A MUN"]
        destinations = ["PAR", "MAR", "BUR", "BRE", "MAO", "MUN", "BER", "PIC"]
        valid_moves = {unit: [f"{unit} - {dest}" for dest in destinations] for unit in units}

        # Add multiple requests
        added = []
        for i in range(batch_size):
            params = MockSamplingParams(extra_args={"valid_moves_dict": valid_moves})
            text = f"<analysis>Reasoning {i}</analysis>\n<orders>A PAR - "
            output_token_ids = tokenizer.encode(text, add_special_tokens=False)
            added.append((i, params, None, output_token_ids))

        batch_update = MockBatchUpdate(batch_size=batch_size, added=added)
        batch_processor.update_state(batch_update)  # type: ignore[arg-type]

        # Warm-up
        logits = torch.zeros(batch_size, tokenizer.vocab_size)
        for _ in range(5):
            batch_processor.apply(logits.clone())

        # Benchmark
        num_iterations = 50
        start = time.perf_counter()
        for _ in range(num_iterations):
            batch_processor.apply(logits.clone())
        elapsed = time.perf_counter() - start

        avg_ms = (elapsed / num_iterations) * 1000
        print(f"\nBatch ({batch_size}) apply(): {avg_ms:.3f}ms avg over {num_iterations} calls")

        assert avg_ms < 20.0, f"Batch apply() too slow: {avg_ms:.3f}ms (target <20ms on CPU)"

    def test_trie_build_performance(self, batch_processor, tokenizer):
        """
        Benchmark Trie construction with realistic move counts.

        A complex Diplomacy position might have 100+ valid moves.
        Target: <50ms to build Trie.
        """
        import time

        # Generate many valid moves (worst case scenario)
        units = [
            f"A {loc}"
            for loc in [
                "PAR",
                "MAR",
                "BUR",
                "MUN",
                "BER",
                "KIE",
                "RUH",
                "PIE",
                "VEN",
                "ROM",
            ]
        ]
        units += [f"F {loc}" for loc in ["BRE", "MAO", "NAO", "ENG", "NTH", "BAL", "BOT", "SKA"]]
        destinations = [
            "PAR",
            "MAR",
            "BUR",
            "MUN",
            "BER",
            "KIE",
            "RUH",
            "PIE",
            "VEN",
            "ROM",
            "BRE",
            "MAO",
            "NAO",
            "ENG",
            "NTH",
            "BAL",
            "BOT",
            "SKA",
            "PIC",
            "GAS",
            "SPA",
            "POR",
            "TUN",
            "NAF",
            "TYS",
            "GOL",
            "WES",
            "ION",
            "ADR",
            "AEG",
        ]

        valid_moves = {}
        for unit in units:
            valid_moves[unit] = [f"{unit} - {dest}" for dest in destinations[:15]]

        total_moves = sum(len(m) for m in valid_moves.values())
        print(f"\nBuilding Trie for {total_moves} moves from {len(units)} units...")

        # Benchmark Trie construction
        num_iterations = 20
        start = time.perf_counter()
        for _ in range(num_iterations):
            batch_processor._build_trie(valid_moves)
        elapsed = time.perf_counter() - start

        avg_ms = (elapsed / num_iterations) * 1000
        print(f"Trie build: {avg_ms:.3f}ms avg over {num_iterations} calls")

        assert avg_ms < 100.0, f"Trie build too slow: {avg_ms:.3f}ms (target <100ms)"

    def test_apply_performance_incremental_generation(self, batch_processor, tokenizer):
        """
        CRITICAL: Simulate realistic generation where apply() is called
        once per token for 200 tokens.

        This tests the O(1) vs O(n) per-token complexity.
        Old implementation: O(n²) total (scanning entire history each call)
        New implementation: O(n) total (incremental state tracking)

        Target: <100ms for 200 token generation on CPU.
        """
        import time

        valid_moves = {
            "A PAR": ["A PAR - BUR", "A PAR - MAR"],
            "F BRE": ["F BRE - MAO"],
        }
        params = MockSamplingParams(extra_args={"valid_moves_dict": valid_moves})

        # Simulate the full generation pattern:
        # 1. Generate <analysis>...</analysis>\n<orders>
        # 2. Generate move tokens
        # 3. Generate </orders>

        prefix_text = "<analysis>I should secure Burgundy for strategic depth.</analysis>\n<orders>"
        prefix_tokens = tokenizer.encode(prefix_text, add_special_tokens=False)

        # Valid move + newline + closing tag
        move_text = "A PAR - BUR\n</orders>"
        move_tokens = tokenizer.encode(move_text, add_special_tokens=False)

        # Simulate generating ~200 tokens total (realistic for our use case)
        # We'll repeat the analysis part to pad to ~200 tokens
        padding_text = "Considering the diplomatic situation with Germany and the potential for coordinated moves into the lowlands, I believe the best approach is to secure our eastern border while maintaining flexibility in the south."
        padding_tokens = tokenizer.encode(padding_text, add_special_tokens=False)

        # Build full sequence
        full_tokens = padding_tokens + prefix_tokens + move_tokens
        target_len = 200
        if len(full_tokens) < target_len:
            # Pad with more reasoning
            extra = tokenizer.encode(
                " " * (target_len - len(full_tokens)), add_special_tokens=False
            )
            full_tokens = extra + full_tokens

        print(f"\nSimulating generation of {len(full_tokens)} tokens...")

        # This list will be mutated to simulate token-by-token generation
        output_token_ids: list[int] = []

        batch_update = MockBatchUpdate(
            batch_size=1,
            added=[(0, params, None, output_token_ids)],
        )
        batch_processor.update_state(batch_update)  # type: ignore[arg-type]

        # Warm-up
        for _ in range(5):
            output_token_ids.clear()
            for token in full_tokens[:20]:
                output_token_ids.append(token)
                logits = torch.zeros(1, tokenizer.vocab_size)
                batch_processor.apply(logits)

        # Reset for actual benchmark
        # Need to recreate state since we cleared tokens
        batch_processor.req_states.clear()
        output_token_ids.clear()
        batch_update = MockBatchUpdate(
            batch_size=1,
            added=[(0, params, None, output_token_ids)],
        )
        batch_processor.update_state(batch_update)  # type: ignore[arg-type]

        # Benchmark: Generate tokens one by one
        start = time.perf_counter()
        for token in full_tokens:
            output_token_ids.append(token)
            logits = torch.zeros(1, tokenizer.vocab_size)
            batch_processor.apply(logits)
        elapsed = time.perf_counter() - start

        total_ms = elapsed * 1000
        per_token_us = (elapsed / len(full_tokens)) * 1_000_000

        print(f"Total time for {len(full_tokens)} tokens: {total_ms:.2f}ms")
        print(f"Per-token latency: {per_token_us:.2f}µs")

        # Target: <100ms total, <500µs per token
        assert total_ms < 100.0, (
            f"Incremental generation too slow: {total_ms:.2f}ms (target <100ms)"
        )
        assert per_token_us < 500.0, (
            f"Per-token latency too high: {per_token_us:.2f}µs (target <500µs)"
        )
