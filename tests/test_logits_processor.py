import pytest
import torch
from transformers import AutoTokenizer

from src.inference.logits import DiplomacyLogitsProcessor

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
    return DiplomacyLogitsProcessor(
        tokenizer=tokenizer, valid_moves_dict=SAMPLE_VALID_MOVES
    )


def test_initial_masking(processor, tokenizer):
    """Test that at the start, only the first tokens of valid orders are allowed."""
    input_ids = []
    scores = torch.zeros(tokenizer.vocab_size)

    # Run processor
    processed_scores = processor(input_ids, scores)

    # Check valid start tokens
    # "A" (from A PAR) and "F" (from F BRE) should be valid
    # Note: We encode "A PAR - BUR" -> gets first token id
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


def test_path_following(processor, tokenizer):
    """Test that once we start a move, we must follow the valid path."""
    # Simulate generating "A PAR -"
    # We feed the sequence of tokens corresponding to this string
    history_str = "A PAR -"
    input_ids = tokenizer.encode(history_str)

    scores = torch.zeros(tokenizer.vocab_size)
    processed_scores = processor(input_ids, scores)

    # Next valid tokens should be " BUR" or " MAR"
    # Note: spacing depends on tokenizer. GPT2 usually handles " BUR" as one token.
    token_bur = tokenizer.encode(" BUR")[0]  # Space included if part of prompt flow
    token_mar = tokenizer.encode(" MAR")[0]

    # Check if they are unmasked (using a lenient check for demonstration)
    # In strict testing, we'd check the Trie children directly
    assert processed_scores[token_bur] > -float("inf") or processed_scores[
        tokenizer.encode("BUR")[0]
    ] > -float("inf")


def test_dead_end_blocking(processor, tokenizer):
    """Test that off-path tokens are strictly forbidden."""
    # Simulate "A PAR - "
    input_ids = tokenizer.encode("A PAR -")
    scores = torch.zeros(tokenizer.vocab_size)
    processed_scores = processor(input_ids, scores)

    # "LON" is a valid game token, but NOT valid for A PAR (Paris isn't adjacent to London)
    token_lon = tokenizer.encode(" LON")[0]

    assert processed_scores[token_lon] == -float("inf"), (
        "London is not adjacent to Paris; should be blocked."
    )


def test_newline_reset(processor, tokenizer):
    """Test that hitting \n resets the Trie to the root to allow a new order."""
    # Simulate a full valid order "A PAR - BUR" followed by newline
    # Note: The processor logic needs to see the newline in the input_ids to reset
    full_order = "A PAR - BUR\n"
    input_ids = tokenizer.encode(full_order)

    scores = torch.zeros(tokenizer.vocab_size)

    # We need to artificially reset the processor state for this test
    # since we are jumping ahead in time
    processor.current_node = processor.root
    # Walk the trie manually to simulate state (or mock the internal update)
    # For unit testing the class logic, we can just instantiate a fresh one
    # and feed the newline as the "last token" logic requires.

    # Let's interact properly:
    # 1. Feed order
    for i in range(len(input_ids)):
        # Step through tokens one by one
        current_history = input_ids[:i]
        next_logit = torch.zeros(tokenizer.vocab_size)
        processor(current_history, next_logit)

    # Now we are at the state AFTER "A PAR - BUR\n"
    # The next token should allow "A" or "F" again (start of new order)

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


def test_token_boundary_stress(processor, tokenizer):
    """
    Stress test for sub-word tokenization.
    Example: 'Burgundy' might be split into 'Bur' + 'gundy'.
    The Trie must support multi-token moves.
    """
    # Let's pretend "BURGUNDY" is a valid move (instead of BUR)
    # and force it into the valid moves list
    long_move_processor = DiplomacyLogitsProcessor(
        tokenizer,
        {"A PAR": ["A PAR - BURGUNDY"]},  # Long word
    )

    # Walk: "A PAR - "
    input_ids = tokenizer.encode("A PAR - ")
    scores = torch.zeros(tokenizer.vocab_size)

    # Step 1: Expect "BUR" or "B"
    scores = long_move_processor(input_ids, scores)

    # Step 2: Simulate selecting "BUR"
    input_ids.append(tokenizer.encode("BUR")[0])
    scores = torch.zeros(tokenizer.vocab_size)

    # Step 3: Expect "GUNDY"
    scores = long_move_processor(input_ids, scores)

    token_gundy = tokenizer.encode("GUNDY")[0]
    assert scores[token_gundy] > -float("inf"), (
        "Should allow second half of split token"
    )
