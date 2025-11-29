import modal
import pytest

# Connect to the App
app = modal.App("diplomacy-grpo")

# Dummy Data
VALID_MOVES = {"A PAR": ["A PAR - BUR", "A PAR - PIC"], "F BRE": ["F BRE - MAO"]}
PROMPT = "You are France. <orders>"


@pytest.mark.asyncio
async def test_inference_rpc():
    """
    Tests that we can call the remote GPU inference engine
    and get a valid response respecting the LogitsProcessor.
    """
    # with modal.enable_root_logger():
    # We use .remote() to trigger the cloud execution
    # Note: In a real Pytest suite running locally, this triggers a Modal launch.
    # It might be slow.
    InferenceEngine = modal.Cls.from_name("diplomacy-grpo", "InferenceEngine")
    print("🚀 Calling Inference Engine...")
    # 2. INSTANTIATE THE CLASS HANDLE (The Fix: Add parentheses)
    engine = InferenceEngine()

    # 3. Call the method on the instance
    responses = engine.generate.remote(prompts=[PROMPT], valid_moves=[VALID_MOVES])

    print(f"Response: {responses[0]}")

    # Check if output contains one of our valid moves
    assert "A PAR" in responses[0] or "F BRE" in responses[0]
