#!/usr/bin/env python
"""Debug script to trace completion_logprobs flow through the training pipeline."""

import modal

# Step 1: Get a sample trajectory from WandB or Weave
# Step 2: Check if completion_logprobs is present in the stored data


def check_inference_engine_response():
    """Check that inference engine returns logprobs."""
    print("\n" + "=" * 60)
    print("Step 1: Checking InferenceEngine response format")
    print("=" * 60)

    InferenceEngine = modal.Cls.from_name("diplomacy-grpo", "InferenceEngine")
    engine = InferenceEngine()

    test_prompt = """You are playing France in Diplomacy. The current phase is SPRING 1901.
<analysis>I should move my units strategically.</analysis>
<orders>"""

    test_moves = {"A PAR": ["A PAR - BUR", "A PAR - PIC", "A PAR - GAS", "A PAR H"]}

    responses = engine.generate_batch.remote(
        prompts=[test_prompt],
        valid_moves=[test_moves],
        lora_names=[None],
    )

    resp = responses[0]
    print(f"\nResponse keys: {list(resp.keys())}")
    print(f"token_ids count: {len(resp.get('token_ids', []))}")
    print(f"completion_logprobs count: {len(resp.get('completion_logprobs', []))}")
    print(f"completion_logprobs sample: {resp.get('completion_logprobs', [])[:5]}")

    if resp.get("completion_logprobs"):
        print("\n✅ InferenceEngine returns completion_logprobs correctly")
        return True
    else:
        print("\n❌ InferenceEngine NOT returning completion_logprobs!")
        return False


def check_recent_trajectories_from_wandb():
    """Check a recent training run's logged trajectory stats."""
    print("\n" + "=" * 60)
    print("Step 2: Checking WandB logged stats for ref_logprobs")
    print("=" * 60)

    import subprocess

    # Get recent run metrics
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/wandb_cli.py",
            "get-metrics",
            "-r",
            "reward-discount-gamma-v2-A-20251225-222219",
            "--all-metrics",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        lines = result.stdout.split("\n")
        for line in lines:
            if any(
                k in line.lower()
                for k in ["ref_logprobs", "rollout_logprobs", "ppo", "cached", "ratio"]
            ):
                print(line)
    else:
        print(f"Error getting WandB metrics: {result.stderr}")


def simulate_trajectory_processing():
    """Simulate what trainer.py does with trajectories."""
    print("\n" + "=" * 60)
    print("Step 3: Simulating trajectory processing")
    print("=" * 60)

    # Create a mock trajectory like what build_trajectories produces
    mock_trajectory = {
        "prompt": "Test prompt",
        "completion": "Test completion",
        "reward": 0.5,
        "group_id": "test_game_FRANCE_step0",
        "prompt_token_ids": list(range(100)),
        "completion_token_ids": list(range(10)),
        "completion_logprobs": [-0.5] * 10,  # Simulated logprobs
    }

    print(
        f"Mock trajectory has completion_logprobs: {bool(mock_trajectory.get('completion_logprobs'))}"
    )
    print(f"completion_logprobs length: {len(mock_trajectory.get('completion_logprobs', []))}")

    # Now simulate what trainer.py does
    from transformers import AutoTokenizer

    from src.training.trainer import process_trajectories

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

    # Process the mock trajectory
    batch_data, stats = process_trajectories(
        [mock_trajectory, mock_trajectory],  # Need 2 for min_group_size
        tokenizer,
        min_group_size=2,
        verbose=True,
    )

    print(f"\nProcessed {len(batch_data)} items")
    if batch_data:
        item = batch_data[0]
        print(f"Processed item keys: {list(item.keys())}")
        print(f"Has ref_logprobs: {'ref_logprobs' in item}")
        print(f"Has rollout_logprobs: {'rollout_logprobs' in item}")

        if "ref_logprobs" in item:
            print(f"ref_logprobs value: {item['ref_logprobs']}")
            print("\n✅ Trainer correctly extracts completion_logprobs -> ref_logprobs")
        else:
            print("\n❌ Trainer NOT extracting ref_logprobs!")


def check_old_length_check():
    """Check if old length check would have passed."""
    print("\n" + "=" * 60)
    print("Step 4: Testing old vs new length check logic")
    print("=" * 60)

    # Simulate what the old check looked like
    completion_logprobs = [-0.5] * 10
    completion_token_ids = list(range(10))

    print(f"completion_logprobs length: {len(completion_logprobs)}")
    print(f"completion_token_ids length: {len(completion_token_ids)}")

    # Old check
    old_check_passes = completion_logprobs and len(completion_logprobs) == len(completion_token_ids)
    print(f"\nOld strict check passes: {old_check_passes}")

    # New check
    new_check_passes = bool(completion_logprobs)
    print(f"New relaxed check passes: {new_check_passes}")

    # Mismatched lengths case
    mismatched_logprobs = [-0.5] * 9
    old_mismatch = mismatched_logprobs and len(mismatched_logprobs) == len(completion_token_ids)
    new_mismatch = bool(mismatched_logprobs)
    print("\nWith length mismatch (9 vs 10):")
    print(f"  Old check: {old_mismatch}")
    print(f"  New check: {new_mismatch}")


def check_trainer_code():
    """Verify the trainer.py code has the fix."""
    print("\n" + "=" * 60)
    print("Step 5: Verifying trainer.py code")
    print("=" * 60)

    with open("src/training/trainer.py") as f:
        content = f.read()

    # Look for the old buggy pattern
    old_pattern = "len(completion_logprobs) == len(completion_token_ids)"
    if old_pattern in content:
        print("⚠️ Old strict length check STILL in code!")
    else:
        print("✅ Old strict length check removed")

    # Look for the new pattern
    new_pattern = "if completion_logprobs:"
    if new_pattern in content:
        print("✅ New relaxed check present")

    # Show the relevant section
    import re

    match = re.search(r"ref_logprobs = None.*?ref_logprobs = sum", content, re.DOTALL)
    if match:
        print("\nRelevant code section:")
        lines = match.group(0).split("\n")
        for line in lines[:10]:
            print(f"  {line}")


if __name__ == "__main__":
    print("=" * 60)
    print("DEBUGGING COMPLETION_LOGPROBS FLOW")
    print("=" * 60)

    # Check 1: Inference engine
    try:
        engine_ok = check_inference_engine_response()
    except Exception as e:
        print(f"Could not check inference engine: {e}")
        engine_ok = None

    # Check 2: WandB metrics
    try:
        check_recent_trajectories_from_wandb()
    except Exception as e:
        print(f"Could not check WandB: {e}")

    # Check 3: Trajectory processing
    try:
        simulate_trajectory_processing()
    except Exception as e:
        print(f"Could not simulate trajectory processing: {e}")
        import traceback

        traceback.print_exc()

    # Check 4: Length check logic
    check_old_length_check()

    # Check 5: Code verification
    check_trainer_code()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(
        """
If all checks pass but PPO is still disabled, the issue might be:
1. The deployed code on Modal doesn't have the fix
2. The trajectories from rollouts don't have completion_logprobs
3. Something else is stripping the logprobs

Next steps:
1. Check Axiom logs for 'Missing completion_logprobs' warnings
2. Check the deployed Modal app code
3. Add more debug logging to the deployed trainer
"""
    )
