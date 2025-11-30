import time

import modal

from src.utils.config import ExperimentConfig

app = modal.App("diplomacy-grpo")


def benchmark():
    # 1. Setup
    N_GAMES = 4
    print(f"🚀 Benchmarking: Launching {N_GAMES} concurrent rollouts on Modal...")

    # Look up the function
    run_rollout = modal.Function.from_name("diplomacy-grpo", "run_rollout")

    # Warmup: Ensure InferenceEngine is ready before starting rollouts
    print("🔥 Warming up InferenceEngine...")
    InferenceEngine = modal.Cls.from_name("diplomacy-grpo", "InferenceEngine")
    engine = InferenceEngine()
    # Make a minimal call to trigger container startup and wait for @modal.enter() to complete
    _ = engine.generate.remote(
        prompts=["<orders>"], valid_moves=[{"A PAR": ["A PAR - BUR"]}]
    )
    print("✅ InferenceEngine ready!")

    # Config
    cfg = ExperimentConfig(rollout_horizon_years=2)
    config_dict = cfg.dict()

    start_time = time.time()

    # 2. Launch Async Map
    # .map() returns a generator, usually we iterate it to get results
    results = []

    try:
        # Pass the config to every worker
        for res in run_rollout.map([config_dict] * N_GAMES):
            results.append(res)
            print(f"✅ Game finished. Trajectories collected: {len(res)}")

    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"❌ Critical Failure: {e}")
        return

    total_time = time.time() - start_time

    # 3. Stats
    total_years = N_GAMES * cfg.rollout_horizon_years
    print("\n" + "=" * 40)
    print("BENCHMARK COMPLETE")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Throughput: {total_years / total_time:.2f} Years/Sec")
    print(f"Total Trajectories: {len(results) * 7}")  # 7 powers per game
    print("=" * 40)

    # 4. Validation
    sample = results[0][0]
    required_keys = ["prompt", "completion", "reward"]
    assert all(k in sample for k in required_keys), (
        f"Missing keys in output: {sample.keys()}"
    )
    print("Data Structure Validation: PASS")


if __name__ == "__main__":
    benchmark()
