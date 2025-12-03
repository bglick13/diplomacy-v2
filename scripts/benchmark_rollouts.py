import time

import modal

from src.utils.config import ExperimentConfig

app = modal.App("diplomacy-grpo")


def benchmark():
    # Scale this up to 50 when ready
    N_GAMES = 2
    print(f"🚀 Benchmarking: Launching {N_GAMES} concurrent rollouts...")

    run_rollout = modal.Function.from_name("diplomacy-grpo", "run_rollout")

    # Warmup: Ensure InferenceEngine is ready before starting rollouts
    print("🔥 Warming up InferenceEngine...")
    InferenceEngine = modal.Cls.from_name("diplomacy-grpo", "InferenceEngine")
    engine = InferenceEngine()
    # Make a minimal call to trigger container startup and wait for @modal.enter() to complete
    _ = engine.generate.remote(prompts=["<orders>"], valid_moves=[{"A PAR": ["A PAR - BUR"]}])
    print("✅ InferenceEngine ready!")

    # Ensure this matches your app.py defaults (8 samples usually)
    cfg = ExperimentConfig(rollout_horizon_years=2, samples_per_group=8)

    start_time = time.time()
    results = []

    try:
        # We iterate as they complete to see progress
        for i, res in enumerate(run_rollout.map([cfg.dict()] * N_GAMES)):
            results.append(res)
            print(f"[{i + 1}/{N_GAMES}] ✅ Game finished. Trajectories collected: {len(res)}")

    except Exception as e:
        print(f"❌ Critical Failure: {e}")
        return

    duration = time.time() - start_time

    # --- CORRECTED METRICS ---
    # 1. Total Data Points (The "Gold")
    total_trajectories = sum(len(r) for r in results)

    # 2. Total Simulation Work (The "Throughput")
    # We simulated N games, each split into G clones, for H years.
    total_simulated_years = N_GAMES * cfg.samples_per_group * cfg.rollout_horizon_years

    print("\n" + "=" * 40)
    print("BENCHMARK COMPLETE")
    print(f"Total Wall Time:   {duration:.2f}s")
    print(f"Simulated Years:   {total_simulated_years}")
    print(f"Real Throughput:   {total_simulated_years / duration:.2f} Years/Sec")
    print(f"Total Trajectories: {total_trajectories}")
    print("=" * 40)

    # Validation
    if results:
        sample = results[0][0]
        required_keys = ["prompt", "completion", "reward", "group_id"]
        assert all(k in sample for k in required_keys), f"Missing keys: {sample.keys()}"
        print("Data Structure Validation: PASS")


if __name__ == "__main__":
    benchmark()
