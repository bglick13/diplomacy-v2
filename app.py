# from pathlib import Path

# import modal


# @app.function(
#     image=cpu_image,
#     timeout=86400,  # 24 hours max for long sweeps
#     secrets=[modal.Secret.from_name("axiom-secrets")],
# )
# def run_power_laws_sweep(
#     total_steps: int = 100,
#     num_groups_per_step: int = 8,
#     learning_rate: float = 1e-5,
#     model_id: str = "Qwen/Qwen2.5-7B-Instruct",
#     run_configs: list[str] | None = None,  # ["A", "B", "C"] or None for all
#     parallel: bool = False,
# ) -> dict:
#     """
#     Orchestrate the Power Laws experiment entirely on Modal.

#     This function runs in the cloud, so you can close your laptop after launching.
#     Progress is logged to Axiom and results are returned when complete.

#     Args:
#         total_steps: Training steps per configuration
#         num_groups_per_step: Rollout groups per step
#         learning_rate: Optimizer learning rate
#         model_id: Model to use for inference
#         run_configs: List of configs to run ["A", "B", "C"], or None for all
#         parallel: If True, run configs in parallel (3x cost, 3x faster)

#     Returns:
#         Dict with results from all configurations and analysis
#     """
#     import time
#     from datetime import datetime

#     from src.utils.observability import axiom, logger

#     # Define the three experimental configurations
#     SWEEP_CONFIGS = {
#         "A": {
#             "name": "baseline",
#             "tag": "power-laws-baseline",
#             "rollout_horizon_years": 2,
#             "samples_per_group": 8,
#             "compute_multiplier": 1.0,
#             "description": "Baseline: Fast & Loose (horizon=2, samples=8)",
#         },
#         "B": {
#             "name": "deep-search",
#             "tag": "power-laws-deep",
#             "rollout_horizon_years": 4,
#             "samples_per_group": 8,
#             "compute_multiplier": 2.0,
#             "description": "Deep Search: Time Scaling (horizon=4, samples=8)",
#         },
#         "C": {
#             "name": "broad-search",
#             "tag": "power-laws-broad",
#             "rollout_horizon_years": 2,
#             "samples_per_group": 16,
#             "compute_multiplier": 2.0,
#             "description": "Broad Search: Variance Scaling (horizon=2, samples=16)",
#         },
#     }

#     # Determine which configs to run
#     if run_configs is None:
#         run_configs = ["A", "B", "C"]

#     configs_to_run = [SWEEP_CONFIGS[k] for k in run_configs if k in SWEEP_CONFIGS]
#     timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

#     logger.info("=" * 60)
#     logger.info("🔬 POWER LAWS SWEEP STARTING (Cloud Orchestrated)")
#     logger.info("=" * 60)
#     logger.info(f"Timestamp: {timestamp}")
#     logger.info(f"Total Steps: {total_steps}")
#     logger.info(f"Groups/Step: {num_groups_per_step}")
#     logger.info(f"Model: {model_id}")
#     logger.info(f"Parallel: {parallel}")
#     logger.info(f"Configs: {run_configs}")

#     # Log sweep start to Axiom
#     axiom.log(
#         {
#             "event": "power_laws_sweep_start",
#             "timestamp": timestamp,
#             "total_steps": total_steps,
#             "num_groups_per_step": num_groups_per_step,
#             "model_id": model_id,
#             "parallel": parallel,
#             "configs": run_configs,
#         }
#     )

#     sweep_start = time.time()
#     results = []

#     def run_single_config(config: dict) -> dict:
#         """Run a single configuration and return results."""
#         config_start = time.time()
#         sim_years = (
#             num_groups_per_step
#             * config["samples_per_group"]
#             * config["rollout_horizon_years"]
#             * total_steps
#         )

#         logger.info(f"\n🚀 Starting: {config['description']}")
#         logger.info(f"   Simulated Years: {sim_years}")

#         # Log config start
#         axiom.log(
#             {
#                 "event": "power_laws_config_start",
#                 "config_name": config["name"],
#                 "timestamp": timestamp,
#                 "simulated_years": sim_years,
#             }
#         )

#         try:
#             result = train_grpo.remote(
#                 config_dict={
#                     "total_steps": total_steps,
#                     "num_groups_per_step": num_groups_per_step,
#                     "samples_per_group": config["samples_per_group"],
#                     "rollout_horizon_years": config["rollout_horizon_years"],
#                     "learning_rate": learning_rate,
#                     "rollout_visualize_chance": 0.0,
#                     "compact_prompts": True,
#                     "experiment_tag": config["tag"],
#                     "base_model_id": model_id,
#                 }
#             )

#             duration = time.time() - config_start

#             # Log config complete
#             axiom.log(
#                 {
#                     "event": "power_laws_config_complete",
#                     "config_name": config["name"],
#                     "timestamp": timestamp,
#                     "duration_s": duration,
#                     "final_reward_mean": result.get("final_reward_mean"),
#                     "final_loss": result.get("final_loss"),
#                 }
#             )

#             logger.info(f"✅ {config['name']} complete in {duration:.1f}s")
#             logger.info(f"   Final Reward: {result.get('final_reward_mean', 'N/A')}")

#             return {
#                 "config": config,
#                 "result": result,
#                 "duration_s": duration,
#                 "simulated_years": sim_years,
#             }

#         except Exception as e:
#             logger.error(f"❌ {config['name']} failed: {e}")
#             axiom.log(
#                 {
#                     "event": "power_laws_config_error",
#                     "config_name": config["name"],
#                     "error": str(e),
#                 }
#             )
#             return {
#                 "config": config,
#                 "error": str(e),
#                 "duration_s": time.time() - config_start,
#                 "simulated_years": sim_years,
#             }

#     if parallel:
#         # Run all configs in parallel using spawn
#         logger.info("\n🔀 Running configs in PARALLEL...")
#         handles = [
#             (
#                 config,
#                 train_grpo.spawn(
#                     config_dict={
#                         "total_steps": total_steps,
#                         "num_groups_per_step": num_groups_per_step,
#                         "samples_per_group": config["samples_per_group"],
#                         "rollout_horizon_years": config["rollout_horizon_years"],
#                         "learning_rate": learning_rate,
#                         "rollout_visualize_chance": 0.0,
#                         "compact_prompts": True,
#                         "experiment_tag": config["tag"],
#                         "base_model_id": model_id,
#                     }
#                 ),
#                 time.time(),
#             )
#             for config in configs_to_run
#         ]

#         for config, handle, start_time in handles:
#             sim_years = (
#                 num_groups_per_step
#                 * config["samples_per_group"]
#                 * config["rollout_horizon_years"]
#                 * total_steps
#             )
#             try:
#                 result = handle.get()
#                 duration = time.time() - start_time
#                 results.append(
#                     {
#                         "config": config,
#                         "result": result,
#                         "duration_s": duration,
#                         "simulated_years": sim_years,
#                     }
#                 )
#                 logger.info(f"✅ {config['name']} complete")
#             except Exception as e:
#                 results.append(
#                     {
#                         "config": config,
#                         "error": str(e),
#                         "duration_s": time.time() - start_time,
#                         "simulated_years": sim_years,
#                     }
#                 )
#     else:
#         # Run configs sequentially
#         logger.info("\n📋 Running configs SEQUENTIALLY...")
#         for config in configs_to_run:
#             result = run_single_config(config)
#             results.append(result)

#     # Analyze results
#     total_duration = time.time() - sweep_start

#     logger.info("\n" + "=" * 60)
#     logger.info("📊 POWER LAWS SWEEP COMPLETE")
#     logger.info("=" * 60)
#     logger.info(f"Total Duration: {total_duration / 3600:.2f} hours")

#     # Build comparison table
#     valid_results = [r for r in results if "result" in r and r["result"]]
#     comparison = []

#     for r in results:
#         config = r["config"]
#         entry = {
#             "name": config["name"],
#             "description": config["description"],
#             "compute_multiplier": config["compute_multiplier"],
#             "simulated_years": r["simulated_years"],
#             "duration_s": r["duration_s"],
#         }
#         if "result" in r and r["result"]:
#             entry["final_reward_mean"] = r["result"].get("final_reward_mean")
#             entry["final_loss"] = r["result"].get("final_loss")
#             entry["final_kl"] = r["result"].get("final_kl")
#             entry["run_name"] = r["result"].get("run_name")
#         else:
#             entry["error"] = r.get("error", "Unknown error")
#         comparison.append(entry)

#     # Determine winner
#     analysis = {"winner": None, "interpretation": ""}
#     if valid_results:
#         best = max(
#             valid_results, key=lambda r: r["result"].get("final_reward_mean") or float("-inf")
#         )
#         best_name = best["config"]["name"]
#         analysis["winner"] = best_name
#         analysis["best_reward"] = best["result"].get("final_reward_mean")

#         if best_name == "deep-search":
#             analysis["interpretation"] = (
#                 "HORIZON SCALING WINS: Simple reward works, increase rollout_horizon_years"
#             )
#         elif best_name == "broad-search":
#             analysis["interpretation"] = (
#                 "VARIANCE SCALING WINS: Simple reward works, increase samples_per_group"
#             )
#         else:
#             analysis["interpretation"] = (
#                 "BASELINE WINS: Scaling alone insufficient, consider reward engineering"
#             )

#         logger.info(f"\n🏆 Winner: {best_name}")
#         logger.info(f"   {analysis['interpretation']}")

#     # Log sweep complete
#     axiom.log(
#         {
#             "event": "power_laws_sweep_complete",
#             "timestamp": timestamp,
#             "total_duration_s": total_duration,
#             "winner": analysis.get("winner"),
#             "results_count": len(results),
#         }
#     )

#     import asyncio

#     asyncio.run(axiom.flush())

#     return {
#         "timestamp": timestamp,
#         "total_duration_s": total_duration,
#         "total_duration_hours": total_duration / 3600,
#         "comparison": comparison,
#         "analysis": analysis,
#         "config": {
#             "total_steps": total_steps,
#             "num_groups_per_step": num_groups_per_step,
#             "learning_rate": learning_rate,
#             "model_id": model_id,
#             "parallel": parallel,
#         },
#     }


# ==============================================================================
# 9. EVALUATION RUNNER
# ==============================================================================


# ------------------------------------------------------------------------------
# 9.1 Async Elo Evaluator (runs in background during training)
# ------------------------------------------------------------------------------


