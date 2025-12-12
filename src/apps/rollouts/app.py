import os
import random
import time

import cloudpickle
import modal

from src.agents import LLMAgent, PromptConfig
from src.agents.baselines import ChaosBot, RandomBot
from src.apps.common.images import cpu_image
from src.apps.common.volumes import REPLAYS_PATH, VOLUME_PATH, volume
from src.engine.wrapper import DiplomacyWrapper
from src.utils.config import ExperimentConfig
from src.utils.observability import (
    RolloutMetrics,
    axiom,
    log_orders_extracted,
    log_rollout_complete,
    log_rollout_error,
    log_rollout_start,
    logger,
    stopwatch,
)
from src.utils.parsing import extract_orders
from src.utils.scoring import calculate_final_scores
from src.utils.vis import GameVisualizer

app = modal.App("diplomacy-grpo-rollouts")

CURRENT_ROLLOUT_LORA: str | None = None


@app.function(
    image=cpu_image,
    cpu=1.0,
    memory=2048,  # Increased memory to hold G game copies
    timeout=3600,
    secrets=[modal.Secret.from_name("axiom-secrets")],
    volumes={str(VOLUME_PATH): volume},
)
async def run_rollout(
    config_dict: dict,
    lora_name: str | None = None,
    *,
    power_adapters: dict[str, str | None] | None = None,
    hero_power: str | None = None,
):
    """
    Run a single rollout (game simulation) with the given config.

    Args:
        config_dict: ExperimentConfig as dict
        lora_name: (DEPRECATED) Name of the LoRA adapter for ALL powers.
                   Use power_adapters instead for league training.
        power_adapters: Mapping of power name to adapter. Values can be:
                       - "random_bot" or "chaos_bot": Use baseline bot (no LLM)
                       - None or "base_model": Use base LLM (no LoRA)
                       - str: LoRA adapter path (e.g., "run/adapter_v50")
                       If None, falls back to using lora_name for all powers.
        hero_power: Which power's trajectories to collect for training.
                   If None, collects trajectories for ALL powers (original behavior).
    """
    # Get InferenceEngine class from the deployed app at runtime
    # This ensures it's properly hydrated in the combined app context
    InferenceEngineCls = modal.Cls.from_name("diplomacy-grpo", "InferenceEngine")

    # Baseline bot instances (stateless, can be shared)
    BASELINE_BOTS = {
        "random_bot": RandomBot(),
        "chaos_bot": ChaosBot(),
    }

    # Build power_adapters from legacy lora_name if not provided
    POWERS = ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]
    if power_adapters is None:
        # Legacy mode: same adapter for all powers
        power_adapters = dict.fromkeys(POWERS, lora_name)

    # NOTE: We use vLLM's lora_filesystem_resolver to load adapters dynamically
    # The resolver looks in VLLM_LORA_RESOLVER_CACHE_DIR (/data/models) for the adapter
    # We just need to reload the volume to see newly committed adapter files
    volume_reload_time = 0.0

    # Collect unique LoRA adapters that need to be loaded
    unique_loras = {
        adapter
        for adapter in power_adapters.values()
        if adapter is not None and adapter not in BASELINE_BOTS and adapter != "base_model"
    }

    # Reload volume if there are any LoRAs to load
    if unique_loras:
        logger.info(f"📂 Using LoRA adapters: {unique_loras}")
        global CURRENT_ROLLOUT_LORA

        # Check if we need to reload (any new adapter not seen before)
        needs_reload = any(
            adapter != CURRENT_ROLLOUT_LORA and not os.path.exists(f"/data/models/{adapter}")
            for adapter in unique_loras
        )

        if needs_reload:
            reload_start = time.time()
            volume.reload()  # Ensure we see the latest adapter files
            volume_reload_time = time.time() - reload_start
            logger.info(f"⏱️ Volume reload took {volume_reload_time:.2f}s")

        # Verify all adapters exist
        for adapter in unique_loras:
            full_path = f"/data/models/{adapter}"
            if os.path.exists(full_path):
                files = os.listdir(full_path)
                logger.info(f"✅ LoRA adapter found at {full_path}. Files: {files}")
            else:
                logger.error(f"❌ LoRA adapter NOT found at: {full_path}")

        # Track the "main" adapter (for backwards compat in reference logprob logic)
        if unique_loras:
            CURRENT_ROLLOUT_LORA = next(iter(unique_loras))

    rollout_start_time = time.time()
    rollout_id = ""
    metrics: RolloutMetrics | None = None

    try:
        cfg = ExperimentConfig(**config_dict)
        should_visualize = random.random() < cfg.rollout_visualize_chance

        # Initialize the LLM agent for prompt generation
        prompt_config = PromptConfig(
            compact_mode=cfg.compact_prompts,
            prefix_cache_optimized=cfg.prefix_cache_optimized,
            show_valid_moves=cfg.show_valid_moves,
        )
        agent = LLMAgent(config=prompt_config)

        # 1. THE WARMUP (Generate a random state)
        # ---------------------------------------
        warmup_phases = random.randint(0, 8)
        if random.random() < cfg.rollout_no_warmup_chance:
            warmup_phases = 0

        # Init main game
        main_game = DiplomacyWrapper(horizon=99)
        rollout_id = main_game.game.game_id
        metrics = RolloutMetrics(rollout_id=rollout_id)

        # Log rollout start
        log_rollout_start(
            rollout_id=rollout_id,
            warmup_phases=warmup_phases,
            samples_per_group=cfg.samples_per_group,
            horizon_years=cfg.rollout_horizon_years,
        )

        vis = None
        if should_visualize:
            logger.info("Visualizing game...")
            vis = GameVisualizer()
            vis.capture_turn(main_game.game, "Warmup Start")
        logger.info(f"🔥 Starting Warmup: {warmup_phases} phases")

        # Play through warmup
        # NOTE: Warmup uses base model for LLM powers (not per-power adapters)
        # This is fine since warmup is just for generating diverse game states
        for i in range(warmup_phases):
            if main_game.is_done():
                break

            all_orders = []
            phase = main_game.get_current_phase()

            # Get inputs for all powers
            inputs = main_game.get_all_inputs(agent=agent)

            # Separate baseline bots from LLM powers
            baseline_indices = []
            llm_indices = []
            for idx, power in enumerate(inputs["power_names"]):
                adapter = power_adapters.get(power)
                if adapter in BASELINE_BOTS:
                    baseline_indices.append(idx)
                else:
                    llm_indices.append(idx)

            # 1. Handle baseline bots directly (no LLM)
            for idx in baseline_indices:
                power = inputs["power_names"][idx]
                adapter = power_adapters.get(power)
                assert adapter is not None and adapter in BASELINE_BOTS
                bot = BASELINE_BOTS[adapter]
                orders = bot.get_orders(main_game, power)
                expected_count = len(inputs["valid_moves"][idx])

                log_orders_extracted(
                    rollout_id=rollout_id,
                    power_name=power,
                    orders_count=len(orders),
                    expected_count=expected_count,
                    raw_response_length=0,
                    phase=phase,
                    raw_response="[BASELINE BOT]",
                )
                metrics.record_extraction(len(orders), expected_count)
                all_orders.extend(orders)

            # 2. Handle LLM powers with batched inference
            if llm_indices:
                llm_prompts = [inputs["prompts"][idx] for idx in llm_indices]
                llm_valid_moves = [inputs["valid_moves"][idx] for idx in llm_indices]

                with stopwatch(f"Warmup Inference {i + 1}/{warmup_phases}"):
                    metrics.inference_calls += 1
                    raw_responses = await InferenceEngineCls(  # pyright: ignore[reportCallIssue]
                        model_id=cfg.base_model_id
                    ).generate.remote.aio(
                        prompts=llm_prompts,
                        valid_moves=llm_valid_moves,
                        lora_name=None,  # Use base model for warmup
                        temperature=cfg.temperature,
                        max_new_tokens=cfg.max_new_tokens,
                    )

                for resp_idx, game_idx in enumerate(llm_indices):
                    response_data = raw_responses[resp_idx]
                    power = inputs["power_names"][game_idx]
                    response_text = response_data["text"]
                    orders = extract_orders(response_text)
                    expected_count = len(inputs["valid_moves"][game_idx])

                    log_orders_extracted(
                        rollout_id=rollout_id,
                        power_name=power,
                        orders_count=len(orders),
                        expected_count=expected_count,
                        raw_response_length=len(response_text),
                        phase=phase,
                        raw_response=response_text,
                    )
                    metrics.record_extraction(len(orders), expected_count)
                    all_orders.extend(orders)

            main_game.step(all_orders)

            if should_visualize and vis:
                vis.capture_turn(
                    main_game.game,
                    f"Warmup step {i + 1}/{warmup_phases}\n{chr(10).join(all_orders)}",
                )

        if main_game.is_done():
            logger.info("Game ended during warmup, discarding.")
            return []

        # 2. THE FORK (Clone the state G times)
        # -------------------------------------
        frozen_state = cloudpickle.dumps(main_game)
        frozen_vis = cloudpickle.dumps(vis)
        if should_visualize and vis:
            # IMPORTANT: Create SEPARATE visualizer objects for each game
            # Using [obj] * n would create n references to the SAME object!
            visualizers = [cloudpickle.loads(frozen_vis) for _ in range(cfg.samples_per_group)]
        else:
            visualizers = None

        games = [cloudpickle.loads(frozen_state) for _ in range(cfg.samples_per_group)]
        current_year = main_game.get_year()
        target_year = current_year + cfg.rollout_horizon_years

        fork_data = {i: {} for i in range(len(games))}

        # 3. THE GROUP ROLLOUT (ASYNC FORK PIPELINE)
        # --------------------------------------------------
        # Convert to async tasks per cloned game for concurrent inference + parsing
        import asyncio

        async def run_game_async(g_idx: int, game: DiplomacyWrapper, vis_obj) -> dict:
            """Run a single game clone asynchronously until completion."""
            game_fork_data = {}
            step_count = 0

            while game.get_year() < target_year and not game.is_done():
                step_count += 1

                # Get inputs for all powers in this game
                inputs = game.get_all_inputs(agent=agent)
                phase = game.get_current_phase()

                # Results will be populated as we process each power
                # Format: {power_idx: {"orders": [...], "response_data": {...}}}
                power_results: dict[int, dict] = {}

                # 1. Handle baseline bots (no LLM inference needed)
                baseline_count = 0
                llm_count = 0
                for idx, power in enumerate(inputs["power_names"]):
                    adapter = power_adapters.get(power)
                    if adapter in BASELINE_BOTS:
                        baseline_count += 1
                        bot = BASELINE_BOTS[adapter]
                        orders = bot.get_orders(game, power)
                        power_results[idx] = {
                            "orders": orders,
                            "response_data": {
                                "text": "[BASELINE BOT]",
                                "prompt_token_ids": [],
                                "token_ids": [],
                                "completion_logprobs": [],
                            },
                        }
                    else:
                        llm_count += 1

                # 2. Group LLM powers by adapter for batched inference
                # Map adapter -> list of (original_idx, prompt, valid_moves)
                adapter_groups: dict[str | None, list[tuple[int, str, dict]]] = {}
                for idx, power in enumerate(inputs["power_names"]):
                    if idx in power_results:
                        continue  # Already handled by baseline bot

                    adapter = power_adapters.get(power)
                    # Normalize adapter: None and "base_model" both mean no LoRA
                    adapter_key = None if adapter in (None, "base_model") else adapter

                    if adapter_key not in adapter_groups:
                        adapter_groups[adapter_key] = []
                    adapter_groups[adapter_key].append(
                        (idx, inputs["prompts"][idx], inputs["valid_moves"][idx])
                    )

                # 3. Run batched inference for each adapter group
                # Note: batch_size=1 is common when:
                # - Only one power uses a particular adapter (league training)
                # - Powers are eliminated early in the game
                # - Using baseline bots for some powers
                # vLLM's continuous batching will still batch these across concurrent games
                for adapter_key, group_items in adapter_groups.items():
                    group_indices = [item[0] for item in group_items]
                    group_prompts = [item[1] for item in group_items]
                    group_valid_moves = [item[2] for item in group_items]
                    batch_size = len(group_prompts)

                    with stopwatch(
                        f"Game {g_idx} Step {step_count} Adapter={adapter_key} (Batch: {batch_size})"
                    ):
                        metrics.inference_calls += 1
                        responses = await InferenceEngineCls(  # pyright: ignore[reportCallIssue]
                            model_id=cfg.base_model_id
                        ).generate.remote.aio(  # pyright: ignore[reportCallIssue]
                            prompts=group_prompts,
                            valid_moves=group_valid_moves,
                            lora_name=adapter_key,
                            temperature=cfg.temperature,
                            max_new_tokens=cfg.max_new_tokens,
                        )

                    # Map responses back to original indices
                    for resp_idx, orig_idx in enumerate(group_indices):
                        response_data = responses[resp_idx]
                        response_text = response_data["text"]
                        orders = extract_orders(response_text)
                        power_results[orig_idx] = {
                            "orders": orders,
                            "response_data": response_data,
                        }

                # 4. Collect all orders and log metrics
                all_orders = []
                for idx, power in enumerate(inputs["power_names"]):
                    result = power_results[idx]
                    orders = result["orders"]
                    response_data = result["response_data"]
                    response_text = response_data["text"]
                    expected_count = len(inputs["valid_moves"][idx])

                    log_orders_extracted(
                        rollout_id=rollout_id,
                        power_name=power,
                        orders_count=len(orders),
                        expected_count=expected_count,
                        raw_response_length=len(response_text),
                        phase=phase,
                        raw_response=response_text,
                    )
                    metrics.record_extraction(len(orders), expected_count)

                    # Store fork data on first step for hero power only
                    # (or all powers if hero_power is None for backwards compat)
                    should_collect = (hero_power is None) or (power == hero_power)
                    is_baseline = power_adapters.get(power) in BASELINE_BOTS

                    if step_count == 1 and should_collect and not is_baseline:
                        game_fork_data[power] = {
                            "prompt": inputs["prompts"][idx],
                            "completion": response_text,
                            "prompt_token_ids": response_data.get("prompt_token_ids", []),
                            "completion_token_ids": response_data.get("token_ids", []),
                            "completion_logprobs": response_data.get("completion_logprobs", []),
                        }

                    all_orders.extend(orders)

                # Step the game
                game.step(all_orders)

                # Update visualization if enabled
                if should_visualize and vis_obj is not None:
                    vis_obj.capture_turn(
                        game.game,
                        f"Rollout step {step_count}\n{chr(10).join(all_orders)}",
                    )

            return {"g_idx": g_idx, "fork_data": game_fork_data}

        # Run all games concurrently
        with stopwatch("Async Group Rollout"):
            game_tasks = [
                run_game_async(g_idx, game, visualizers[g_idx] if visualizers else None)
                for g_idx, game in enumerate(games)
            ]
            game_results = await asyncio.gather(*game_tasks)

        # Collect fork data from all games
        for result in game_results:
            fork_data[result["g_idx"]] = result["fork_data"]

        # 4. REFERENCE LOGPROBS (Optional - eliminates trainer reference forward pass)
        # ---------------------------------------------------------------------------
        # If enabled, compute base model logprobs for all completions
        # This adds ~1 forward pass to rollouts but saves ~1 forward pass in trainer
        ref_logprobs_map: dict[tuple[int, str], float | None] = {}

        # Check if any hero power uses a LoRA adapter (need ref logprobs)
        hero_uses_lora = False
        if hero_power:
            hero_adapter = power_adapters.get(hero_power)
            hero_uses_lora = (
                hero_adapter not in (None, "base_model") and hero_adapter not in BASELINE_BOTS
            )
        else:
            # Legacy mode: check if any unique LoRA is used
            hero_uses_lora = bool(unique_loras)

        if cfg.compute_ref_logprobs_in_rollout and hero_uses_lora:
            # Only needed when generating with LoRA (step > 0)
            # For step 0 (no LoRA), generation logprobs ARE reference logprobs
            with stopwatch("Reference Logprob Scoring"):
                # Batch all sequences for scoring
                score_prompts = []
                score_completions = []
                score_prompt_token_ids = []
                score_keys = []  # (g_idx, power) to map results back

                for g_idx in range(len(games)):
                    for power, data in fork_data[g_idx].items():
                        prompt_tids = data.get("prompt_token_ids", [])
                        if prompt_tids:  # Only if we have token data
                            score_prompts.append(data["prompt"])
                            score_completions.append(data["completion"])
                            score_prompt_token_ids.append(prompt_tids)
                            score_keys.append((g_idx, power))

                if score_prompts:
                    # Call scoring endpoint (uses base model, no LoRA)
                    score_results = await InferenceEngineCls(  # pyright: ignore[reportCallIssue]
                        model_id=cfg.base_model_id
                    ).score.remote.aio(
                        prompts=score_prompts,
                        completions=score_completions,
                        prompt_token_ids_list=score_prompt_token_ids,
                    )

                    # Map results back
                    for key, result in zip(score_keys, score_results, strict=True):
                        ref_logprobs_map[key] = result.get("ref_completion_logprobs")

                    logger.info(
                        f"📊 Computed {len(score_results)} reference logprobs (using base model)"
                    )

        # 5. SCORING & RETURN
        # -------------------
        trajectories = []

        for g_idx, game in enumerate(games):
            # Use win_bonus from config for scoring
            final_scores = calculate_final_scores(
                game,
                win_bonus=cfg.win_bonus,
                winner_threshold_sc=cfg.winner_threshold_sc,
            )

            for power, data in fork_data[g_idx].items():
                if power in final_scores:
                    # Get reference logprobs if computed
                    ref_logprobs = ref_logprobs_map.get((g_idx, power))

                    # If no LoRA used (step 0), generation logprobs ARE reference logprobs
                    power_adapter = power_adapters.get(power)
                    power_uses_lora = (
                        power_adapter not in (None, "base_model")
                        and power_adapter not in BASELINE_BOTS
                    )
                    if ref_logprobs is None and not power_uses_lora:
                        completion_logprobs = data.get("completion_logprobs", [])
                        if completion_logprobs:
                            ref_logprobs = sum(completion_logprobs)

                    traj = {
                        "prompt": data["prompt"],
                        "completion": data["completion"],
                        "reward": final_scores[power],
                        "group_id": f"{main_game.game.game_id}_{power}_{current_year}",
                        # Token data for trainer optimization (skip tokenization)
                        "prompt_token_ids": data.get("prompt_token_ids", []),
                        "completion_token_ids": data.get("completion_token_ids", []),
                        "completion_logprobs": data.get("completion_logprobs", []),
                    }

                    # Add reference logprobs if available (enables skipping trainer ref forward)
                    if ref_logprobs is not None:
                        traj["ref_logprobs"] = ref_logprobs

                    trajectories.append(traj)

        # Save visualizations
        if should_visualize and vis and visualizers is not None:
            output_file = str(REPLAYS_PATH / f"rollout_{main_game.game.game_id}.html")
            vis.save_html(output_file)
            logger.info(f"✅ Saved replay to {output_file}")

            for g_idx, game in enumerate(games):
                output_file = str(REPLAYS_PATH / f"rollout_{game.game.game_id}_{g_idx}.html")
                visualizers[g_idx].save_html(output_file)
                logger.info(f"✅ Saved group replay {g_idx} to {output_file}")

        # Log successful completion
        total_duration_ms = int((time.time() - rollout_start_time) * 1000)
        log_rollout_complete(
            rollout_id=rollout_id,
            trajectories_count=len(trajectories),
            total_inference_calls=metrics.inference_calls,
            total_duration_ms=total_duration_ms,
        )
        metrics.log_summary()

        # Return trajectories with extraction stats for WandB aggregation
        return {
            "trajectories": trajectories,
            "extraction_stats": {
                "orders_expected": metrics.total_orders_expected,
                "orders_extracted": metrics.total_orders_extracted,
                "empty_responses": metrics.empty_responses,
                "partial_responses": metrics.partial_responses,
                "extraction_rate": metrics.get_extraction_rate(),
            },
            "timing": {
                "volume_reload_s": volume_reload_time,
                "total_s": time.time() - rollout_start_time,
            },
        }

    except Exception as e:
        logger.error(f"❌ Error running rollout: {e}")
        if rollout_id:
            log_rollout_error(rollout_id=rollout_id, error=str(e))
        raise e

    finally:
        await axiom.flush()
