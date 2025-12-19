import asyncio
import os
import random
import time
from dataclasses import dataclass

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

# Constants
POWERS = ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]
BASELINE_BOTS = {
    "random_bot": RandomBot(),
    "chaos_bot": ChaosBot(),
}
# Removed CURRENT_ROLLOUT_LORA global - unreliable across container restarts.
# Instead, just check if adapter files exist on disk before each rollout.


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


@dataclass
class PowerResult:
    """Results from processing a single power's turn."""

    orders: list[str]
    response_data: dict


@dataclass
class AdapterConfig:
    """Configuration for managing LoRA adapters."""

    power_adapters: dict[str, str | None]
    unique_loras: set[str]
    hero_power: str | None

    @classmethod
    def from_params(
        cls,
        lora_name: str | None,
        power_adapters: dict[str, str | None] | None,
        hero_power: str | None,
    ) -> "AdapterConfig":
        """Build adapter config from legacy and new parameters."""
        # Build power_adapters from legacy lora_name if not provided
        if power_adapters is None:
            power_adapters = dict.fromkeys(POWERS, lora_name)

        # Validate power_adapters completeness - missing powers would silently use base model
        missing_powers = set(POWERS) - set(power_adapters.keys())
        if missing_powers:
            print(
                f"⚠️ Warning: power_adapters missing {len(missing_powers)} powers: "
                f"{sorted(missing_powers)}. These powers will use base model (no LoRA). "
                "If this is intentional, explicitly set them to None or 'base_model'."
            )
            # Fill in missing powers with None to make behavior explicit
            for power in missing_powers:
                power_adapters[power] = None

        # Validate hero_power if specified
        if hero_power and hero_power not in POWERS:
            raise ValueError(f"Invalid hero_power '{hero_power}'. Must be one of: {POWERS}")

        # Collect unique LoRA adapters that need to be loaded
        unique_loras = {
            adapter
            for adapter in power_adapters.values()
            if adapter and adapter not in BASELINE_BOTS and adapter != "base_model"
        }

        return cls(
            power_adapters=power_adapters,
            unique_loras=unique_loras,
            hero_power=hero_power,
        )

    def should_collect_power(self, power: str) -> bool:
        """Check if we should collect trajectories for this power."""
        return (self.hero_power is None) or (power == self.hero_power)

    def is_baseline_power(self, power: str) -> bool:
        """Check if power uses a baseline bot."""
        return self.power_adapters.get(power) in BASELINE_BOTS

    def uses_lora(self, power: str | None = None) -> bool:
        """Check if hero power (or any power) uses a LoRA adapter."""
        if power:
            adapter = self.power_adapters.get(power)
            return adapter not in (None, "base_model") and adapter not in BASELINE_BOTS

        # Check if hero uses LoRA, or any power if no hero specified
        if self.hero_power:
            return self.uses_lora(self.hero_power)
        return bool(self.unique_loras)


def ensure_adapters_loaded(adapter_config: AdapterConfig) -> float:
    """
    Ensure all required LoRA adapters are loaded and available.

    Returns:
        Volume reload time in seconds.
    """
    if not adapter_config.unique_loras:
        return 0.0

    logger.info(f"📂 Using LoRA adapters: {adapter_config.unique_loras}")

    # Check if we need to reload (any adapter file missing from disk)
    needs_reload = any(
        not os.path.exists(f"/data/models/{adapter}") for adapter in adapter_config.unique_loras
    )

    reload_time = 0.0
    if needs_reload:
        reload_start = time.time()
        volume.reload()
        reload_time = time.time() - reload_start
        logger.info(f"⏱️ Volume reload took {reload_time:.2f}s")

    # Verify all adapters exist after reload
    for adapter in adapter_config.unique_loras:
        full_path = f"/data/models/{adapter}"
        if os.path.exists(full_path):
            files = os.listdir(full_path)
            logger.info(f"✅ LoRA adapter found at {full_path}. Files: {files}")
        else:
            logger.error(f"❌ LoRA adapter NOT found at: {full_path}")

    return reload_time


def process_baseline_bot(
    game: DiplomacyWrapper,
    power: str,
    adapter: str,
    valid_moves: dict,
) -> PowerResult:
    """Process a single baseline bot's turn."""
    bot = BASELINE_BOTS[adapter]
    orders = bot.get_orders(game, power)

    return PowerResult(
        orders=orders,
        response_data={
            "text": "[BASELINE BOT]",
            "prompt_token_ids": [],
            "token_ids": [],
            "completion_logprobs": [],
        },
    )


def collect_llm_powers(
    inputs: dict,
    power_adapters: dict[str, str | None],
    exclude_indices: set[int],
) -> tuple[list[int], list[str], list[dict], list[str | None]]:
    """
    Collect all LLM powers (non-baseline) for batched inference.

    Returns:
        Tuple of (indices, prompts, valid_moves, lora_names)
    """
    indices: list[int] = []
    prompts: list[str] = []
    valid_moves: list[dict] = []
    lora_names: list[str | None] = []

    for idx, power in enumerate(inputs["power_names"]):
        if idx in exclude_indices:
            continue

        adapter = power_adapters.get(power)
        # Normalize: None and "base_model" both mean no LoRA
        adapter_key = None if adapter in (None, "base_model") else adapter

        indices.append(idx)
        prompts.append(inputs["prompts"][idx])
        valid_moves.append(inputs["valid_moves"][idx])
        lora_names.append(adapter_key)

    return indices, prompts, valid_moves, lora_names


async def run_batched_inference(
    inference_engine_cls,
    base_model_id: str,
    indices: list[int],
    prompts: list[str],
    valid_moves: list[dict],
    lora_names: list[str | None],
    temperature: float,
    max_new_tokens: int,
    metrics: RolloutMetrics,
    stopwatch_prefix: str = "",
) -> dict[int, dict]:
    """
    Run batched inference for all powers in a single call.

    Uses per-prompt LoRA adapters for efficient mixed-adapter batching,
    reducing Modal round-trip overhead from O(adapters) to O(1) per phase.

    Returns:
        Map of original_idx -> response_data
    """
    if not prompts:
        return {}

    batch_size = len(prompts)
    unique_adapters = len({name for name in lora_names if name is not None})
    stopwatch_label = f"{stopwatch_prefix}Batch={batch_size}, Adapters={unique_adapters}"

    with stopwatch(stopwatch_label):
        inference_start = time.time()
        metrics.inference_calls += 1
        responses = await inference_engine_cls(model_id=base_model_id).generate.remote.aio(
            prompts=prompts,
            valid_moves=valid_moves,
            lora_names=lora_names,  # Per-prompt adapters
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        metrics.timing.add("inference", time.time() - inference_start)

    # Map responses back to original indices
    return dict(zip(indices, responses, strict=True))


def log_and_record_orders(
    rollout_id: str,
    power: str,
    orders: list[str],
    expected_count: int,
    response_text: str,
    phase: str,
    metrics: RolloutMetrics,
) -> None:
    """Log order extraction and record metrics."""
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


async def process_game_step(
    game: DiplomacyWrapper,
    agent: LLMAgent,
    adapter_config: AdapterConfig,
    inference_engine_cls,
    cfg: ExperimentConfig,
    metrics: RolloutMetrics,
    rollout_id: str,
    step_count: int = 0,
    collect_fork_data: bool = False,
) -> tuple[list[str], dict]:
    """
    Process a single game step for all powers.

    Returns:
        Tuple of (all_orders, fork_data)
    """
    # Track prompt building time
    prompt_start = time.time()
    inputs = game.get_all_inputs(agent=agent)
    phase = game.get_current_phase()
    metrics.timing.add("prompt_building", time.time() - prompt_start)
    power_results: dict[int, PowerResult] = {}
    fork_data = {}

    # 1. Handle baseline bots directly
    for idx, power in enumerate(inputs["power_names"]):
        adapter = adapter_config.power_adapters.get(power)
        if adapter in BASELINE_BOTS:
            result = process_baseline_bot(game, power, adapter, inputs["valid_moves"][idx])
            power_results[idx] = result

    # 2. Collect LLM powers for batched inference (single call with per-prompt adapters)
    indices, prompts, valid_moves_list, lora_names = collect_llm_powers(
        inputs,
        adapter_config.power_adapters,
        exclude_indices=set(power_results.keys()),
    )

    # 3. Run batched inference (single Modal call for all powers)
    stopwatch_prefix = f"Game Step {step_count} " if step_count else ""
    inference_results = await run_batched_inference(
        inference_engine_cls,
        cfg.base_model_id,
        indices,
        prompts,
        valid_moves_list,
        lora_names,
        cfg.temperature,
        cfg.max_new_tokens,
        metrics,
        stopwatch_prefix,
    )

    # Convert inference results to PowerResult format
    for idx, response_data in inference_results.items():
        response_text = response_data["text"]
        orders = extract_orders(response_text)
        power_results[idx] = PowerResult(orders=orders, response_data=response_data)

    # 4. Collect all orders and log metrics
    all_orders = []
    for idx, power in enumerate(inputs["power_names"]):
        result = power_results[idx]
        response_text = result.response_data["text"]
        expected_count = len(inputs["valid_moves"][idx])

        log_and_record_orders(
            rollout_id,
            power,
            result.orders,
            expected_count,
            response_text,
            phase,
            metrics,
        )

        # Collect fork data if requested
        should_collect = (
            collect_fork_data
            and adapter_config.should_collect_power(power)
            and not adapter_config.is_baseline_power(power)
        )

        if should_collect:
            fork_data[power] = {
                "prompt": inputs["prompts"][idx],
                "completion": response_text,
                "prompt_token_ids": result.response_data.get("prompt_token_ids", []),
                "completion_token_ids": result.response_data.get("token_ids", []),
                "completion_logprobs": result.response_data.get("completion_logprobs", []),
            }

        all_orders.extend(result.orders)

    return all_orders, fork_data


async def run_warmup_phase(
    game: DiplomacyWrapper,
    agent: LLMAgent,
    adapter_config: AdapterConfig,
    inference_engine_cls,
    cfg: ExperimentConfig,
    metrics: RolloutMetrics,
    rollout_id: str,
    warmup_phases: int,
    vis: GameVisualizer | None = None,
    step_timeout_s: float = 120.0,  # Per-step timeout to prevent hangs
) -> bool:
    """
    Run warmup phase to generate random initial state.

    Args:
        step_timeout_s: Timeout in seconds for each step. If inference hangs,
                       the step will fail rather than blocking the entire rollout.

    Returns:
        True if warmup completed successfully, False if game ended early.
    """
    logger.info(f"🔥 Starting Warmup: {warmup_phases} phases")

    for i in range(warmup_phases):
        if game.is_done():
            logger.info("Game ended during warmup, discarding.")
            return False

        try:
            all_orders, _ = await asyncio.wait_for(
                process_game_step(
                    game=game,
                    agent=agent,
                    adapter_config=adapter_config,
                    inference_engine_cls=inference_engine_cls,
                    cfg=cfg,
                    metrics=metrics,
                    rollout_id=rollout_id,
                    step_count=i + 1,
                    collect_fork_data=False,  # Don't collect during warmup
                ),
                timeout=step_timeout_s,
            )
        except TimeoutError:
            logger.error(
                f"❌ Warmup step {i + 1}/{warmup_phases} timed out after {step_timeout_s}s. "
                "This may indicate an inference engine hang."
            )
            return False

        # Track game engine time
        engine_start = time.time()
        game.step(all_orders)
        metrics.timing.add("game_engine", time.time() - engine_start)

        if vis:
            vis_start = time.time()
            vis.capture_turn(
                game.game,
                f"Warmup step {i + 1}/{warmup_phases}\n{chr(10).join(all_orders)}",
            )
            metrics.timing.add("visualization", time.time() - vis_start)

    return True


async def run_game_fork(
    g_idx: int,
    game: DiplomacyWrapper,
    agent: LLMAgent,
    adapter_config: AdapterConfig,
    inference_engine_cls,
    cfg: ExperimentConfig,
    metrics: RolloutMetrics,
    rollout_id: str,
    target_year: int,
    vis_obj: GameVisualizer | None = None,
) -> dict:
    """Run a single game fork asynchronously until completion."""
    game_fork_data = {}
    step_count = 0

    while game.get_year() < target_year and not game.is_done():
        step_count += 1

        all_orders, fork_data = await process_game_step(
            game=game,
            agent=agent,
            adapter_config=adapter_config,
            inference_engine_cls=inference_engine_cls,
            cfg=cfg,
            metrics=metrics,
            rollout_id=rollout_id,
            step_count=step_count,
            collect_fork_data=(step_count == 1),  # Only collect first step
        )

        # Store fork data from first step
        if step_count == 1:
            game_fork_data.update(fork_data)

        # Track game engine time
        engine_start = time.time()
        game.step(all_orders)
        metrics.timing.add("game_engine", time.time() - engine_start)

        if vis_obj:
            vis_start = time.time()
            vis_obj.capture_turn(
                game.game,
                f"Rollout step {step_count}\n{chr(10).join(all_orders)}",
            )
            metrics.timing.add("visualization", time.time() - vis_start)

    return {"g_idx": g_idx, "fork_data": game_fork_data}


# ============================================================================
# SYNCHRONIZED FORK EXECUTION
# ============================================================================


def _log_fork_timing_breakdown(rollout_id: str, step_timings: list[dict]) -> None:
    """Log detailed per-step timing for debugging slow rollouts."""
    if not step_timings:
        return

    total_inference = sum(s.get("inference_ms", 0) for s in step_timings)
    total_game_step = sum(s.get("game_step_ms", 0) for s in step_timings)
    total_input = sum(s.get("input_collection_ms", 0) for s in step_timings)
    total_vis = sum(s.get("visualization_ms", 0) for s in step_timings)
    n_steps = len(step_timings)

    logger.info(
        f"📊 Fork timing breakdown for {rollout_id}:\n"
        f"   Steps: {n_steps}\n"
        f"   Inference: {total_inference:.0f}ms ({total_inference / n_steps:.0f}ms/step)\n"
        f"   Game step: {total_game_step:.0f}ms\n"
        f"   Input collection: {total_input:.0f}ms\n"
        f"   Visualization: {total_vis:.0f}ms"
    )

    # Log to Axiom for dashboard visualization
    axiom.log(
        {
            "event": "fork_timing_breakdown",
            "rollout_id": rollout_id,
            "steps": n_steps,
            "total_inference_ms": total_inference,
            "total_game_step_ms": total_game_step,
            "total_input_collection_ms": total_input,
            "total_visualization_ms": total_vis,
            "avg_inference_per_step_ms": total_inference / max(1, n_steps),
        }
    )


async def process_synchronized_step(
    fork_inputs: list[tuple[int, DiplomacyWrapper, dict]],
    adapter_config: "AdapterConfig",
    inference_engine_cls,
    cfg: ExperimentConfig,
    metrics: RolloutMetrics,
    rollout_id: str,
    step_count: int,
    collect_fork_data: bool = False,
) -> dict[int, tuple[list[str], dict]]:
    """
    Process one step for multiple forks in a single batched inference call.

    Args:
        fork_inputs: List of (g_idx, game, inputs_dict) tuples
        adapter_config: Configuration for power adapters
        inference_engine_cls: Modal inference engine class
        cfg: Experiment configuration
        metrics: Rollout metrics tracker
        rollout_id: ID for logging
        step_count: Current step number
        collect_fork_data: Whether to collect trajectory data

    Returns:
        Dict mapping g_idx → (orders_list, fork_data_dict)
    """
    # 1. Flatten all prompts from all forks
    all_prompts: list[str] = []
    all_valid_moves: list[dict] = []
    all_lora_names: list[str | None] = []
    # Track (g_idx, power_idx, power_name, inputs) for each prompt
    fork_power_map: list[tuple[int, int, str, dict]] = []
    # Track baseline bot results per fork
    baseline_results_by_fork: dict[int, dict[int, PowerResult]] = {}

    for g_idx, game, inputs in fork_inputs:
        baseline_results_by_fork[g_idx] = {}

        # Handle baseline bots first (no inference needed)
        for idx, power in enumerate(inputs["power_names"]):
            adapter = adapter_config.power_adapters.get(power)
            if adapter in BASELINE_BOTS:
                baseline_results_by_fork[g_idx][idx] = process_baseline_bot(
                    game, power, adapter, inputs["valid_moves"][idx]
                )

        # Collect LLM powers
        indices, prompts, valid_moves_list, lora_names = collect_llm_powers(
            inputs,
            adapter_config.power_adapters,
            exclude_indices=set(baseline_results_by_fork[g_idx].keys()),
        )

        # Add to batch with tracking
        for idx, prompt, moves, lora_name in zip(
            indices, prompts, valid_moves_list, lora_names, strict=True
        ):
            all_prompts.append(prompt)
            all_valid_moves.append(moves)
            all_lora_names.append(lora_name)
            fork_power_map.append((g_idx, idx, inputs["power_names"][idx], inputs))

    # 2. Single batched inference call for ALL forks
    inference_results: dict[int, dict] = {}
    if all_prompts:
        batch_indices = list(range(len(all_prompts)))
        inference_results = await run_batched_inference(
            inference_engine_cls,
            cfg.base_model_id,
            batch_indices,
            all_prompts,
            all_valid_moves,
            all_lora_names,
            cfg.temperature,
            cfg.max_new_tokens,
            metrics,
            f"Sync Step {step_count} ",
        )

    # 3. Distribute results back to forks
    results: dict[int, tuple[list[str], dict]] = {}

    for g_idx, game, inputs in fork_inputs:
        power_orders: dict[int, list[str]] = {}
        fork_data_for_step: dict[str, dict] = {}

        # Get baseline bot orders
        for idx, result in baseline_results_by_fork[g_idx].items():
            power_orders[idx] = result.orders

        # Get LLM orders from batch results
        for batch_idx, (fg_idx, power_idx, power_name, fork_inputs_dict) in enumerate(
            fork_power_map
        ):
            if fg_idx != g_idx:
                continue

            response_data = inference_results.get(batch_idx, {})
            response_text = response_data.get("text", "")
            orders = extract_orders(response_text)
            power_orders[power_idx] = orders

            # Log and record metrics
            expected_count = len(fork_inputs_dict["valid_moves"][power_idx])
            log_and_record_orders(
                rollout_id=rollout_id,
                power=power_name,
                orders=orders,
                expected_count=expected_count,
                response_text=response_text,
                phase=game.get_current_phase(),
                metrics=metrics,
            )

            # Collect fork data if needed
            if collect_fork_data and adapter_config.should_collect_power(power_name):
                fork_data_for_step[power_name] = {
                    "prompt": fork_inputs_dict["prompts"][power_idx],
                    "completion": response_text,
                    "prompt_token_ids": response_data.get("prompt_token_ids", []),
                    "completion_token_ids": response_data.get("token_ids", []),
                    "completion_logprobs": response_data.get("completion_logprobs", []),
                }

        # Combine all orders in correct order
        all_orders: list[str] = []
        for idx in range(len(inputs["power_names"])):
            all_orders.extend(power_orders.get(idx, []))

        results[g_idx] = (all_orders, fork_data_for_step)

    return results


async def run_synchronized_forks(
    games: list[DiplomacyWrapper],
    agent: LLMAgent,
    adapter_config: "AdapterConfig",
    inference_engine_cls,
    cfg: ExperimentConfig,
    metrics: RolloutMetrics,
    rollout_id: str,
    target_year: int,
    visualizers: list[GameVisualizer] | None = None,
) -> dict[int, dict]:
    """
    Run all game forks with synchronized inference calls.

    Instead of each fork making independent inference calls,
    batch all forks' prompts together at each step. This reduces
    Modal round-trips from (N_forks × N_steps) to N_steps.

    Args:
        games: List of game instances (one per fork)
        agent: LLM agent for prompt building
        adapter_config: Configuration for power adapters
        inference_engine_cls: Modal inference engine class
        cfg: Experiment configuration
        metrics: Rollout metrics tracker
        rollout_id: ID for logging
        target_year: Year to run games until
        visualizers: Optional list of visualizers (one per fork)

    Returns:
        Dict mapping g_idx → fork_data (trajectory data from step 1)
    """
    fork_data: dict[int, dict] = {g_idx: {} for g_idx in range(len(games))}
    step_count = 0

    # Track which forks are still active (not ended)
    active_forks = set(range(len(games)))

    # Timing accumulators
    step_timings: list[dict] = []

    while active_forks:
        step_count += 1
        step_timing: dict[str, float | int] = {
            "step": step_count,
            "active_forks": len(active_forks),
        }

        # Time: Collect inputs from all active forks
        t0 = time.time()
        fork_inputs: list[tuple[int, DiplomacyWrapper, dict]] = []
        for g_idx in list(active_forks):
            game = games[g_idx]
            if game.get_year() >= target_year or game.is_done():
                active_forks.remove(g_idx)
                continue
            inputs = game.get_all_inputs(agent=agent)
            fork_inputs.append((g_idx, game, inputs))
        step_timing["input_collection_ms"] = (time.time() - t0) * 1000

        if not fork_inputs:
            break

        step_timing["prompts_batched"] = sum(len(inputs["prompts"]) for _, _, inputs in fork_inputs)

        # Time: Process synchronized step (includes inference)
        t0 = time.time()
        results = await process_synchronized_step(
            fork_inputs=fork_inputs,
            adapter_config=adapter_config,
            inference_engine_cls=inference_engine_cls,
            cfg=cfg,
            metrics=metrics,
            rollout_id=rollout_id,
            step_count=step_count,
            collect_fork_data=(step_count == 1),
        )
        step_timing["inference_ms"] = (time.time() - t0) * 1000

        # Time: Apply results to forks
        t0 = time.time()
        for g_idx, game, _ in fork_inputs:
            orders, step_fork_data = results[g_idx]
            game.step(orders)

            if step_count == 1 and step_fork_data:
                fork_data[g_idx] = step_fork_data
        step_timing["game_step_ms"] = (time.time() - t0) * 1000

        # Time: Visualization
        if visualizers:
            t0 = time.time()
            for g_idx, game, _ in fork_inputs:
                if g_idx < len(visualizers) and visualizers[g_idx]:
                    visualizers[g_idx].capture_turn(game.game, f"Sync step {step_count}")
            step_timing["visualization_ms"] = (time.time() - t0) * 1000

        step_timings.append(step_timing)

    # Log detailed timing breakdown
    _log_fork_timing_breakdown(rollout_id, step_timings)

    return fork_data


async def compute_reference_logprobs(
    fork_data: dict[int, dict],
    inference_engine_cls,
    base_model_id: str,
) -> dict[tuple[int, str], float]:
    """
    Compute base model logprobs for all completions.

    Returns:
        Map of (g_idx, power) -> ref_logprobs
    """
    score_prompts = []
    score_completions = []
    score_prompt_token_ids = []
    score_keys = []

    for g_idx, power_data in fork_data.items():
        for power, data in power_data.items():
            prompt_tids = data.get("prompt_token_ids", [])
            if prompt_tids:
                score_prompts.append(data["prompt"])
                score_completions.append(data["completion"])
                score_prompt_token_ids.append(prompt_tids)
                score_keys.append((g_idx, power))

    if not score_prompts:
        return {}

    score_results = await inference_engine_cls(model_id=base_model_id).score.remote.aio(
        prompts=score_prompts,
        completions=score_completions,
        prompt_token_ids_list=score_prompt_token_ids,
    )

    ref_logprobs_map = {}
    for key, result in zip(score_keys, score_results, strict=True):
        ref_logprobs_map[key] = result.get("ref_completion_logprobs")

    logger.info(f"📊 Computed {len(score_results)} reference logprobs (using base model)")
    return ref_logprobs_map


def build_trajectories(
    games: list[DiplomacyWrapper],
    fork_data: dict[int, dict],
    adapter_config: AdapterConfig,
    ref_logprobs_map: dict[tuple[int, str], float],
    game_id: str,
    current_year: int,
    cfg: ExperimentConfig,
) -> tuple[list[dict], dict]:
    """Build trajectory data for training from game results.

    Returns:
        Tuple of (trajectories, game_stats) where game_stats contains:
        - sc_counts: List of SC counts for hero powers at end of games
        - win_bonus_awarded: Number of games where win bonus was achieved
        - total_games: Total number of games processed
    """
    trajectories = []
    sc_counts: list[int] = []
    win_bonus_awarded = 0

    for g_idx, game in enumerate(games):
        # Check for missing or empty fork data - indicates collection failure
        if g_idx not in fork_data:
            print(
                f"⚠️ Warning: Game index {g_idx} missing from fork_data. "
                "This may indicate a rollout collection failure."
            )
            continue
        if not fork_data[g_idx]:
            print(
                f"⚠️ Warning: Empty fork_data for game index {g_idx} (game_id={game_id}). "
                "No trajectories will be generated for this fork. "
                "Check if powers are correctly configured for data collection."
            )
            # Continue to calculate scores but will skip trajectory creation
            # (the inner loops will naturally skip since fork_data[g_idx] is empty)

        final_scores = calculate_final_scores(
            game,
            win_bonus=cfg.win_bonus,
            winner_threshold_sc=cfg.winner_threshold_sc,
        )

        # Track SC counts for hero power (if set) or all powers
        for power in fork_data.get(g_idx, {}).keys():
            if adapter_config.hero_power and power != adapter_config.hero_power:
                continue
            n_sc = len(game.game.powers[power].centers)
            sc_counts.append(n_sc)
            # Check if this power got win bonus (sole leader above threshold)
            if n_sc >= cfg.winner_threshold_sc:
                # Check if sole leader
                all_sc = [len(p.centers) for p in game.game.powers.values()]
                if n_sc == max(all_sc) and all_sc.count(n_sc) == 1:
                    win_bonus_awarded += 1

        for power, data in fork_data.get(g_idx, {}).items():
            if power not in final_scores:
                continue

            # Get reference logprobs if computed
            ref_logprobs = ref_logprobs_map.get((g_idx, power))

            # If no LoRA used, generation logprobs ARE reference logprobs
            if ref_logprobs is None and not adapter_config.uses_lora(power):
                completion_logprobs = data.get("completion_logprobs", [])
                if completion_logprobs:
                    ref_logprobs = sum(completion_logprobs)

            traj = {
                "prompt": data["prompt"],
                "completion": data["completion"],
                "reward": final_scores[power],
                "group_id": f"{game_id}_{power}_{current_year}",
                "prompt_token_ids": data.get("prompt_token_ids", []),
                "completion_token_ids": data.get("completion_token_ids", []),
                "completion_logprobs": data.get("completion_logprobs", []),
            }

            if ref_logprobs is not None:
                traj["ref_logprobs"] = ref_logprobs

            trajectories.append(traj)

    game_stats = {
        "sc_counts": sc_counts,
        "win_bonus_awarded": win_bonus_awarded,
        "total_games": len(games),
    }

    return trajectories, game_stats


def save_visualizations(
    main_game: DiplomacyWrapper,
    games: list[DiplomacyWrapper],
    vis: GameVisualizer | None,
    visualizers: list[GameVisualizer] | None,
) -> None:
    """Save game visualizations to HTML files."""
    if not vis or not visualizers:
        return

    # Save main warmup visualization
    output_file = str(REPLAYS_PATH / f"rollout_{main_game.game.game_id}.html")
    vis.save_html(output_file)
    logger.info(f"✅ Saved replay to {output_file}")

    # Save fork visualizations
    for g_idx, game in enumerate(games):
        output_file = str(REPLAYS_PATH / f"rollout_{game.game.game_id}_{g_idx}.html")
        visualizers[g_idx].save_html(output_file)
        logger.info(f"✅ Saved group replay {g_idx} to {output_file}")


# ============================================================================
# MAIN ROLLOUT FUNCTION
# ============================================================================


@app.function(
    image=cpu_image,
    cpu=1.0,
    memory=2048,
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
    rollout_start_time = time.time()
    rollout_id = ""
    metrics: RolloutMetrics | None = None

    try:
        cfg = ExperimentConfig(**config_dict)
        adapter_config = AdapterConfig.from_params(lora_name, power_adapters, hero_power)

        # Get InferenceEngine class from deployed app
        InferenceEngineCls = modal.Cls.from_name("diplomacy-grpo", "InferenceEngine")

        # Ensure LoRA adapters are loaded
        volume_reload_time = ensure_adapters_loaded(adapter_config)

        # Initialize agent
        prompt_config = PromptConfig(
            compact_mode=cfg.compact_prompts,
            prefix_cache_optimized=cfg.prefix_cache_optimized,
            show_valid_moves=cfg.show_valid_moves,
            show_map_windows=cfg.show_map_windows,
        )
        agent = LLMAgent(config=prompt_config)

        # Determine warmup phases
        warmup_phases = random.randint(0, 8)
        if random.random() < cfg.rollout_no_warmup_chance:
            warmup_phases = 0

        # Initialize main game and metrics
        main_game = DiplomacyWrapper(horizon=99)
        rollout_id = main_game.game.game_id
        metrics = RolloutMetrics(rollout_id=rollout_id)

        log_rollout_start(
            rollout_id=rollout_id,
            warmup_phases=warmup_phases,
            samples_per_group=cfg.samples_per_group,
            horizon_years=cfg.rollout_horizon_years,
        )

        # Initialize visualization
        should_visualize = random.random() < cfg.rollout_visualize_chance
        vis = GameVisualizer() if should_visualize else None
        if vis:
            logger.info("Visualizing game...")
            vis.capture_turn(main_game.game, "Warmup Start")

        # Run warmup phase
        warmup_success = await run_warmup_phase(
            game=main_game,
            agent=agent,
            adapter_config=adapter_config,
            inference_engine_cls=InferenceEngineCls,
            cfg=cfg,
            metrics=metrics,
            rollout_id=rollout_id,
            warmup_phases=warmup_phases,
            vis=vis,
        )

        if not warmup_success:
            return []

        # Fork the game state
        fork_start = time.time()
        frozen_state = cloudpickle.dumps(main_game)
        frozen_vis = cloudpickle.dumps(vis)

        games = [cloudpickle.loads(frozen_state) for _ in range(cfg.samples_per_group)]
        visualizers = (
            [cloudpickle.loads(frozen_vis) for _ in range(cfg.samples_per_group)]
            if should_visualize
            else None
        )
        metrics.timing.add("forking", time.time() - fork_start)

        current_year = main_game.get_year()
        target_year = current_year + cfg.rollout_horizon_years

        # Run all game forks with synchronized inference calls
        # This batches all forks' prompts together at each step, reducing Modal round-trips
        # from 60 (4 forks × 15 steps) to 15 (1 call per step)
        with stopwatch("Synchronized Fork Rollout"):
            fork_data = await run_synchronized_forks(
                games=games,
                agent=agent,
                adapter_config=adapter_config,
                inference_engine_cls=InferenceEngineCls,
                cfg=cfg,
                metrics=metrics,
                rollout_id=rollout_id,
                target_year=target_year,
                visualizers=visualizers,
            )

        # Compute reference logprobs if needed
        ref_logprobs_map = {}
        if cfg.compute_ref_logprobs_in_rollout and adapter_config.uses_lora():
            with stopwatch("Reference Logprob Scoring"):
                ref_start = time.time()
                ref_logprobs_map = await compute_reference_logprobs(
                    fork_data,
                    InferenceEngineCls,
                    cfg.base_model_id,
                )
                metrics.timing.add("ref_scoring", time.time() - ref_start)

        # Build trajectories
        trajectories, game_stats = build_trajectories(
            games=games,
            fork_data=fork_data,
            adapter_config=adapter_config,
            ref_logprobs_map=ref_logprobs_map,
            game_id=main_game.game.game_id,
            current_year=current_year,
            cfg=cfg,
        )

        # Save visualizations
        vis_save_start = time.time()
        save_visualizations(main_game, games, vis, visualizers)
        metrics.timing.add("visualization", time.time() - vis_save_start)

        # Calculate total and other time
        total_s = time.time() - rollout_start_time
        tracked_time = metrics.timing.total()
        other_time = max(0, total_s - tracked_time - volume_reload_time)
        metrics.timing.add("other", other_time)

        # Log completion
        total_duration_ms = int(total_s * 1000)
        log_rollout_complete(
            rollout_id=rollout_id,
            trajectories_count=len(trajectories),
            total_inference_calls=metrics.inference_calls,
            total_duration_ms=total_duration_ms,
        )
        metrics.log_summary()

        return {
            "trajectories": trajectories,
            "extraction_stats": {
                "orders_expected": metrics.total_orders_expected,
                "orders_extracted": metrics.total_orders_extracted,
                "empty_responses": metrics.empty_responses,
                "partial_responses": metrics.partial_responses,
                "extraction_rate": metrics.get_extraction_rate(),
            },
            "game_stats": game_stats,
            "timing": {
                "volume_reload_s": volume_reload_time,
                "total_s": total_s,
                # Detailed timing breakdown for waterfall charts
                **metrics.timing.to_dict(),
            },
        }

    except Exception as e:
        logger.error(f"❌ Error running rollout: {e}")
        if rollout_id:
            log_rollout_error(rollout_id=rollout_id, error=str(e))
        raise e

    finally:
        await axiom.flush()
