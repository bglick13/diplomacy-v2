import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import modal
import wandb

from src.agents import LLMAgent, PromptConfig
from src.agents.baselines import ChaosBot, RandomBot
from src.apps.common.images import cpu_image
from src.apps.common.volumes import EVALS_PATH, VOLUME_PATH, volume
from src.engine.wrapper import DiplomacyWrapper
from src.league import LeagueRegistry, update_elo_from_match
from src.league.types import MatchResult
from src.utils.observability import axiom, logger
from src.utils.parsing import extract_orders
from src.utils.scoring import calculate_final_scores

app = modal.App("diplomacy-grpo-evaluation")

# Constants
POWERS = ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]
BASELINE_BOTS = {
    "random_bot": RandomBot(),
    "chaos_bot": ChaosBot(),
    "base_model": None,  # Placeholder for base model
}


# ============================================================================
# SHARED DATA STRUCTURES
# ============================================================================


@dataclass
class GameTiming:
    """Timing breakdown for a single game step."""

    step: int
    prompt_time_s: float
    inference_time_s: float
    combine_time_s: float
    step_game_time_s: float
    total_time_s: float
    num_llm_powers: int
    num_baseline_powers: int


@dataclass
class GameResult:
    """Result from a completed game."""

    game_id: str
    final_scores: dict[str, float]
    power_agents: dict[str, str]
    step_timings: list[GameTiming]
    num_years: int


@dataclass
class MatchSummary:
    """Summary of a single match for logging."""

    gatekeeper: str
    game_idx: int
    challenger_score: float
    gatekeeper_avg_score: float
    win: bool


# ============================================================================
# HELPER FUNCTIONS - AGENT INITIALIZATION
# ============================================================================


def create_llm_agent(
    compact_prompts: bool,
    prefix_cache_optimized: bool,
    show_valid_moves: bool,
) -> LLMAgent:
    """Create LLM agent with consistent prompt configuration."""
    prompt_config = PromptConfig(
        compact_mode=compact_prompts,
        prefix_cache_optimized=prefix_cache_optimized,
        show_valid_moves=show_valid_moves,
        show_map_windows=True,
    )
    return LLMAgent(config=prompt_config)


def get_baseline_agent(opponent_type: str) -> RandomBot | ChaosBot:
    """Get baseline agent by type."""
    if opponent_type == "random":
        return RandomBot()
    elif opponent_type == "chaos":
        return ChaosBot()
    else:
        logger.warning(f"Unknown opponent type: {opponent_type}, using random")
        return RandomBot()


# ============================================================================
# HELPER FUNCTIONS - GAME EXECUTION
# ============================================================================


async def run_game_step(
    game: DiplomacyWrapper,
    power_adapters: dict[str, str | None],
    llm_agent: LLMAgent,
    InferenceEngineCls: Any,
    model_id: str,
    temperature: float,
    max_new_tokens: int,
    is_league_eval: bool,
) -> GameTiming:
    """
    Execute a single game step with batched inference.

    Groups powers by adapter for efficient batching, then combines results.

    Args:
        game: The diplomacy game wrapper
        power_adapters: Mapping of power -> adapter path (or baseline bot name)
        llm_agent: LLM agent for prompt building
        InferenceEngineCls: Modal class for inference engine
        model_id: Base model ID
        temperature: Sampling temperature
        max_new_tokens: Max tokens to generate
        is_league_eval: Whether this is league evaluation (affects engine pool)

    Returns:
        GameTiming with performance breakdown
    """
    step_start = time.time()

    # Phase 1: Collect prompts and group by adapter
    prompt_start = time.time()
    adapter_groups: dict[str | None, list[tuple[str, str, dict]]] = {}
    baseline_orders: dict[str, list[str]] = {}

    for power in POWERS:
        adapter = power_adapters.get(power)

        # Get orders based on adapter type
        if adapter in BASELINE_BOTS and adapter is not None:
            bot = BASELINE_BOTS[adapter]
            if bot:
                orders = bot.get_orders(game, power)
                baseline_orders[power] = orders
            else:
                baseline_orders[power] = []
        else:
            # LLM power - collect prompt for batching
            prompt, valid_moves = llm_agent.build_prompt(game, power)
            if valid_moves:
                adapter_key = adapter or "base_model"
                if adapter_key not in adapter_groups:
                    adapter_groups[adapter_key] = []
                adapter_groups[adapter_key].append((power, prompt, valid_moves))
            else:
                baseline_orders[power] = []

    prompt_time = time.time() - prompt_start

    # Phase 2: Batch inference for each adapter group
    inference_start = time.time()
    inference_results: dict[str, list[str]] = {}  # power -> orders

    for adapter_key, group_items in adapter_groups.items():
        group_powers = [item[0] for item in group_items]
        group_prompts = [item[1] for item in group_items]
        group_valid_moves = [item[2] for item in group_items]

        # Batch inference call
        batch_start = time.time()
        responses = await InferenceEngineCls(  # pyright: ignore[reportCallIssue]
            model_id=model_id,
            # Hardcode is_for_league_evaluation for now
            is_for_league_evaluation=False,
        ).generate.remote.aio(
            prompts=group_prompts,
            valid_moves=group_valid_moves,
            lora_name=adapter_key if adapter_key != "base_model" else None,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        batch_time = time.time() - batch_start

        # Extract orders for each power
        for power, response_data in zip(group_powers, responses, strict=True):
            orders = extract_orders(response_data["text"])
            inference_results[power] = orders

        # Log batch timing (warn for slow single-item batches)
        batch_size = len(group_prompts)
        if batch_size == 1 and batch_time > 5.0:
            logger.warning(
                f"  Very slow single-item batch: adapter={adapter_key}, "
                f"time={batch_time:.3f}s (normal: 2-3s)"
            )
        else:
            logger.debug(
                f"  Batch inference: adapter={adapter_key}, "
                f"batch_size={batch_size}, time={batch_time:.3f}s"
            )

    inference_time = time.time() - inference_start

    # Phase 3: Combine all orders
    combine_start = time.time()
    all_orders = []
    for power in POWERS:
        if power in baseline_orders:
            all_orders.extend(baseline_orders[power])
        elif power in inference_results:
            all_orders.extend(inference_results[power])
    combine_time = time.time() - combine_start

    # Phase 4: Step game
    step_game_start = time.time()
    game.step(all_orders)
    step_game_time = time.time() - step_game_start

    step_total = time.time() - step_start

    return GameTiming(
        step=0,  # Will be set by caller
        prompt_time_s=prompt_time,
        inference_time_s=inference_time,
        combine_time_s=combine_time,
        step_game_time_s=step_game_time,
        total_time_s=step_total,
        num_llm_powers=sum(len(group) for group in adapter_groups.values()),
        num_baseline_powers=len(baseline_orders),
    )


async def run_full_game(
    power_adapters: dict[str, str | None],
    llm_agent: LLMAgent,
    InferenceEngineCls: Any,
    model_id: str,
    max_years: int,
    temperature: float,
    max_new_tokens: int,
    is_league_eval: bool,
) -> GameResult:
    """
    Run a complete game to completion.

    Args:
        power_adapters: Mapping of power -> adapter path
        llm_agent: LLM agent for prompts
        InferenceEngineCls: Modal inference engine class
        model_id: Base model ID
        max_years: Maximum game length
        temperature: Sampling temperature
        max_new_tokens: Max tokens to generate
        is_league_eval: Whether this is league evaluation

    Returns:
        GameResult with final scores and timing data
    """
    game = DiplomacyWrapper(horizon=max_years * 2)
    game_id = game.game.game_id
    step_timings: list[GameTiming] = []
    step_count = 0

    while not game.is_done():
        step_count += 1

        timing = await run_game_step(
            game=game,
            power_adapters=power_adapters,
            llm_agent=llm_agent,
            InferenceEngineCls=InferenceEngineCls,
            model_id=model_id,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            is_league_eval=is_league_eval,
        )
        timing.step = step_count
        step_timings.append(timing)

        # Log every 5 steps to avoid spam
        if step_count % 5 == 0:
            logger.info(
                f"  Step {step_count}: {timing.total_time_s:.3f}s total "
                f"(prompt: {timing.prompt_time_s:.3f}s, inference: {timing.inference_time_s:.3f}s, "
                f"game: {timing.step_game_time_s:.3f}s)"
            )

    # Compute final scores
    final_scores = calculate_final_scores(game)

    # Build power_agents mapping
    power_agents = {p: power_adapters[p] or "base_model" for p in POWERS}

    return GameResult(
        game_id=game_id,
        final_scores=final_scores,
        power_agents=power_agents,
        step_timings=step_timings,
        num_years=(len(step_timings) + 1) // 2,  # 2 phases per year, ceiling division
    )


# ============================================================================
# HELPER FUNCTIONS - LEAGUE EVALUATION
# ============================================================================


def ensure_challenger_in_registry(
    challenger_path: str,
    registry: LeagueRegistry,
) -> None:
    """
    Ensure challenger is in registry, adding it if missing.

    This handles the race condition where evaluation starts before the
    trainer adds the checkpoint to the registry.

    Args:
        challenger_path: Path to challenger adapter
        registry: League registry
    """
    all_agents = registry.get_all_agents()
    challenger_name = challenger_path

    if challenger_name in [a.name for a in all_agents]:
        return  # Already in registry

    logger.warning(f"⚠️ Challenger {challenger_name} not in registry yet, adding it...")

    # Extract step and run_name from path (e.g., "grpo-20251209-200240/adapter_v20" -> 20)
    step = 0
    run_name = None
    try:
        parts = challenger_path.rsplit("/", 1)
        if len(parts) == 2:
            run_name = parts[0]
        step_str = challenger_path.split("adapter_v")[-1]
        step = int(step_str)
    except (ValueError, IndexError):
        logger.warning(f"Could not extract step from challenger_path: {challenger_path}")

    # Find parent checkpoint to inherit Elo
    current_elos = {a.name: a.elo for a in all_agents}
    parent_name = None
    parent_elo = 1000.0

    if step > 1 and run_name:
        # Try to find parent checkpoint (previous step)
        for prev_step in range(step - 1, 0, -1):
            potential_parent = f"{run_name}/adapter_v{prev_step}"
            if potential_parent in current_elos:
                parent_name = potential_parent
                parent_elo = current_elos[potential_parent]
                logger.info(f"  Inheriting Elo {parent_elo:.0f} from parent {parent_name}")
                break
    elif step == 1:
        parent_name = "base_model"
        parent_elo = current_elos.get("base_model", 1000.0)

    # Add challenger to registry
    registry.add_checkpoint(
        name=challenger_name,
        path=challenger_path,
        step=step,
        parent=parent_name,
        initial_elo=parent_elo,
    )


def compute_elo_updates(
    matches: list[GameResult],
    registry: LeagueRegistry,
    k: float = 32.0,
) -> dict[str, float]:
    """
    Compute Elo updates from match results.

    Args:
        matches: List of game results
        registry: League registry for current Elos
        k: Elo K-factor

    Returns:
        Dict mapping agent name to new Elo
    """
    # Get current Elos
    all_agents = registry.get_all_agents()
    current_elos = {a.name: a.elo for a in all_agents}

    # Apply Elo updates from all matches
    for match in matches:
        current_elos = update_elo_from_match(
            power_agents=match.power_agents,
            power_scores=match.final_scores,
            agent_elos=current_elos,
            k=k,
        )

    return current_elos


def create_match_results(
    matches: list[GameResult],
    challenger_name: str,
    training_step: int,
) -> list[MatchResult]:
    """
    Convert GameResult objects to MatchResult for registry.

    Args:
        matches: List of game results
        challenger_name: Name of the challenger
        training_step: Current training step

    Returns:
        List of MatchResult objects
    """
    match_results = []

    for idx, match in enumerate(matches):
        # Calculate rankings from scores
        sorted_powers = sorted(match.final_scores.items(), key=lambda x: x[1], reverse=True)
        rankings = {power: rank + 1 for rank, (power, _) in enumerate(sorted_powers)}

        # Determine winner (agent with highest score)
        winner_power = sorted_powers[0][0] if sorted_powers else None
        winner_agent = match.power_agents.get(winner_power) if winner_power else None

        match_result = MatchResult(
            game_id=f"eval-{challenger_name}-{int(time.time())}-{idx}",
            step=training_step,
            power_agents=match.power_agents,
            scores=match.final_scores,
            rankings=rankings,
            num_years=match.num_years,
            winner=winner_agent,
        )
        match_results.append(match_result)

    return match_results


def log_game_timing(
    game_idx: int,
    step_timings: list[GameTiming],
    challenger_path: str,
    gatekeeper_name: str,
) -> None:
    """Log timing summary for a completed game."""
    if not step_timings:
        return

    step_count = len(step_timings)
    total_time = sum(s.total_time_s for s in step_timings)
    avg_inference_time = sum(s.inference_time_s for s in step_timings) / step_count
    avg_prompt_time = sum(s.prompt_time_s for s in step_timings) / step_count
    avg_game_time = sum(s.step_game_time_s for s in step_timings) / step_count

    logger.info(
        f"  Game {game_idx} complete: {step_count} steps, {total_time:.2f}s total "
        f"(avg: prompt={avg_prompt_time:.3f}s, inference={avg_inference_time:.3f}s, "
        f"game={avg_game_time:.3f}s)"
    )

    # Log to Axiom for analysis
    axiom.log(
        {
            "event": "league_eval_game_timing",
            "challenger_path": challenger_path,
            "gatekeeper": gatekeeper_name,
            "game_idx": game_idx,
            "total_steps": step_count,
            "total_time_s": total_time,
            "avg_prompt_time_s": avg_prompt_time,
            "avg_inference_time_s": avg_inference_time,
            "avg_game_time_s": avg_game_time,
            "num_llm_powers": step_timings[-1].num_llm_powers if step_timings else 0,
        }
    )


def log_elo_to_wandb(
    wandb_run_id: str,
    challenger_elo: float,
    win_rate: float,
    total_games: int,
    training_step: int,
    all_elos: dict[str, float],
    all_agents: list[Any],
    opponent_win_rates: dict[str, float] | None = None,
) -> None:
    """Log Elo metrics to WandB."""
    try:
        wandb.init(id=wandb_run_id, resume="allow", project="diplomacy-grpo")
        wandb.log(
            {
                "elo/challenger": challenger_elo,
                "elo/win_rate": win_rate,
                "elo/games_played": total_games,
                "elo/evaluation_step": training_step,
            }
        )
        # Log per-opponent-type win rates
        if opponent_win_rates:
            for opp_type, opp_rate in opponent_win_rates.items():
                wandb.log({f"elo/vs_{opp_type}_win_rate": opp_rate})
        # Log Elo for all tracked agents
        for agent_name, elo in all_elos.items():
            if agent_name in [a.name for a in all_agents]:
                safe_name = agent_name.replace("/", "_")
                wandb.log({f"elo/{safe_name}": elo})
    except Exception as e:
        logger.warning(f"Failed to log to WandB: {e}")


# ============================================================================
# LEAGUE EVALUATION FUNCTION
# ============================================================================


@app.function(
    image=cpu_image,
    timeout=60 * 60,  # 1 hour max per evaluation
    retries=0,  # Don't retry - if it fails, training continues
    secrets=[
        modal.Secret.from_name("axiom-secrets"),
        modal.Secret.from_name("wandb-secret"),
    ],
    volumes={str(VOLUME_PATH): volume},
)
async def evaluate_league(
    challenger_path: str,
    league_registry_path: str,
    games_per_opponent: int = 3,
    max_years: int = 5,
    model_id: str = "Qwen/Qwen2.5-7B-Instruct",
    wandb_run_id: str | None = None,
    training_step: int = 0,
    # Prompt/inference settings (should match training config)
    show_valid_moves: bool = True,
    compact_prompts: bool = True,
    prefix_cache_optimized: bool = True,
    temperature: float = 0.8,
    max_new_tokens: int = 256,
) -> dict:
    """
    Run Elo evaluation for a new checkpoint against gatekeepers.

    This function is spawned asynchronously during training to update Elo
    ratings without blocking the training loop.

    Gatekeepers are:
    - All baseline bots (random_bot, chaos_bot)
    - Top N checkpoints by Elo from the registry

    Args:
        challenger_path: Path to the new checkpoint (e.g., "run-name/adapter_v50")
        league_registry_path: Path to league.json
        games_per_opponent: Games per gatekeeper (default 3 for speed)
        max_years: Max years per game
        model_id: Base model ID
        wandb_run_id: WandB run ID to log to (from training)
        training_step: Current training step (for logging)
        show_valid_moves: Include valid moves in prompts (should match training)
        compact_prompts: Use compact prompt format (should match training)
        prefix_cache_optimized: Optimize prompts for prefix caching (should match training)
        temperature: Sampling temperature (should match training)
        max_new_tokens: Max tokens to generate (should match training)

    Returns:
        Dict with Elo updates and match results
    """
    # Get InferenceEngine class from the deployed app at runtime
    InferenceEngineCls = modal.Cls.from_name("diplomacy-grpo", "InferenceEngine")

    eval_start = time.time()
    logger.info(f"🏆 Starting Elo evaluation for {challenger_path}")

    # CRITICAL: Reload volume to get latest registry updates from trainer
    volume.reload()
    registry = LeagueRegistry(Path(league_registry_path))

    # Ensure challenger is in registry (handles race condition)
    ensure_challenger_in_registry(challenger_path, registry)

    # Get gatekeepers: baselines + top checkpoints
    baselines = registry.get_baselines()
    checkpoints = registry.get_checkpoints()

    # Select top 3 checkpoints by Elo (excluding the challenger itself)
    top_checkpoints = sorted(
        [c for c in checkpoints if c.path != challenger_path],
        key=lambda x: x.elo,
        reverse=True,
    )[:3]

    gatekeepers = baselines + top_checkpoints
    logger.info(f"📊 Gatekeepers: {[g.name for g in gatekeepers]}")

    # Initialize LLM agent
    llm_agent = create_llm_agent(compact_prompts, prefix_cache_optimized, show_valid_moves)

    # Track all matches for Elo updates
    all_matches: list[GameResult] = []
    match_summaries: list[MatchSummary] = []

    for gatekeeper in gatekeepers:
        logger.info(f"  vs {gatekeeper.name}...")

        for game_idx in range(games_per_opponent):
            # Randomly assign challenger to a power
            challenger_power = random.choice(POWERS)
            opponent_powers = [p for p in POWERS if p != challenger_power]

            # Build power_adapters
            power_adapters: dict[str, str | None] = {}
            power_adapters[challenger_power] = challenger_path

            # Assign gatekeeper to all other powers
            for p in opponent_powers:
                if gatekeeper.agent_type.value == "baseline":
                    power_adapters[p] = gatekeeper.name  # "random_bot" or "chaos_bot"
                else:
                    power_adapters[p] = gatekeeper.path  # Checkpoint path

            # Run game
            game_result = await run_full_game(
                power_adapters=power_adapters,
                llm_agent=llm_agent,
                InferenceEngineCls=InferenceEngineCls,
                model_id=model_id,
                max_years=max_years,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                is_league_eval=True,  # CRITICAL: Use league evaluation pool
            )

            all_matches.append(game_result)

            # Log timing
            log_game_timing(game_idx, game_result.step_timings, challenger_path, gatekeeper.name)

            # Track for summary
            challenger_score = game_result.final_scores[challenger_power]
            gatekeeper_scores = [
                game_result.final_scores[p]
                for p in opponent_powers
                if power_adapters[p] == gatekeeper.path or power_adapters[p] == gatekeeper.name
            ]
            gatekeeper_avg = (
                sum(gatekeeper_scores) / len(gatekeeper_scores) if gatekeeper_scores else 0.0
            )

            match_summaries.append(
                MatchSummary(
                    gatekeeper=gatekeeper.name,
                    game_idx=game_idx,
                    challenger_score=challenger_score,
                    gatekeeper_avg_score=gatekeeper_avg,
                    win=challenger_score > gatekeeper_avg,
                )
            )

            logger.info(
                f"    Game {game_idx + 1}: Challenger {challenger_score:.1f} vs Gatekeeper avg {gatekeeper_avg:.1f}"
            )

    # Compute Elo updates from all matches
    logger.info("📈 Computing Elo updates...")
    updated_elos = compute_elo_updates(all_matches, registry)

    # Update registry with new Elos
    registry.bulk_update_elos(updated_elos)

    # Add match history
    match_results = create_match_results(all_matches, challenger_path, training_step)
    for match_result in match_results:
        registry.add_match(match_result)

    # Final save and commit
    registry._save_unlocked()
    volume.commit()

    # Compute summary stats
    challenger_new_elo = updated_elos.get(challenger_path, 1000.0)
    wins = sum(1 for m in match_summaries if m.win)
    total_games = len(match_summaries)
    win_rate = wins / total_games if total_games > 0 else 0.0

    # Compute per-opponent-type win rates
    opponent_win_rates: dict[str, float] = {}
    for opponent_type in ["random_bot", "chaos_bot", "checkpoint"]:
        if opponent_type == "checkpoint":
            # All non-baseline opponents
            type_matches = [m for m in match_summaries if m.gatekeeper not in BASELINE_BOTS]
        else:
            type_matches = [m for m in match_summaries if m.gatekeeper == opponent_type]

        if type_matches:
            type_wins = sum(1 for m in type_matches if m.win)
            opponent_win_rates[opponent_type] = type_wins / len(type_matches)

    eval_duration = time.time() - eval_start

    logger.info(f"✅ Elo evaluation complete in {eval_duration:.1f}s")
    logger.info(f"   Challenger Elo: {challenger_new_elo:.0f}")
    logger.info(f"   Win rate: {win_rate:.1%} ({wins}/{total_games})")
    for opp_type, opp_rate in opponent_win_rates.items():
        logger.info(f"   vs {opp_type}: {opp_rate:.1%}")

    # Log to WandB if run ID provided
    if wandb_run_id:
        all_agents = registry.get_all_agents()
        log_elo_to_wandb(
            wandb_run_id,
            challenger_new_elo,
            win_rate,
            total_games,
            training_step,
            updated_elos,
            all_agents,
            opponent_win_rates,
        )

    await axiom.flush()

    return {
        "challenger_path": challenger_path,
        "challenger_elo": challenger_new_elo,
        "win_rate": win_rate,
        "opponent_win_rates": opponent_win_rates,
        "games_played": total_games,
        "elo_updates": updated_elos,
        "duration_s": eval_duration,
    }


# ============================================================================
# STANDARD EVALUATION RUNNER
# ============================================================================


@app.function(
    image=cpu_image,
    timeout=86400,  # 24 hours max
    secrets=[
        modal.Secret.from_name("axiom-secrets"),
        modal.Secret.from_name("wandb-secret"),
    ],
    volumes={str(VOLUME_PATH): volume},
)
async def run_evaluation(
    checkpoint_path: str,
    opponents: list[str] | None = None,
    games_per_opponent: int = 10,
    max_years: int = 10,
    eval_powers: list[str] | None = None,
    model_id: str = "Qwen/Qwen2.5-7B-Instruct",
    visualize: bool = True,
    visualize_sample_rate: float = 0.3,
    log_to_wandb: bool = True,
    wandb_run_name: str | None = None,
    # Prompt/inference settings
    show_valid_moves: bool = True,
    compact_prompts: bool = True,
    prefix_cache_optimized: bool = True,
    temperature: float = 0.8,
    max_new_tokens: int = 256,
) -> dict:
    """
    Evaluate a checkpoint against various opponents.

    This function runs evaluation games and logs results to WandB with
    HTML visualizations of sample games.

    Args:
        checkpoint_path: Path to checkpoint (relative to /data/models)
                        e.g., "benchmark-20251205/adapter_v10"
        opponents: List of opponent types ["random", "chaos"]
        games_per_opponent: Number of games per opponent type
        max_years: Maximum game length in years
        eval_powers: Which powers use the checkpoint (default: ["FRANCE"])
        model_id: Base model ID
        visualize: Whether to generate HTML visualizations
        visualize_sample_rate: Fraction of games to visualize
        log_to_wandb: Whether to log results to WandB
        wandb_run_name: Optional WandB run name

    Returns:
        Dict with evaluation metrics and visualization paths
    """
    from datetime import datetime

    from src.utils.vis import GameVisualizer

    # Get InferenceEngine class from the deployed app at runtime
    InferenceEngineCls = modal.Cls.from_name("diplomacy-grpo", "InferenceEngine")

    # Ensure evals directory exists
    EVALS_PATH.mkdir(parents=True, exist_ok=True)

    # Default opponents
    if opponents is None:
        opponents = ["random", "chaos"]

    # Default eval powers (FRANCE for consistency)
    if eval_powers is None:
        eval_powers = ["FRANCE"]

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    original_run_name = "/".join(checkpoint_path.split("/")[:-1])
    checkpoint_name = checkpoint_path.split("/")[-1]

    logger.info("=" * 60)
    logger.info(f"🎯 EVALUATION: {checkpoint_path}")
    logger.info("=" * 60)
    logger.info(f"Opponents: {opponents}")
    logger.info(f"Games per opponent: {games_per_opponent}")
    logger.info(f"Eval powers: {eval_powers}")
    logger.info(f"Max years: {max_years}")

    # Log to Axiom
    axiom.log(
        {
            "event": "evaluation_start",
            "checkpoint_path": checkpoint_path,
            "opponents": opponents,
            "games_per_opponent": games_per_opponent,
            "timestamp": timestamp,
        }
    )

    # Initialize WandB
    if log_to_wandb:
        run_name = wandb_run_name or f"eval-{original_run_name}-{checkpoint_name}-{timestamp}"
        wandb.init(
            project="diplomacy-grpo",
            name=run_name,
            tags=["evaluation"],
            config={
                "checkpoint_path": checkpoint_path,
                "opponents": opponents,
                "games_per_opponent": games_per_opponent,
                "max_years": max_years,
                "eval_powers": eval_powers,
                "model_id": model_id,
            },
        )

    # Initialize LLM agent
    llm_agent = create_llm_agent(compact_prompts, prefix_cache_optimized, show_valid_moves)

    # Results storage
    all_results = []
    all_visualizations = []

    start_time = time.time()

    for opponent_type in opponents:
        logger.info(f"\n📊 Running {games_per_opponent} games vs {opponent_type}...")

        opponent_agent = get_baseline_agent(opponent_type)
        opponent_results = []

        for game_idx in range(games_per_opponent):
            random.seed(42 + game_idx)  # Reproducible

            # Initialize game
            game = DiplomacyWrapper(horizon=max_years * 2)
            game_id = game.game.game_id

            # Visualization for this game
            should_viz = visualize and (random.random() < visualize_sample_rate)
            vis = GameVisualizer() if should_viz else None
            if vis:
                vis.capture_turn(game.game, f"Game Start vs {opponent_type}")

            # Track metrics
            checkpoint_centers_history = []
            all_powers = list(game.game.powers.keys())
            checkpoint_power_set = set(eval_powers)

            # Game loop
            year = 0
            while not game.is_done() and year < max_years:
                year = game.get_year() - 1901
                all_orders = []

                # Collect prompts for checkpoint powers
                checkpoint_prompts = []
                checkpoint_valid_moves = []
                active_checkpoint_powers = []

                for power_name in all_powers:
                    power = game.game.powers[power_name]
                    if len(power.units) == 0 and len(power.centers) == 0:
                        continue  # Eliminated

                    if power_name in checkpoint_power_set:
                        prompt, valid_moves = llm_agent.build_prompt(game, power_name)
                        checkpoint_prompts.append(prompt)
                        checkpoint_valid_moves.append(valid_moves)
                        active_checkpoint_powers.append(power_name)
                    else:
                        orders = opponent_agent.get_orders(game, power_name)
                        all_orders.extend(orders)

                # Batch inference for checkpoint powers
                if checkpoint_prompts:
                    responses = await InferenceEngineCls(model_id=model_id).generate.remote.aio(  # pyright: ignore[reportCallIssue]
                        prompts=checkpoint_prompts,
                        valid_moves=checkpoint_valid_moves,
                        lora_name=checkpoint_path,
                        temperature=temperature,
                        max_new_tokens=max_new_tokens,
                    )

                    for response_data, valid_moves in zip(
                        responses, checkpoint_valid_moves, strict=True
                    ):
                        orders = extract_orders(response_data["text"])
                        if not orders:
                            # Fallback: use first valid move for each unit
                            orders = [moves[0] for moves in valid_moves.values() if moves]
                        all_orders.extend(orders)

                # Step game
                game.step(all_orders)

                # Track checkpoint centers
                total_ckpt_centers = sum(
                    len(game.game.powers[p].centers)
                    for p in checkpoint_power_set
                    if p in game.game.powers
                )
                checkpoint_centers_history.append(total_ckpt_centers)

                # Capture visualization
                if vis:
                    phase = game.get_current_phase()
                    vis.capture_turn(game.game, f"Year {game.get_year()} - {phase}")

            # Game complete - collect results
            game_result = {
                "game_id": game_id,
                "opponent": opponent_type,
                "num_years": game.get_year() - 1901,
                "checkpoint_powers": {},
                "opponent_powers": {},
            }

            for power_name in all_powers:
                power = game.game.powers[power_name]
                power_data = {
                    "final_centers": len(power.centers),
                    "final_units": len(power.units),
                    "survived": len(power.centers) > 0 or len(power.units) > 0,
                    "won": len(power.centers) >= 18,
                }

                if power_name in checkpoint_power_set:
                    game_result["checkpoint_powers"][power_name] = power_data
                else:
                    game_result["opponent_powers"][power_name] = power_data

            opponent_results.append(game_result)

            # Save visualization
            if vis:
                vis_filename = f"eval_{checkpoint_name}_{opponent_type}_{game_idx}.html"
                vis_path = str(EVALS_PATH / original_run_name / "replays" / vis_filename)
                vis.save_html(vis_path)
                all_visualizations.append(
                    {
                        "path": vis_path,
                        "game_id": game_id,
                        "opponent": opponent_type,
                    }
                )
                logger.info(f"  Saved visualization: {vis_path}")

            # Log progress
            ckpt_centers = sum(
                p["final_centers"] for p in game_result["checkpoint_powers"].values()
            )
            logger.info(
                f"  Game {game_idx + 1}/{games_per_opponent}: "
                f"{ckpt_centers} centers, {game_result['num_years']} years"
            )

        # Compute metrics for this opponent
        total_games = len(opponent_results)
        total_wins = sum(
            1 for g in opponent_results for p in g["checkpoint_powers"].values() if p["won"]
        )
        total_survivals = sum(
            1 for g in opponent_results for p in g["checkpoint_powers"].values() if p["survived"]
        )
        total_checkpoint_powers = sum(len(g["checkpoint_powers"]) for g in opponent_results)
        avg_centers = (
            sum(
                p["final_centers"]
                for g in opponent_results
                for p in g["checkpoint_powers"].values()
            )
            / total_checkpoint_powers
            if total_checkpoint_powers > 0
            else 0
        )

        opponent_metrics = {
            "opponent": opponent_type,
            "games": total_games,
            "win_rate": total_wins / total_checkpoint_powers if total_checkpoint_powers > 0 else 0,
            "survival_rate": total_survivals / total_checkpoint_powers
            if total_checkpoint_powers > 0
            else 0,
            "avg_centers": avg_centers,
            "results": opponent_results,
        }

        all_results.append(opponent_metrics)

        logger.info(f"\n  vs {opponent_type} Summary:")
        logger.info(f"    Win Rate: {opponent_metrics['win_rate']:.1%}")
        logger.info(f"    Survival Rate: {opponent_metrics['survival_rate']:.1%}")
        logger.info(f"    Avg Centers: {opponent_metrics['avg_centers']:.1f}")

        # Log to WandB
        if log_to_wandb:
            wandb.log(
                {
                    f"eval/vs_{opponent_type}/win_rate": opponent_metrics["win_rate"],
                    f"eval/vs_{opponent_type}/survival_rate": opponent_metrics["survival_rate"],
                    f"eval/vs_{opponent_type}/avg_centers": opponent_metrics["avg_centers"],
                }
            )

    total_duration = time.time() - start_time

    # Commit visualizations to volume
    volume.commit()

    # Log visualizations to WandB
    if log_to_wandb and all_visualizations:
        logger.info("\n📊 Logging visualizations to WandB...")

        # Create artifact for all visualizations
        artifact = wandb.Artifact(
            f"eval-replays-{checkpoint_name}",
            type="evaluation-replays",
            description=f"Game replays for {checkpoint_path}",
        )

        for viz in all_visualizations:
            if os.path.exists(viz["path"]):
                artifact.add_file(viz["path"])

                # Also log inline HTML for first game of each opponent
                first_per_opponent = {}
                for v in all_visualizations:
                    if v["opponent"] not in first_per_opponent:
                        first_per_opponent[v["opponent"]] = v

                if viz == first_per_opponent.get(viz["opponent"]):
                    with open(viz["path"], encoding="utf-8") as f:
                        html_content = f.read()
                    wandb.log({f"eval/replay_vs_{viz['opponent']}": wandb.Html(html_content)})

        wandb.log_artifact(artifact)

        # Create summary table
        table = wandb.Table(
            columns=[
                "opponent",
                "games",
                "win_rate",
                "survival_rate",
                "avg_centers",
            ]
        )
        for r in all_results:
            table.add_data(
                r["opponent"],
                r["games"],
                r["win_rate"],
                r["survival_rate"],
                r["avg_centers"],
            )
        wandb.log({"eval/summary_table": table})

    # Log completion to Axiom
    axiom.log(
        {
            "event": "evaluation_complete",
            "checkpoint_path": checkpoint_path,
            "total_duration_s": total_duration,
            "results": [
                {
                    "opponent": r["opponent"],
                    "win_rate": r["win_rate"],
                    "survival_rate": r["survival_rate"],
                    "avg_centers": r["avg_centers"],
                }
                for r in all_results
            ],
        }
    )

    # Finish WandB
    if log_to_wandb:
        wandb.finish()

    # Flush Axiom
    await axiom.flush()

    logger.info("\n" + "=" * 60)
    logger.info("✅ EVALUATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Duration: {total_duration:.1f}s")
    logger.info(f"Visualizations saved: {len(all_visualizations)}")

    return {
        "checkpoint_path": checkpoint_path,
        "timestamp": timestamp,
        "total_duration_s": total_duration,
        "results": [
            {
                "opponent": r["opponent"],
                "games": r["games"],
                "win_rate": r["win_rate"],
                "survival_rate": r["survival_rate"],
                "avg_centers": r["avg_centers"],
            }
            for r in all_results
        ],
        "visualization_paths": [v["path"] for v in all_visualizations],
    }
