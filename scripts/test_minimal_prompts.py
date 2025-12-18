#!/usr/bin/env python3
"""Test script to compare prompt sizes with different configurations."""

from collections import defaultdict

import tiktoken

from src.agents.baselines import ChaosBot
from src.agents.llm_agent import LLMAgent, PromptConfig
from src.engine.wrapper import DiplomacyWrapper


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~1 token per 4 characters."""
    return len(tiktoken.encoding_for_model("gpt-4o").encode(text))


def test_prompt_modes():
    """Compare different prompt configurations."""
    # Initialize game
    game = DiplomacyWrapper(horizon=5)
    bot = ChaosBot()
    while not game.is_done():
        all_orders = []
        for power in game.game.powers:
            orders = bot.get_orders(game, power)
            all_orders.extend(orders)
        game.step(all_orders)
        if game.is_done():
            break

    print("=" * 80)
    print("PROMPT MODE COMPARISON")
    print("=" * 80)

    configs = [
        (
            "Full Prompts (Baseline)",
            PromptConfig(
                show_valid_moves=True,
                show_board_context=False,
                show_map_windows=False,
                show_action_counts=False,
                compact_mode=True,
                prefix_cache_optimized=True,
            ),
        ),
        (
            "Minimal (No Hints)",
            PromptConfig(
                show_valid_moves=False,
                show_board_context=True,
                show_map_windows=True,
                show_action_counts=False,
                compact_mode=True,
                prefix_cache_optimized=True,
            ),
        ),
        (
            "Minimal + Action Counts",
            PromptConfig(
                show_valid_moves=False,
                show_board_context=True,
                show_map_windows=True,
                show_action_counts=True,
                compact_mode=True,
                prefix_cache_optimized=True,
            ),
        ),
    ]
    results = defaultdict(list)
    for name, config in configs:
        agent = LLMAgent(config)
        for power in game.game.powers:
            prompt, valid_moves = agent.build_prompt(game, power)
            total_valid_moves = []
            for valid_move in valid_moves.values():
                total_valid_moves.extend(valid_move)
            results[name].append((power, estimate_tokens(prompt), len(total_valid_moves)))
    end_year = game.get_year()
    print(f"End year: {end_year}")
    for name, r in results.items():
        print(f"{name}:")
        average_tokens = sum(tokens for _, tokens, _ in r) / len(r)
        average_valid_moves = sum(valid_moves for _, _, valid_moves in r) / len(results)
        print(f"  Average tokens: {average_tokens}")
        print(f"  Average valid moves: {average_valid_moves}")
        print(f"  Average tokens per valid move: {average_tokens / average_valid_moves}")
        print()


if __name__ == "__main__":
    test_prompt_modes()
