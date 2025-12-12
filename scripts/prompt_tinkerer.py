#!/usr/bin/env python3
"""
Prompt Tinkerer - Interactive tool to experiment with Diplomacy prompts.

Usage:
    # Show prompts for different game states
    python scripts/prompt_tinkerer.py

    # Run with a local model (requires transformers + torch)
    python scripts/prompt_tinkerer.py --model Qwen/Qwen2.5-7B-Instruct

    # Run with a LoRA adapter (uses vLLM, same as Modal)
    python scripts/prompt_tinkerer.py --model Qwen/Qwen2.5-7B-Instruct --lora ./adapter_v10

    # Test specific config
    python scripts/prompt_tinkerer.py --no-show-valid-moves --compact

    # Advance game to mid-game state
    python scripts/prompt_tinkerer.py --advance 6

    # Test specific power
    python scripts/prompt_tinkerer.py --power GERMANY
"""

import argparse
import json
import random
from typing import Any

from src.agents.llm_agent import LLMAgent, PromptConfig
from src.engine.wrapper import DiplomacyWrapper
from src.utils.parsing import extract_orders


def colorize(text: str, color: str) -> str:
    """Add ANSI color codes."""
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "bold": "\033[1m",
        "reset": "\033[0m",
    }
    return f"{colors.get(color, '')}{text}{colors['reset']}"


def print_section(title: str, content: str, color: str = "cyan") -> None:
    """Print a formatted section."""
    print()
    print(colorize(f"{'=' * 60}", color))
    print(colorize(f" {title}", f"{color}"))
    print(colorize(f"{'=' * 60}", color))
    print(content)


def count_tokens(text: str, tokenizer: Any = None) -> int:
    """Count tokens in text."""
    if tokenizer:
        return len(tokenizer.encode(text))
    # Rough estimate: ~4 chars per token
    return len(text) // 4


def advance_game(game: DiplomacyWrapper, phases: int) -> None:
    """Advance game by N phases using random first valid move."""
    for _ in range(phases):
        for power in ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]:
            valid_moves = game.get_valid_moves(power)
            if valid_moves:
                orders = [moves[0] for unit, moves in valid_moves.items() if moves]
                if orders:
                    game.game.set_orders(power, orders)
        game.game.process()
        if game.is_done():
            print(colorize(f"Game ended at {game.get_current_phase()}", "yellow"))
            break


def generate_with_vllm(
    prompt: str,
    valid_moves: dict[str, list[str]],
    model_id: str,
    temperature: float = 0.8,
    max_tokens: int = 256,
    lora_path: str | None = None,
) -> str:
    """Generate output using vLLM locally, optionally with a LoRA adapter.

    This matches how Modal's InferenceEngine works, ensuring consistent behavior.
    """
    try:
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
    except ImportError:
        print(colorize("Install vllm for local generation: pip install vllm", "red"))
        return ""

    # Import our custom logits processor
    from src.inference.logits import DiplomacyLogitsProcessor

    print(colorize(f"Loading vLLM with model {model_id}...", "yellow"))
    if lora_path:
        print(colorize(f"LoRA adapter: {lora_path}", "cyan"))

    # Initialize vLLM with LoRA support and our logits processor
    llm = LLM(
        model=model_id,
        enable_lora=bool(lora_path),
        max_lora_rank=16 if lora_path else None,
        gpu_memory_utilization=0.9,
        logits_processors=[DiplomacyLogitsProcessor],
    )

    # Pass valid_moves via extra_args - this is how DiplomacyLogitsProcessor receives them
    # start_active=True because prompt ends with <orders>, so we start constraining immediately
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        extra_args={"valid_moves_dict": valid_moves, "start_active": True},
        stop=["</orders>", "</Orders>"],
    )

    print(colorize(f"Valid moves passed to logits processor: {len(valid_moves)} units", "cyan"))

    # Generate with or without LoRA
    if lora_path:
        lora_request = LoRARequest(
            lora_name="test_adapter",
            lora_int_id=1,
            lora_path=lora_path,
        )
        outputs = llm.generate([prompt], sampling_params, lora_request=lora_request)
    else:
        outputs = llm.generate([prompt], sampling_params)

    return outputs[0].outputs[0].text if outputs else ""


def generate_with_transformers(
    prompt: str,
    valid_moves: dict[str, list[str]],
    model_id: str,
    temperature: float = 0.8,
    max_tokens: int = 256,
) -> str:
    """Generate output using transformers (no LoRA support - use vLLM for LoRA)."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print(colorize("Install transformers and torch for local generation", "red"))
        return ""

    print(colorize(f"Loading model {model_id}...", "yellow"))

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated part
    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    return generated


def mock_generation(valid_moves: dict[str, list[str]], strategy: str = "random") -> str:
    """Generate mock output for testing extraction."""
    if strategy == "random":
        # Pick random valid moves
        orders = []
        for _, moves in valid_moves.items():
            if moves:
                orders.append(random.choice(moves))
        return "\n".join(orders) + "\n</orders>"

    elif strategy == "first":
        # Pick first valid move for each unit
        orders = [moves[0] for unit, moves in valid_moves.items() if moves]
        return "\n".join(orders) + "\n</orders>"

    elif strategy == "garbage":
        # Return garbage to test extraction failure
        return "I think we should move to BUR\nMaybe support from MAR?\n</orders>"

    elif strategy == "empty":
        return "</orders>"

    elif strategy == "partial":
        # Return some valid, some invalid
        orders = []
        for i, (unit, moves) in enumerate(valid_moves.items()):
            if moves:
                if i % 2 == 0:
                    orders.append(moves[0])
                else:
                    orders.append(f"Invalid order for {unit}")
        return "\n".join(orders) + "\n</orders>"

    return "</orders>"


def interactive_mode(
    agent: LLMAgent, game: DiplomacyWrapper, power: str, tokenizer: Any = None
) -> None:
    """Interactive REPL for prompt experimentation."""
    print(colorize("\n🎮 Interactive Mode", "bold"))
    print("Commands:")
    print("  prompt      - Show current prompt")
    print("  mock [strategy] - Mock generation (random/first/garbage/empty/partial)")
    print("  advance [n] - Advance game by n phases")
    print("  power NAME  - Switch power")
    print("  phase       - Show current phase")
    print("  valid       - Show valid moves")
    print("  config      - Show current config")
    print("  quit        - Exit")
    print()

    while True:
        try:
            cmd = input(colorize(">>> ", "green")).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not cmd:
            continue

        parts = cmd.split()
        command = parts[0]

        if command == "quit" or command == "q":
            break

        elif command == "prompt" or command == "p":
            prompt, valid_moves = agent.build_prompt(game, power)
            tokens = count_tokens(prompt, tokenizer)
            print_section(
                f"PROMPT for {power} ({tokens} tokens, {len(prompt)} chars)", prompt, "cyan"
            )

        elif command == "mock" or command == "m":
            strategy = parts[1] if len(parts) > 1 else "random"
            prompt, valid_moves = agent.build_prompt(game, power)

            print_section("PROMPT", prompt, "cyan")

            mock_output = mock_generation(valid_moves, strategy)
            print_section(f"MOCK OUTPUT (strategy={strategy})", mock_output, "yellow")

            # Full output as model would generate it
            full_output = mock_output
            orders = extract_orders(full_output)

            print_section("EXTRACTED ORDERS", "\n".join(orders) if orders else "(none)", "green")

            # Validate against valid moves
            all_valid = set()
            for moves in valid_moves.values():
                all_valid.update(moves)

            valid_count = sum(1 for o in orders if o in all_valid)
            print(f"\nValid: {valid_count}/{len(orders)} orders")

        elif command == "advance" or command == "a":
            n = int(parts[1]) if len(parts) > 1 else 2
            advance_game(game, n)
            print(colorize(f"Advanced to {game.get_current_phase()}", "green"))

        elif command == "power":
            if len(parts) > 1:
                power = parts[1].upper()
                print(colorize(f"Switched to {power}", "green"))
            else:
                print(f"Current power: {power}")

        elif command == "phase":
            print(f"Phase: {game.get_current_phase()}")
            print(f"Done: {game.is_done()}")

        elif command == "valid" or command == "v":
            valid_moves = game.get_valid_moves(power)
            output = json.dumps(valid_moves, indent=2)
            print_section(f"VALID MOVES for {power}", output, "magenta")

        elif command == "config" or command == "c":
            print(f"compact_mode: {agent.config.compact_mode}")
            print(f"prefix_cache_optimized: {agent.config.prefix_cache_optimized}")
            print(f"show_valid_moves: {agent.config.show_valid_moves}")

        else:
            print(colorize(f"Unknown command: {command}", "red"))


def main():
    parser = argparse.ArgumentParser(
        description="Prompt Tinkerer - Experiment with Diplomacy prompts"
    )

    # Config options
    parser.add_argument(
        "--compact", action="store_true", default=True, help="Use compact mode (default: True)"
    )
    parser.add_argument("--no-compact", action="store_true", help="Disable compact mode")
    parser.add_argument(
        "--prefix-cache",
        action="store_true",
        default=True,
        help="Optimize for prefix caching (default: True)",
    )
    parser.add_argument(
        "--no-prefix-cache", action="store_true", help="Disable prefix cache optimization"
    )
    parser.add_argument(
        "--show-valid-moves", default=False, action="store_true", help="Show valid moves in prompt"
    )
    parser.add_argument(
        "--no-show-valid-moves",
        default=True,
        action="store_true",
        help="Hide valid moves (use logits processor)",
    )

    # Game state
    parser.add_argument("--power", default="FRANCE", help="Power to generate prompt for")
    parser.add_argument("--advance", type=int, default=4, help="Advance game by N phases")

    # Model options
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-7B-Instruct",
        type=str,
        help="Model ID for local generation",
    )
    parser.add_argument(
        "--lora",
        default="./adapter_v100",
        type=str,
        help="Path to local LoRA adapter (e.g., ./adapter_v10)",
    )
    parser.add_argument("--temperature", type=float, default=0.8, help="Generation temperature")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens to generate")

    # Output options
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--all-powers", action="store_true", help="Show prompts for all powers")
    parser.add_argument("--mock", type=str, help="Mock generation strategy (random/first/garbage)")

    args = parser.parse_args()

    # Build config
    config = PromptConfig(
        compact_mode=not args.no_compact,
        prefix_cache_optimized=not args.no_prefix_cache,
        show_valid_moves=args.show_valid_moves and not args.no_show_valid_moves,
    )

    agent = LLMAgent(config)
    game = DiplomacyWrapper()

    # Try to load tokenizer for accurate token counts
    tokenizer = None
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    except Exception:
        pass

    # Advance game if requested
    if args.advance > 0:
        advance_game(game, args.advance)
        print(colorize(f"Game at: {game.get_current_phase()}", "green"))

    # Interactive mode
    if args.interactive:
        interactive_mode(agent, game, args.power.upper(), tokenizer)
        return

    # Show prompts
    powers = (
        ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]
        if args.all_powers
        else [args.power.upper()]
    )

    for power in powers:
        prompt, valid_moves = agent.build_prompt(game, power)
        tokens = count_tokens(prompt, tokenizer)

        print_section(f"PROMPT for {power} ({tokens} tokens, {len(prompt)} chars)", prompt, "cyan")

        # Show valid moves summary
        total_moves = sum(len(m) for m in valid_moves.values())
        units = list(valid_moves.keys())
        print(colorize(f"\nUnits: {len(units)} | Total valid moves: {total_moves}", "white"))

        # Mock or real generation
        if args.mock:
            mock_output = mock_generation(valid_moves, args.mock)
            print_section(f"MOCK OUTPUT (strategy={args.mock})", mock_output, "yellow")

            orders = extract_orders(mock_output)
            print_section("EXTRACTED ORDERS", "\n".join(orders) if orders else "(none)", "green")

        elif args.model:
            # Use vLLM if LoRA is specified (matches Modal's inference), else transformers
            if args.lora:
                output = generate_with_vllm(
                    prompt,
                    valid_moves,
                    args.model,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    lora_path=args.lora,
                )
                lora_info = f" + LoRA {args.lora}"
            else:
                output = generate_with_transformers(
                    prompt,
                    valid_moves,
                    args.model,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
                lora_info = ""
            print_section(f"MODEL OUTPUT ({args.model}{lora_info})", output, "yellow")

            orders = extract_orders(output)
            print_section("EXTRACTED ORDERS", "\n".join(orders) if orders else "(none)", "green")
            print_section(
                "VALID MOVES",
                "\n".join(f"{unit}: {', '.join(moves)}" for unit, moves in valid_moves.items()),
                "magenta",
            )
            print_section("FULL OUTPUT", output, "yellow")
            # Validate
            all_valid = set()
            for moves in valid_moves.values():
                all_valid.update(moves)

            valid_count = sum(1 for o in orders if o in all_valid)
            invalid = [o for o in orders if o not in all_valid]

            print(f"\nValid: {valid_count}/{len(orders)} orders")
            if invalid:
                print(colorize(f"Invalid orders: {invalid}", "red"))

    print()


if __name__ == "__main__":
    main()
