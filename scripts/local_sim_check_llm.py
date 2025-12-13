#!/usr/bin/env python3
"""
Local simulation check for LLM agent.

Runs a single game simulation with the LLM agent and generates
an HTML visualization. Useful for debugging prompts and outputs.

Usage:
    # Run with default settings
    python scripts/local_sim_check_llm.py

    # Custom horizon
    python scripts/local_sim_check_llm.py --rollout-horizon-years 4

    # With a trained checkpoint
    python scripts/local_sim_check_llm.py --checkpoint "benchmark-20251205/adapter_v10"
"""

from __future__ import annotations

import argparse

import modal

from src.agents.llm_agent import LLMAgent, PromptConfig
from src.engine.wrapper import DiplomacyWrapper
from src.utils.config import ExperimentConfig, add_config_args, config_from_args
from src.utils.parsing import extract_orders
from src.utils.vis import GameVisualizer


def run_simulation(
    cfg: ExperimentConfig, checkpoint: str | None = None, run_name: str = "LLMAgent"
):
    """Run a single game simulation with the LLM agent."""
    print(f"🤖 Starting Simulation with {run_name}...")

    # Warmup inference engine
    print("🔥 Warming up InferenceEngine...")
    InferenceEngine = modal.Cls.from_name("diplomacy-grpo", "InferenceEngine")
    engine = InferenceEngine(model_id=cfg.base_model_id)
    _ = engine.generate.remote(
        prompts=["<orders>"],
        valid_moves=[{"A PAR": ["A PAR - BUR"]}],
    )
    print("✅ InferenceEngine ready!")

    # Initialize game
    game = DiplomacyWrapper(horizon=cfg.rollout_horizon_years)
    vis = GameVisualizer()

    # Initialize agent
    prompt_config = PromptConfig(
        compact_mode=cfg.compact_prompts,
        show_map_windows=cfg.show_map_windows,
    )
    agent = LLMAgent(config=prompt_config)

    vis.capture_turn(game.game, f"Game Start ({run_name})")

    # Game loop
    while not game.is_done():
        phase = game.get_current_phase()
        print(f"=== {phase} ===")

        all_orders = []
        logs = []

        # Get inputs for all powers
        inputs = game.get_all_inputs(agent=agent)

        # Run inference
        raw_responses = engine.generate.remote(
            prompts=inputs["prompts"],
            valid_moves=inputs["valid_moves"],
            lora_name=checkpoint,
        )

        # Parse responses (handle rich response format with token data)
        for idx, (response_data, power) in enumerate(
            zip(raw_responses, inputs["power_names"], strict=True)
        ):
            response_text = response_data["text"]
            orders = extract_orders(response_text)
            expected_count = len(inputs["valid_moves"][idx])

            all_orders.extend(orders)
            logs.append(f"{power}:\n\t{chr(10).join(orders)}")
            if len(orders) != expected_count:
                logs.append(f"⚠️  Expected {expected_count} orders, got {len(orders)}")

        # Execute orders
        game.step(all_orders)
        vis.capture_turn(game.game, "\n".join(logs))

    # Save visualization
    output_file = f"sim_{run_name.lower().replace(' ', '_')}.html"
    vis.save_html(output_file)
    print(f"✅ Saved replay to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run a local simulation check with the LLM agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Add relevant ExperimentConfig args
    add_config_args(
        parser,
        ExperimentConfig,
        exclude={
            "run_name",
            "total_steps",
            "num_groups_per_step",
            "learning_rate",
            "max_grad_norm",
            "chunk_size",
            "profiling_mode",
            "profile_run_name",
            "profiling_trace_steps",
            "experiment_tag",
            "wandb_project",
            "rollout_visualize_chance",
            "rollout_no_warmup_chance",
            "max_new_tokens",
            "temperature",
        },
    )

    # Script-specific args
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path relative to /data/models (optional)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="LLMAgent",
        help="Name for this simulation run",
    )

    args = parser.parse_args()

    # Build config from args
    cfg = config_from_args(args, ExperimentConfig)  # type: ignore[type-var]
    assert isinstance(cfg, ExperimentConfig)

    # Run simulation
    run_simulation(cfg, checkpoint=args.checkpoint, run_name=args.name)


if __name__ == "__main__":
    main()
