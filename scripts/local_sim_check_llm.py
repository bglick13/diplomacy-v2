import modal

from src.agents.llm_agent import LLMAgent
from src.engine.wrapper import DiplomacyWrapper
from src.utils.config import ExperimentConfig
from src.utils.parsing import extract_orders
from src.utils.vis import GameVisualizer


def run_simulation(cfg: ExperimentConfig, run_name: str = "LLMAgent"):
    print(f"🤖 Starting Simulation with {run_name}...")

    print("🔥 Warming up InferenceEngine...")
    InferenceEngine = modal.Cls.from_name("diplomacy-grpo", "InferenceEngine")
    engine = InferenceEngine()
    # Make a minimal call to trigger container startup and wait for @modal.enter() to complete
    _ = engine.generate.remote(prompts=["<orders>"], valid_moves=[{"A PAR": ["A PAR - BUR"]}])
    print("✅ InferenceEngine ready!")

    game = DiplomacyWrapper(horizon=cfg.rollout_horizon_years)
    vis = GameVisualizer()

    # We can assign different agents to different powers if we want!
    # For now, everyone uses the same bot class.
    agent = LLMAgent()

    vis.capture_turn(game.game, f"Game Start ({run_name})")

    while not game.is_done():
        print(f"=== {game.get_current_phase()} ===")

        all_orders = []
        logs = []

        inputs = game.get_all_inputs(agent=agent)
        raw_responses = engine.generate.remote(
            prompts=inputs["prompts"],
            valid_moves=inputs["valid_moves"],
            lora_name=None,
        )
        for idx, (response, power) in enumerate(
            zip(raw_responses, inputs["power_names"], strict=True)
        ):
            orders = extract_orders(response)
            expected_count = len(inputs["valid_moves"][idx])

            all_orders.extend(orders)
            logs.append(f"{power}:\n\t{'\n'.join(orders)}")
            if len(orders) != expected_count:
                logs.append(f"Expected {expected_count} orders, got {len(orders)}")

        # EXECUTE
        game.step(all_orders)
        vis.capture_turn(game.game, "\n".join(logs))

    output_file = f"sim_{run_name.lower()}.html"
    vis.save_html(output_file)
    print(f"✅ Saved replay to {output_file}")


if __name__ == "__main__":
    # Run two comparisons
    cfg = ExperimentConfig(rollout_horizon_years=4, samples_per_group=8)
    run_simulation(cfg)
