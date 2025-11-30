from src.agents.baselines import ChaosBot, RandomBot
from src.engine.wrapper import DiplomacyWrapper
from src.utils.vis import GameVisualizer


def run_simulation(agent_class, run_name: str):
    print(f"🤖 Starting Simulation with {run_name}...")

    game = DiplomacyWrapper()
    vis = GameVisualizer()

    # We can assign different agents to different powers if we want!
    # For now, everyone uses the same bot class.
    agents = {power: agent_class() for power in game.game.powers}

    max_years = 3
    start_year = game.get_year()

    vis.capture_turn(game.game, f"Game Start ({run_name})")

    while not game.is_done():
        current_year = game.get_year()
        if current_year >= start_year + max_years:
            break

        print(f"=== {game.get_current_phase()} ===")

        all_orders = []
        logs = []

        # POLL AGENTS
        for power, agent in agents.items():
            try:
                orders = agent.get_orders(game, power)
                all_orders.extend(orders)

                # Log a few moves for the report
                if orders:
                    logs.append(f"{power}: {orders[0]} ...")
            except Exception as e:
                print(f"❌ Error getting orders for {power}: {e}")

        # EXECUTE
        game.step(all_orders)
        vis.capture_turn(game.game, "\n".join(logs))

    output_file = f"sim_{run_name.lower()}.html"
    vis.save_html(output_file)
    print(f"✅ Saved replay to {output_file}")


if __name__ == "__main__":
    # Run two comparisons
    run_simulation(RandomBot, "RandomBot")
    run_simulation(ChaosBot, "ChaosBot")
