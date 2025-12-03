#!/usr/bin/env python3
"""
Generate a cache of serialized DiplomacyWrapper states for warm-starting rollouts.

This script creates ~1k mid-game states by playing random games through
varying numbers of phases. The states are serialized with cloudpickle
and stored in the Modal volume at /data/state_cache/.

Usage:
    # Local generation (for testing)
    python scripts/generate_state_cache.py --local --count 100

    # Modal deployment (production)
    modal run scripts/generate_state_cache.py --count 1000

The generated states skip the expensive warmup inference during rollouts,
saving 30-60s per rollout.
"""

import argparse
import random
import sys
from pathlib import Path

# Add src to path for local execution
sys.path.insert(0, str(Path(__file__).parent.parent))


def generate_random_state(min_phases: int = 2, max_phases: int = 10):
    """
    Generate a random game state by playing through a variable number of phases.

    Args:
        min_phases: Minimum phases to simulate
        max_phases: Maximum phases to simulate

    Returns:
        DiplomacyWrapper in a mid-game state
    """
    from src.engine.wrapper import DiplomacyWrapper

    game = DiplomacyWrapper(horizon=99)
    num_phases = random.randint(min_phases, max_phases)

    for _ in range(num_phases):
        if game.is_done():
            break

        # Get random orders for all powers
        all_orders = []
        for power_name in game.game.powers:
            valid_moves = game.get_valid_moves(power_name)
            for moves in valid_moves.values():
                if moves:
                    # Pick a random valid move
                    all_orders.append(random.choice(moves))

        game.step(all_orders)

    return game


def generate_state_cache_local(output_dir: Path, count: int = 100):
    """Generate states locally for testing."""
    import cloudpickle

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {count} game states to {output_dir}")

    for i in range(count):
        game = generate_random_state()
        output_path = output_dir / f"state_{i:04d}.pkl"

        with output_path.open("wb") as f:
            cloudpickle.dump(game, f)

        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{count} states")

    print(f"✅ Done! Generated {count} states in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate state cache for rollouts")
    parser.add_argument(
        "--count", type=int, default=100, help="Number of states to generate"
    )
    parser.add_argument(
        "--local", action="store_true", help="Run locally instead of on Modal"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./state_cache",
        help="Output directory for local mode",
    )
    parser.add_argument(
        "--min-phases", type=int, default=2, help="Minimum phases per state"
    )
    parser.add_argument(
        "--max-phases", type=int, default=10, help="Maximum phases per state"
    )

    args = parser.parse_args()

    if args.local:
        generate_state_cache_local(Path(args.output), args.count)
    else:
        # Import Modal components only when needed
        import modal

        app = modal.App("generate-state-cache")
        volume = modal.Volume.from_name("diplomacy-data", create_if_missing=True)

        cpu_image = (
            modal.Image.debian_slim()
            .pip_install("diplomacy", "cloudpickle", "pydantic", "numpy")
            .add_local_python_source("src")
        )

        @app.function(
            image=cpu_image,
            volumes={"/data": volume},
            timeout=3600,
        )
        def generate_on_modal(count: int, min_phases: int, max_phases: int):
            import cloudpickle

            output_dir = Path("/data/state_cache")
            output_dir.mkdir(parents=True, exist_ok=True)

            print(f"Generating {count} game states to {output_dir}")

            for i in range(count):
                game = generate_random_state(min_phases, max_phases)
                output_path = output_dir / f"state_{i:04d}.pkl"

                with output_path.open("wb") as f:
                    cloudpickle.dump(game, f)

                if (i + 1) % 50 == 0:
                    print(f"  Generated {i + 1}/{count} states")
                    volume.commit()  # Commit periodically

            volume.commit()
            print(f"✅ Done! Generated {count} states in {output_dir}")
            return count

        with app.run():
            result = generate_on_modal.remote(
                args.count, args.min_phases, args.max_phases
            )
            print(f"Generated {result} states on Modal")


if __name__ == "__main__":
    main()
