import random
from typing import Any

from src.agents.base import DiplomacyAgent
from src.engine.wrapper import DiplomacyWrapper


class RandomBot(DiplomacyAgent):
    """
    The 'Standard' baseline. Picks a valid move uniformly at random.
    Usually results in lots of Holds due to Support dominance.
    """

    def get_orders(self, game: DiplomacyWrapper, power_name: str) -> list[str]:
        valid_moves = game.get_valid_moves(power_name)
        orders = []

        for _unit, move_options in valid_moves.items():
            if move_options:
                orders.append(random.choice(move_options))

        return orders

    def get_press(self, game: DiplomacyWrapper, power_name: str) -> list[dict[str, Any]]:
        return []


class ChaosBot(DiplomacyAgent):
    """
    The 'Aggressive' baseline.
    Prioritizes MOVE orders over HOLD/SUPPORT orders.
    Useful for visual debugging to ensure the engine is processing movement.
    """

    def get_orders(self, game: DiplomacyWrapper, power_name: str) -> list[str]:
        valid_moves = game.get_valid_moves(power_name)
        orders = []

        for _unit, move_options in valid_moves.items():
            if not move_options:
                continue

            # Filter for moves that contain " - " (Movement indicator)
            # Standard notation: "A PAR - BUR" vs "A PAR S ..."
            aggressive_moves = [m for m in move_options if " - " in m]

            if aggressive_moves:
                orders.append(random.choice(aggressive_moves))
            else:
                # Fallback if no moves available (e.g. surrounded)
                orders.append(random.choice(move_options))

        return orders

    def get_press(self, game: DiplomacyWrapper, power_name: str) -> list[dict[str, Any]]:
        return []
