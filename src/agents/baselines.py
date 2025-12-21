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


class DefensiveBot(DiplomacyAgent):
    """
    Prioritizes holding centers and supporting adjacent units.
    Represents 'cautious' play style - harder to beat than random.
    """

    def get_orders(self, game: DiplomacyWrapper, power_name: str) -> list[str]:
        valid_moves = game.get_valid_moves(power_name)
        orders = []

        for _unit, move_options in valid_moves.items():
            if not move_options:
                continue

            # Prefer: Support > Hold > Move
            supports = [m for m in move_options if " S " in m]
            holds = [m for m in move_options if " H" in m]

            if supports:
                orders.append(random.choice(supports))
            elif holds:
                orders.append(holds[0])
            else:
                orders.append(random.choice(move_options))

        return orders

    def get_press(self, game: DiplomacyWrapper, power_name: str) -> list[dict[str, Any]]:
        return []


class TerritorialBot(DiplomacyAgent):
    """
    Prioritizes moving toward neutral supply centers.
    Represents 'greedy expansion' strategy.
    """

    # Neutral supply centers at game start
    NEUTRAL_CENTERS = {
        "BEL",
        "HOL",
        "DEN",
        "SWE",
        "NWY",
        "SPA",
        "POR",
        "TUN",
        "GRE",
        "SER",
        "RUM",
        "BUL",
    }

    def get_orders(self, game: DiplomacyWrapper, power_name: str) -> list[str]:
        valid_moves = game.get_valid_moves(power_name)
        orders = []

        for _unit, move_options in valid_moves.items():
            if not move_options:
                continue

            # Prefer moves toward neutral centers
            toward_neutral = [
                m for m in move_options if any(nc in m for nc in self.NEUTRAL_CENTERS)
            ]

            if toward_neutral:
                orders.append(random.choice(toward_neutral))
            else:
                # Fall back to aggressive moves
                moves = [m for m in move_options if " - " in m]
                if moves:
                    orders.append(random.choice(moves))
                else:
                    orders.append(random.choice(move_options))

        return orders

    def get_press(self, game: DiplomacyWrapper, power_name: str) -> list[dict[str, Any]]:
        return []


class CoordinatedBot(DiplomacyAgent):
    """
    Attempts to coordinate supports between adjacent units.
    Represents 'team play' strategy - units work together.
    """

    def get_orders(self, game: DiplomacyWrapper, power_name: str) -> list[str]:
        valid_moves = game.get_valid_moves(power_name)
        units = list(valid_moves.keys())

        if not units:
            return []

        # Pick one unit to move (attacker), have others support it
        # Prefer units with move options (not just holds)
        attackers = [u for u in units if any(" - " in m for m in valid_moves.get(u, []))]
        attacker = random.choice(attackers) if attackers else random.choice(units)

        attacker_moves = [m for m in valid_moves.get(attacker, []) if " - " in m]

        if not attacker_moves:
            # No moves available - everyone does their own thing
            return [random.choice(valid_moves[u]) for u in units if valid_moves.get(u)]

        attack_order = random.choice(attacker_moves)
        orders = [attack_order]

        # Extract destination from attack order (e.g., "A PAR - BUR" -> "BUR")
        # Format: "UNIT LOC - DEST" or "UNIT LOC - DEST VIA"
        attack_dest = attack_order.split(" - ")[-1].split()[0] if " - " in attack_order else None

        # Have other units support if possible
        for unit in units:
            if unit == attacker:
                continue

            unit_moves = valid_moves.get(unit, [])
            if not unit_moves:
                continue

            # Look for supports of the attacker's move
            if attack_dest:
                # Support format: "A MUN S A PAR - BUR" or "A MUN S A BUR"
                supports = [m for m in unit_moves if " S " in m and attack_dest in m]
                if supports:
                    orders.append(random.choice(supports))
                    continue

            # Fallback: any support, or random move
            any_supports = [m for m in unit_moves if " S " in m]
            if any_supports:
                orders.append(random.choice(any_supports))
            else:
                orders.append(random.choice(unit_moves))

        return orders

    def get_press(self, game: DiplomacyWrapper, power_name: str) -> list[dict[str, Any]]:
        return []
