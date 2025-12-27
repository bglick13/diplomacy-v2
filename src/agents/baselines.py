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


class DumbBot(DiplomacyAgent):
    """
    Port of David Norman's DumbBot algorithm.

    Two-stage process:
    1. Calculate province values based on supply center ownership and proximity
    2. Generate orders by selecting highest-value destinations with coordination

    Reference: https://github.com/diplomacy/daide-client/tree/master/bots/dumbbot
    """

    # Weight parameters (from original C++ source - chosen by intuition, not optimized)
    PROXIMITY_WEIGHTS = [100, 1000, 30, 10, 6, 5, 4, 3, 2, 1]
    STRENGTH_WEIGHT = 1000
    COMPETITION_WEIGHT = 1000

    def get_orders(self, game: DiplomacyWrapper, power_name: str) -> list[str]:
        """Generate orders using DumbBot heuristics."""
        valid_moves = game.get_valid_moves(power_name)
        if not valid_moves:
            return []

        phase_type = game.get_phase_type()

        # For adjustment phases, use simpler logic
        if phase_type == "A":
            return self._handle_adjustment(game, power_name, valid_moves)

        # Stage 1: Calculate province values
        province_values = self._calculate_province_values(game, power_name)

        # Stage 2: Generate movement orders
        return self._generate_movement_orders(game, power_name, valid_moves, province_values)

    def _calculate_province_values(
        self, game: DiplomacyWrapper, power_name: str
    ) -> dict[str, float]:
        """
        Calculate value for each province based on:
        - Defense value: size of largest adjacent enemy power (for our SCs)
        - Attack value: size of owning power (for enemy SCs)
        - Proximity blur: propagate values to adjacent provinces
        - Strength: bonus for our adjacent units
        - Competition: penalty for enemy adjacent units
        """
        values: dict[str, float] = {}
        game_map = game.game.map

        # Get all provinces
        all_locs = set()
        for power in game.game.powers.values():
            all_locs.update(power.centers)
            for unit in power.units:
                parts = unit.split()
                if len(parts) >= 2:
                    all_locs.add(parts[1].split("/")[0])
        # Add SC list
        all_locs.update(game_map.scs)

        # Calculate power sizes (unit counts)
        power_sizes = {pn: len(p.units) for pn, p in game.game.powers.items()}

        # Our centers and enemy centers
        my_centers = set(game.game.powers[power_name].centers)
        enemy_centers: dict[str, str] = {}  # loc -> owner
        for pn, p in game.game.powers.items():
            if pn != power_name:
                for c in p.centers:
                    enemy_centers[c] = pn

        # Unit locations by power
        unit_locs_by_power: dict[str, set[str]] = {pn: set() for pn in game.game.powers}
        for pn, p in game.game.powers.items():
            for unit in p.units:
                parts = unit.split()
                if len(parts) >= 2:
                    unit_locs_by_power[pn].add(parts[1].split("/")[0])

        my_unit_locs = unit_locs_by_power[power_name]

        # Stage 1a: Base values
        for loc in all_locs:
            base_value = 0.0

            if loc in my_centers:
                # Defense value: size of largest adjacent enemy
                try:
                    neighbors = game_map.abut_list(loc)
                except Exception:
                    neighbors = []

                max_adjacent_enemy = 0
                for neighbor in neighbors:
                    for pn, locs in unit_locs_by_power.items():
                        if pn != power_name and neighbor in locs:
                            max_adjacent_enemy = max(max_adjacent_enemy, power_sizes.get(pn, 0))
                base_value = max_adjacent_enemy * self.PROXIMITY_WEIGHTS[0]

            elif loc in enemy_centers:
                # Attack value: size of owning power
                owner = enemy_centers[loc]
                base_value = power_sizes.get(owner, 0) * self.PROXIMITY_WEIGHTS[0]

            elif loc in game_map.scs:
                # Neutral SC - moderate value
                base_value = 5 * self.PROXIMITY_WEIGHTS[0]

            values[loc] = base_value

        # Stage 1b: Proximity blur (propagate values to neighbors)
        for depth in range(1, len(self.PROXIMITY_WEIGHTS)):
            new_values = dict(values)
            for loc in all_locs:
                try:
                    neighbors = game_map.abut_list(loc)
                except Exception:
                    continue

                neighbor_sum = sum(values.get(n, 0) for n in neighbors)
                blurred = neighbor_sum / max(1, len(neighbors))
                new_values[loc] = values.get(loc, 0) + blurred * self.PROXIMITY_WEIGHTS[depth] / 100
            values = new_values

        # Stage 1c: Strength and competition
        for loc in all_locs:
            try:
                neighbors = game_map.abut_list(loc)
            except Exception:
                continue

            # Strength: count our adjacent units
            strength = sum(1 for n in neighbors if n in my_unit_locs)
            values[loc] = values.get(loc, 0) + strength * self.STRENGTH_WEIGHT

            # Competition: count max adjacent enemy units from single power
            max_enemy_adjacent = 0
            for pn, locs in unit_locs_by_power.items():
                if pn != power_name:
                    enemy_adjacent = sum(1 for n in neighbors if n in locs)
                    max_enemy_adjacent = max(max_enemy_adjacent, enemy_adjacent)
            values[loc] = values.get(loc, 0) - max_enemy_adjacent * self.COMPETITION_WEIGHT

        return values

    def _generate_movement_orders(
        self,
        game: DiplomacyWrapper,
        power_name: str,
        valid_moves: dict[str, list[str]],
        province_values: dict[str, float],
    ) -> list[str]:
        """Generate movement orders by selecting highest-value destinations."""
        orders = []
        claimed_destinations: set[str] = set()

        # Process units in random order for non-determinism
        units = list(valid_moves.keys())
        random.shuffle(units)

        for unit in units:
            moves = valid_moves.get(unit, [])
            if not moves:
                continue

            # Get current location
            unit_parts = unit.split()
            current_loc = unit_parts[1].split("/")[0] if len(unit_parts) >= 2 else ""

            # Score each move by destination value
            move_scores: list[tuple[str, float, str]] = []  # (move, score, dest)

            for move in moves:
                dest = self._extract_destination(move, current_loc)
                if dest in claimed_destinations:
                    # Already claimed - check if we should support instead
                    if " S " in move:
                        # Support move - score by what we're supporting
                        move_scores.append((move, province_values.get(dest, 0) * 0.9, dest))
                    continue

                score = province_values.get(dest, 0)

                # Slight preference for holds if current position is valuable
                if " H" in move:
                    score = province_values.get(current_loc, 0) * 1.1

                move_scores.append((move, score, dest))

            if not move_scores:
                # All destinations claimed - find a support
                support_moves = [m for m in moves if " S " in m]
                if support_moves:
                    orders.append(random.choice(support_moves))
                elif moves:
                    orders.append(random.choice(moves))
                continue

            # Select best move (with some randomness for ties)
            move_scores.sort(key=lambda x: x[1], reverse=True)
            best_move, _best_score, best_dest = move_scores[0]

            # Claim destination and add order
            if best_dest != current_loc:
                claimed_destinations.add(best_dest)
            orders.append(best_move)

        return orders

    def _extract_destination(self, move: str, current_loc: str) -> str:
        """Extract destination province from an order string."""
        # Check convoy first since it also contains " - "
        if " C " in move:
            # Convoy: "F NTH C A LON - NWY" -> stay in place
            return current_loc
        elif " S " in move:
            # Support: destination is the supported unit's target
            # "A MUN S A PAR - BUR" -> "BUR" or "A MUN S A BUR" -> "BUR"
            parts = move.split(" S ")
            if len(parts) > 1:
                supported = parts[1]
                if " - " in supported:
                    return supported.split(" - ")[-1].split()[0].split("/")[0]
                else:
                    # Support hold: "A MUN S A BUR" -> "BUR"
                    s_parts = supported.split()
                    if len(s_parts) >= 2:
                        return s_parts[1].split("/")[0]
            return current_loc
        elif " - " in move:
            # Movement order: "A PAR - BUR" -> "BUR"
            dest = move.split(" - ")[-1].split()[0]
            return dest.split("/")[0]
        elif " H" in move or "HOLD" in move.upper():
            # Hold: destination is current location
            return current_loc
        else:
            return current_loc

    def _handle_adjustment(
        self, game: DiplomacyWrapper, power_name: str, valid_moves: dict[str, list[str]]
    ) -> list[str]:
        """Handle build/disband phase."""
        orders = []
        delta = game.get_adjustment_delta(power_name)

        if delta > 0:
            # Need to build - prefer army builds in SCs closer to front
            build_moves = []
            for _loc, moves in valid_moves.items():
                for move in moves:
                    if " B" in move:
                        build_moves.append(move)

            # Random selection for builds (could be smarter)
            random.shuffle(build_moves)
            for move in build_moves[:delta]:
                orders.append(move)

        elif delta < 0:
            # Need to disband - disband furthest from front (simplified)
            disband_moves = []
            for _loc, moves in valid_moves.items():
                for move in moves:
                    if " D" in move:
                        disband_moves.append(move)

            random.shuffle(disband_moves)
            for move in disband_moves[: abs(delta)]:
                orders.append(move)

        return orders

    def get_press(self, game: DiplomacyWrapper, power_name: str) -> list[dict[str, Any]]:
        return []
