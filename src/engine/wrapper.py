import json
from typing import Any, Dict, List

import diplomacy
from diplomacy.utils.export import to_saved_game_format


class DiplomacyWrapper:
    """
    Abstractions over the 'diplomacy' pip package to ensure consistent
    tokenization and state representation for LLMs.
    """

    def __init__(self, game_id: str = None, horizon: int = 2):
        self.game = diplomacy.Game(game_id=game_id)
        self.max_years = horizon
        self.start_year = self.get_year()

    def is_done(self) -> bool:
        if self.game.is_game_done:
            return True
        # Check Horizon
        if self.get_year() >= self.start_year + self.max_years:
            return True
        return False

    def get_current_phase(self) -> str:
        # Just return the standard phase string (e.g. "SPRING 1901 MOVEMENT")
        # The previous version prepended phase_type ('M'), which confused parsing.
        return self.game.phase

    def get_year(self) -> int:
        """Robustly extract year from the game state."""
        # The diplomacy package usually formats phase as "SEASON YEAR TYPE"
        try:
            parts = self.game.phase.split()
            for part in parts:
                if part.isdigit():
                    return int(part)
            return 1900
        except:
            return 1900

    def get_valid_moves(self, power_name: str) -> Dict[str, List[str]]:
        # ... (Same as previous correct version) ...
        # Ensure we have a valid power name
        if power_name not in self.game.powers:
            return {}

        possible_orders = self.game.get_all_possible_orders()
        current_units = self.game.powers[power_name].units

        normalized_moves = {}

        for unit_str in current_units:
            parts = unit_str.split()
            if len(parts) < 2:
                continue

            loc_key = parts[1]
            if loc_key in possible_orders:
                orders = possible_orders[loc_key]
                clean_orders = [str(order) for order in orders]
                normalized_moves[unit_str] = clean_orders

        return normalized_moves

    def get_all_inputs(self) -> Dict[str, List]:
        """
        Prepares the batch of inputs for all 7 powers to send to the GPU.
        Returns: {
            "prompts": [str, str, ...],
            "valid_moves": [dict, dict, ...],
            "power_names": [str, str, ...]
        }
        """
        prompts = []
        valid_moves_list = []
        power_names = []

        for power in self.game.powers:
            # 1. Get Valid Moves (for Masking)
            vm = self.get_valid_moves(power)

            # 2. Build Prompt (The Context)
            # For Level 2, we just dump the JSON. Level 3/4 can make this prettier.
            state = {
                "meta": {"role": power, "phase": self.get_current_phase()},
                "valid_moves": vm,  # Showing valid moves in prompt helps the model too
                # We omit full board state for brevity in this snippet,
                # but normally you'd add self.get_state_json() filtered for viewpoint
            }
            prompt_str = f"You are playing Diplomacy as {power}.\nContext: {json.dumps(state)}]\nOutput XML with your moves. The format of your output must be strictly as follows:<think>...</think><orders>...</orders>"

            prompts.append(prompt_str)
            valid_moves_list.append(vm)
            power_names.append(power)

        return {
            "prompts": prompts,
            "valid_moves": valid_moves_list,
            "power_names": power_names,
        }

    def step(self, orders: List[str]):
        """
        Executes orders for the current phase.
        orders: A flat list of strings like ['A PAR - BUR', 'F BRE H']
        """
        # 1. Map all current units to their Power for fast lookup
        # e.g. "A PAR" -> "FRANCE"
        unit_owner_map = {}
        for power_name, power_obj in self.game.powers.items():
            for unit in power_obj.units:
                unit_owner_map[unit] = power_name

        # 2. Group orders by Power
        # e.g. "FRANCE": ["A PAR - BUR", ...]
        orders_by_power = {p: [] for p in self.game.powers}

        for order in orders:
            # Extract the unit string from the order
            # Standard format: "A PAR - BUR" -> Unit is "A PAR"
            # Coast format: "F SPA/NC - MAO" -> Unit is "F SPA/NC"
            parts = order.split()
            if len(parts) < 2:
                continue

            # Construct candidate unit string
            unit_str = f"{parts[0]} {parts[1]}"

            # Check ownership
            owner = unit_owner_map.get(unit_str)

            if owner:
                orders_by_power[owner].append(order)
            else:
                # Fallback logging (useful for debugging Coast inconsistencies)
                # print(f"⚠️ skipped order: {order} (Owner not found for {unit_str})")
                pass

        # 3. Submit orders to the engine
        for power, power_orders in orders_by_power.items():
            if power_orders:
                try:
                    # The correct API method for diplomacy==1.1.0
                    self.game.set_orders(power, power_orders)
                except Exception as e:
                    print(f"❌ Engine rejected orders for {power}: {e}")

        self.game.process()

    def get_state_json(self) -> Dict[str, Any]:
        return to_saved_game_format(self.game)

    def get_unit(self, power_name: str, unit_str: str) -> str | None:
        units: list[str] = self.game.get_units(power_name)
        print(
            f"\nDEBUG get_unit: power={power_name}, location={unit_str}, units={units}"
        )
        try:
            # Find the unit string in the list (e.g., "A PAR" in ["A PAR", "F BRE"])
            # We need to match the full unit string format
            for unit in units:
                # unit is like "A PAR" or "F BRE", we want to check if location matches
                unit_parts = unit.split()
                if len(unit_parts) >= 2 and unit_parts[1] == unit_str:
                    return unit
            return None
        except (ValueError, IndexError):
            return None
