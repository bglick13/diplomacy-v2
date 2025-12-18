from typing import TYPE_CHECKING, Any

import diplomacy
from diplomacy.utils.export import to_saved_game_format

if TYPE_CHECKING:
    from src.agents.llm_agent import LLMAgent


class DiplomacyWrapper:
    """
    Abstractions over the 'diplomacy' pip package to ensure consistent
    tokenization and state representation for LLMs.
    """

    def __init__(self, game_id: str | None = None, horizon: int = 2):
        self.game = diplomacy.Game(game_id=game_id)
        self.max_years = horizon
        self.start_year = self.get_year()
        self._orders_cache: tuple[str, dict] | None = None

    def is_done(self) -> bool:
        if self.game.is_game_done:
            return True
        # Check Horizon
        if self.get_year() >= self.start_year + self.max_years:
            return True
        return False

    def _get_possible_orders(self) -> dict[str, list[str]]:
        """Get all possible orders, cached per phase to avoid redundant engine calls."""
        phase_key = f"{self.game.phase}"
        if not self._orders_cache or self._orders_cache[0] != phase_key:
            self._orders_cache = (phase_key, self.game.get_all_possible_orders())
        return self._orders_cache[1]

    def get_current_phase(self) -> str:
        # Just return the standard phase string (e.g. "SPRING 1901 MOVEMENT")
        # The previous version prepended phase_type ('M'), which confused parsing.
        return str(self.game.phase)

    def get_year(self) -> int:
        """Robustly extract year from the game state."""
        # The diplomacy package usually formats phase as "SEASON YEAR TYPE"
        try:
            parts = str(self.game.phase).split()
            for part in parts:
                if part.isdigit():
                    return int(part)
            return 1900
        except Exception:  # noqa: BLE001
            return 1900

    def get_phase_type(self) -> str:
        """Get the phase type: M (Movement), R (Retreat), A (Adjustment)."""
        return str(self.game.phase_type)

    def get_valid_moves(self, power_name: str) -> dict[str, list[str]]:
        """
        Get valid moves for a power based on the current phase type.

        Returns a dict mapping:
        - MOVEMENT/RETREAT: unit_str -> list of orders (e.g. "A PAR" -> ["A PAR - BUR", ...])
        - ADJUSTMENT: location -> list of orders (e.g. "PAR" -> ["A PAR B", "WAIVE"])
        """
        if power_name not in self.game.powers:
            return {}

        phase_type = str(self.game.phase_type)
        possible_orders = self._get_possible_orders()

        if phase_type == "A":  # Adjustment phase (builds/disbands)
            return self._get_adjustment_moves(power_name, possible_orders)
        else:  # Movement or Retreat phase
            return self._get_movement_moves(power_name, possible_orders)

    def _get_movement_moves(self, power_name: str, possible_orders: dict) -> dict[str, list[str]]:
        """Get valid moves for movement/retreat phases."""
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

    def _get_adjustment_moves(self, power_name: str, possible_orders: dict) -> dict[str, list[str]]:
        """
        Get valid moves for adjustment phases (builds/disbands).

        During adjustments:
        - Builds: orderable_locations are empty home centers
        - Disbands: orderable_locations are all unit locations
        - Orders format: "A LOC B", "F LOC B", "WAIVE", "A LOC D", "F LOC D"
        """
        orderable_locs = self.game.get_orderable_locations(power_name)
        normalized_moves = {}

        for loc in orderable_locs:
            if loc in possible_orders:
                orders = possible_orders[loc]
                clean_orders = [str(order) for order in orders]
                # Use location as key (not unit string) for adjustments
                normalized_moves[loc] = clean_orders

        return normalized_moves

    def get_adjustment_delta(self, power_name: str) -> int:
        """
        Get the number of builds (positive) or disbands (negative) needed.
        """
        if power_name not in self.game.powers:
            return 0
        power = self.game.powers[power_name]
        return len(power.centers) - len(power.units)

    def get_all_inputs(self, agent: "LLMAgent") -> dict[str, list]:
        """
        Prepares the batch of inputs for all 7 powers to send to the GPU.

        Args:
            agent: Optional LLMAgent instance for prompt generation.
                   If None, uses a default LLMAgent.

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
            # Use agent to build prompt
            prompt, valid_moves = agent.build_prompt(self, power)

            prompts.append(prompt)
            valid_moves_list.append(valid_moves)
            power_names.append(power)

        return {
            "prompts": prompts,
            "valid_moves": valid_moves_list,
            "power_names": power_names,
        }

    def step(self, orders: list[str]):
        """
        Executes orders for the current phase.

        Movement phase orders: ['A PAR - BUR', 'F BRE H', ...]
        Adjustment phase orders: ['A PAR B', 'F LON B', 'WAIVE', 'A MUN D', ...]
        """
        phase_type = self.game.phase_type

        if phase_type == "A":
            self._step_adjustment(orders)
        else:
            self._step_movement(orders)

        self.game.process()
        self._orders_cache = None

    def _step_movement(self, orders: list[str]):
        """Handle movement/retreat phase orders."""
        # Map units to their Power
        unit_owner_map = {}
        for power_name, power_obj in self.game.powers.items():
            for unit in power_obj.units:
                unit_owner_map[unit] = power_name

        # Group orders by Power
        orders_by_power = {p: [] for p in self.game.powers}

        for order in orders:
            parts = order.split()
            if len(parts) < 2:
                continue

            # Extract unit string (e.g., "A PAR" from "A PAR - BUR")
            unit_str = f"{parts[0]} {parts[1]}"
            owner = unit_owner_map.get(unit_str)

            if owner:
                orders_by_power[owner].append(order)

        # Submit orders
        for power, power_orders in orders_by_power.items():
            if power_orders:
                try:
                    self.game.set_orders(power, power_orders)
                except Exception as e:
                    print(f"❌ Engine rejected orders for {power}: {e}")

    def _step_adjustment(self, orders: list[str]):
        """
        Handle adjustment phase orders (builds/disbands).

        Order formats:
        - Build: "A PAR B", "F LON B"
        - Disband: "A PAR D", "F LON D"
        - Waive: "WAIVE"
        """
        # Map orderable locations to their Power
        loc_owner_map = {}
        for power_name in self.game.powers:
            orderable_locs = self.game.get_orderable_locations(power_name)
            for loc in orderable_locs:
                loc_owner_map[loc] = power_name

        # Group orders by Power
        orders_by_power = {p: [] for p in self.game.powers}

        for order in orders:
            order = order.strip()
            if not order:
                continue

            if order.upper() == "WAIVE":
                # WAIVE applies to any power with builds available
                # We'll let each power that needs a waive get one
                continue  # Skip WAIVE for now, engine handles defaults

            parts = order.split()
            if len(parts) < 2:
                continue

            # Extract location from order (e.g., "PAR" from "A PAR B")
            loc = parts[1]

            # Handle coast notation (e.g., "STP/NC" from "F STP/NC B")
            base_loc = loc.split("/")[0]

            # Find owner by checking orderable locations
            owner = loc_owner_map.get(loc) or loc_owner_map.get(base_loc)

            if owner:
                orders_by_power[owner].append(order)

        # Submit orders
        for power, power_orders in orders_by_power.items():
            if power_orders:
                try:
                    self.game.set_orders(power, power_orders)
                except Exception as e:
                    print(f"❌ Engine rejected adjustment order for {power}: {e}")

    def get_state_json(self) -> dict[str, Any]:
        return to_saved_game_format(self.game)

    def get_board_context(
        self, power_name: str, include_windows: bool = True, include_action_counts: bool = False
    ) -> dict[str, Any]:
        """
        Get board context for strategic decision making.

        Returns a dict with:
        - my_units: List of our units
        - my_centers: List of our supply centers
        - opponent_units: Dict of power -> units for other powers
        - opponent_centers: Dict of power -> centers for other powers
        - unowned_centers: List of neutral supply centers
        - power_rankings: List of (power, center_count) sorted by centers
        - compact_map_view: Token-efficient per-unit neighbor windows
        """
        all_centers = set(self.game.map.scs)  # All supply centers on map

        my_units = list(self.game.powers[power_name].units)
        my_centers = list(self.game.powers[power_name].centers)

        opponent_units = {}
        opponent_centers = {}
        owned_centers = set(my_centers)

        for other_power, power_obj in self.game.powers.items():
            if other_power != power_name:
                units = list(power_obj.units)
                centers = list(power_obj.centers)
                if units:  # Only include powers with units (not eliminated)
                    opponent_units[other_power] = units
                if centers:
                    opponent_centers[other_power] = centers
                    owned_centers.update(centers)

        unowned_centers = list(all_centers - owned_centers)

        # Power rankings by supply centers
        power_rankings = []
        for pwr, power_obj in self.game.powers.items():
            center_count = len(power_obj.centers)
            if center_count > 0:  # Only include non-eliminated powers
                power_rankings.append((pwr, center_count))
        power_rankings.sort(key=lambda x: x[1], reverse=True)

        compact_map_view = (
            self.get_compact_map_view(power_name, include_action_counts=include_action_counts)
            if include_windows
            else ""
        )

        return {
            "my_units": my_units,
            "my_centers": my_centers,
            "opponent_units": opponent_units,
            "opponent_centers": opponent_centers,
            "unowned_centers": unowned_centers,
            "power_rankings": power_rankings,
            "compact_map_view": compact_map_view,
        }

    def get_compact_map_view(
        self, power_name: str, max_neighbors: int = 4, include_action_counts: bool = False
    ) -> str:
        """
        Return a compact, token-efficient map view focused on the player's units.

        Format (one line per unit):
            A PAR: PAR->BUR->PIC->GAS | n=BUR | e=GER:BUR
            (with action counts): A PAR (15 moves): PAR->BUR->PIC | n=BUR

        - Path shows the unit's province followed by up to `max_neighbors` adjacent spaces
        - n= lists adjacent neutral SCs
        - e= lists adjacent enemy units with their owning power
        - (15 moves) shows number of valid actions for this unit (if include_action_counts=True)
        """
        if power_name not in self.game.powers:
            return ""

        game_map = self.game.map
        my_units = list(self.game.powers[power_name].units)

        # Get valid moves for action counts
        valid_moves = self.get_valid_moves(power_name) if include_action_counts else {}

        # Owner lookup for supply centers
        owner_by_center: dict[str, str] = {}
        for pwr, power_obj in self.game.powers.items():
            for center in power_obj.centers:
                owner_by_center[center] = pwr

        # Enemy units keyed by location (base province, strip coasts)
        enemy_units: dict[str, str] = {}
        for pwr, power_obj in self.game.powers.items():
            if pwr == power_name:
                continue
            for unit in power_obj.units:
                parts = unit.split()
                if len(parts) >= 2:
                    loc = parts[1].split("/")[0]
                    enemy_units[loc] = pwr

        lines: list[str] = []
        for unit in my_units:
            parts = unit.split()
            if len(parts) < 2:
                continue
            loc = parts[1].split("/")[0]
            try:
                neighbors = game_map.abut_list(loc)
            except Exception:
                neighbors = []
            neighbor_path = "->".join([loc] + neighbors[:max_neighbors])

            # Neutral SCs adjacent
            neutral_adj = [
                n for n in neighbors if n in game_map.scs and owner_by_center.get(n) is None
            ][:2]

            # Enemy units adjacent
            enemy_adj = [f"{owner}:{n}" for n, owner in enemy_units.items() if n in neighbors][:3]

            # Build unit description with optional action count
            if include_action_counts:
                move_count = len(valid_moves.get(unit, []))
                # Classify action types
                moves = valid_moves.get(unit, [])
                can_move = any(" - " in m for m in moves)
                can_support = any(" S " in m for m in moves)
                can_convoy = any(" C " in m for m in moves)
                actions = []
                if can_move:
                    actions.append("move")
                if can_support:
                    actions.append("support")
                if can_convoy:
                    actions.append("convoy")
                action_str = ",".join(actions) if actions else "hold"
                unit_desc = f"{unit} ({move_count} moves | {action_str})"
            else:
                unit_desc = unit

            line_parts = [f"{unit_desc}: {neighbor_path}"]
            if neutral_adj:
                line_parts.append(f"n={','.join(neutral_adj)}")
            if enemy_adj:
                line_parts.append(f"e={','.join(enemy_adj)}")

            lines.append(" | ".join(line_parts))

        return "\n".join(lines)

    def get_unit(self, power_name: str, unit_str: str) -> str | None:
        units: list[str] = list(self.game.get_units(power_name))
        print(f"\nDEBUG get_unit: power={power_name}, location={unit_str}, units={units}")
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
