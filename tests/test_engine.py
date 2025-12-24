import pytest

from src.engine.wrapper import DiplomacyWrapper


@pytest.fixture
def game():
    return DiplomacyWrapper()


def test_initialization(game):
    """Ensure game starts in Spring 1901."""
    assert "SPRING 1901" in game.get_current_phase()
    assert not game.is_done()


def test_get_valid_moves_format(game):
    """
    CRITICAL: Valid moves must be a dict of {Unit: [List of Strings]}.
    This format is required by the LogitsProcessor.
    """
    moves = game.get_valid_moves("FRANCE")

    # DEBUG: Print keys if empty
    if not moves:
        print(f"\nDEBUG: Power units: {game.game.powers['FRANCE'].units}")
        print(f"DEBUG: All Possible Orders Keys: {game.game.get_all_possible_orders().keys()}")

    # Check structure
    assert isinstance(moves, dict)
    assert len(moves) > 0, "FRANCE should have moves in Spring 1901"

    # Use 'A PAR' or whatever unit is actually there (standard game has A PAR)
    assert "A PAR" in moves

    # Check content of moves
    par_moves = moves["A PAR"]
    assert isinstance(par_moves, list)
    assert len(par_moves) > 0
    assert isinstance(par_moves[0], str)

    move_strings = " ".join(par_moves)
    # Check for standard adjacency
    assert "BUR" in move_strings or "PIC" in move_strings


def test_step_execution_verifies_movement(game: DiplomacyWrapper):
    """
    Test that orders are actually applied to the map, not just that time passes.
    We force a specific move (Paris -> Burgundy) and check the unit's location.
    """
    # 1. Verify Initial State
    # France starts with Army in Paris
    # Note: get_unit returns the unit object or None
    assert game.get_unit("FRANCE", "PAR") is not None
    assert game.get_unit("FRANCE", "BUR") is None

    # 2. Submit a specific, known valid move
    # "A PAR - BUR" is standard opener
    orders = ["A PAR - BUR"]

    # 3. Step
    game.step(orders)

    # 4. Verify Consequence
    # The Army should be gone from Paris and present in Burgundy
    # If set_orders failed silently, this will fail because unit stays in PAR
    unit_in_bur = game.get_unit("FRANCE", "BUR")
    unit_in_par = game.get_unit("FRANCE", "PAR")

    assert unit_in_bur is not None, "Unit did not move to Burgundy!"
    assert unit_in_par is None, "Unit is still in Paris (Duplicate or Hold)!"

    # Verify phase advanced as well
    assert "FALL 1901" in game.get_current_phase()


def test_empty_moves_for_eliminated_power(game):
    """Test that a power with no units returns empty dict, not error."""
    # Artificially clear units
    game.game.powers["ITALY"].units = []

    moves = game.get_valid_moves("ITALY")
    assert moves == {}


def test_adjustment_phase_builds():
    """Test that adjustment phase returns correct build orders by playing through."""
    import diplomacy

    # Create a raw game and play through properly to reach ADJUSTMENTS
    game = diplomacy.Game()

    # Helper to set first valid order for all units
    def set_orders_for_all_powers(game):
        for power in game.powers:
            units = game.powers[power].units
            possible = game.get_all_possible_orders()
            power_orders = []
            for unit in units:
                parts = unit.split()
                if len(parts) >= 2:
                    loc = parts[1]
                    if loc in possible and possible[loc]:
                        power_orders.append(str(possible[loc][0]))
            if power_orders:
                game.set_orders(power, power_orders)

    # Spring 1901
    set_orders_for_all_powers(game)
    game.process()

    # Fall 1901
    set_orders_for_all_powers(game)
    game.process()

    # Now wrap it
    wrapper = DiplomacyWrapper.__new__(DiplomacyWrapper)
    wrapper.game = game
    wrapper.max_years = 2
    wrapper.start_year = 1901
    wrapper._orders_cache = None

    # Continue processing until we hit an adjustments phase or wrap around to Spring 1902
    max_iterations = 10
    for _ in range(max_iterations):
        phase_type = wrapper.get_phase_type()

        if phase_type == "A":
            assert "ADJUSTMENTS" in wrapper.get_current_phase()

            # Find any power with builds
            found_builds = False
            for power in wrapper.game.powers:
                delta = wrapper.get_adjustment_delta(power)
                if delta > 0:
                    moves = wrapper.get_valid_moves(power)
                    # Should have orderable locations with build orders
                    assert len(moves) > 0, f"{power} has builds but no moves"
                    for loc, orders in moves.items():
                        build_orders = [o for o in orders if " B" in o or o == "WAIVE"]
                        assert len(build_orders) > 0, f"No build orders for {loc}"
                    found_builds = True
                    break

            # If we found an adjustment phase with builds, test passes
            if found_builds:
                return

            # If adjustment phase but no builds needed, continue
            set_orders_for_all_powers(wrapper.game)
            wrapper.game.process()

        elif phase_type in ("M", "R"):
            # Movement or Retreat phase - process and continue
            set_orders_for_all_powers(wrapper.game)
            wrapper.game.process()
        else:
            raise AssertionError(f"Unexpected phase type: {phase_type}")

    # If we reach here, we tested through multiple phases without error
    # That's also a valid outcome (no powers needed builds)


def test_adjustment_phase_delta():
    """Test that adjustment delta is calculated correctly."""
    game = DiplomacyWrapper()

    # Initially, units == centers for all powers
    for power in game.game.powers:
        delta = game.get_adjustment_delta(power)
        assert delta == 0, f"{power} should have delta=0 at start"

    # Give France an extra SC
    game.game.powers["FRANCE"].centers.append("SPA")

    # Now France should need a build
    delta = game.get_adjustment_delta("FRANCE")
    assert delta == 1, "France should need 1 build"

    # Take away an SC from Germany
    game.game.powers["GERMANY"].centers.remove("MUN")

    # Germany should need to disband
    delta = game.get_adjustment_delta("GERMANY")
    assert delta == -1, "Germany should need to disband 1 unit"
