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
        print(
            f"DEBUG: All Possible Orders Keys: {game.game.get_all_possible_orders().keys()}"
        )

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
