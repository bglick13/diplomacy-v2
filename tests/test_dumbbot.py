"""Tests for DumbBot baseline agent."""

import pytest

from src.agents.baselines import DumbBot
from src.engine.wrapper import DiplomacyWrapper


@pytest.fixture
def game() -> DiplomacyWrapper:
    """Create a fresh game."""
    return DiplomacyWrapper(horizon=10)


@pytest.fixture
def bot() -> DumbBot:
    """Create a DumbBot instance."""
    return DumbBot()


class TestDumbBotBasics:
    """Test basic DumbBot functionality."""

    def test_generates_orders_for_all_powers(self, game: DiplomacyWrapper, bot: DumbBot):
        """DumbBot should generate orders for all 7 powers."""
        for power in ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]:
            orders = bot.get_orders(game, power)
            assert isinstance(orders, list)
            assert len(orders) > 0, f"{power} should have orders in starting position"

    def test_order_count_matches_units(self, game: DiplomacyWrapper, bot: DumbBot):
        """Order count should match unit count for movement phases."""
        for power in ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]:
            orders = bot.get_orders(game, power)
            unit_count = len(game.game.powers[power].units)
            assert len(orders) == unit_count, (
                f"{power}: expected {unit_count} orders, got {len(orders)}"
            )

    def test_orders_are_valid_strings(self, game: DiplomacyWrapper, bot: DumbBot):
        """Orders should be non-empty strings."""
        for power in ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]:
            orders = bot.get_orders(game, power)
            for order in orders:
                assert isinstance(order, str)
                assert len(order) > 0


class TestDumbBotProvinceValues:
    """Test province value calculation."""

    def test_province_values_calculated(self, game: DiplomacyWrapper, bot: DumbBot):
        """Province values should be calculated for known locations."""
        values = bot._calculate_province_values(game, "FRANCE")
        assert isinstance(values, dict)
        # Should include SCs
        assert "PAR" in values or any("PAR" in k for k in values)
        # Values should be numeric
        assert all(isinstance(v, (int, float)) for v in values.values())

    def test_supply_centers_have_positive_value(self, game: DiplomacyWrapper, bot: DumbBot):
        """Supply centers should generally have positive value."""
        values = bot._calculate_province_values(game, "FRANCE")
        # At least some SCs should have positive value
        sc_values = [v for k, v in values.items() if k in game.game.map.scs]
        assert any(v > 0 for v in sc_values), "Some SCs should have positive value"


class TestDumbBotMultiPhase:
    """Test DumbBot through multiple game phases."""

    def test_survives_multiple_phases(self, game: DiplomacyWrapper, bot: DumbBot):
        """DumbBot should generate valid orders through multiple phases."""
        for _ in range(6):  # ~2 years
            if game.is_done():
                break

            all_orders = []
            for power in game.game.powers:
                orders = bot.get_orders(game, power)
                all_orders.extend(orders)

            # Should have some orders
            assert len(all_orders) > 0

            # Advance game
            game.step(all_orders)

    def test_handles_adjustment_phase(self, game: DiplomacyWrapper, bot: DumbBot):
        """DumbBot should handle build/disband phases."""
        # Advance to adjustment phase
        for _ in range(10):
            if game.is_done():
                break
            if game.get_phase_type() == "A":
                break

            all_orders = []
            for power in game.game.powers:
                orders = bot.get_orders(game, power)
                all_orders.extend(orders)
            game.step(all_orders)

        # If we hit adjustment phase, verify we can handle it
        if game.get_phase_type() == "A":
            for power in game.game.powers:
                orders = bot.get_orders(game, power)
                # Orders should be list (possibly empty for balanced powers)
                assert isinstance(orders, list)


class TestDumbBotDestinationExtraction:
    """Test destination extraction from order strings."""

    def test_extract_movement(self, bot: DumbBot):
        """Extract destination from movement orders."""
        assert bot._extract_destination("A PAR - BUR", "PAR") == "BUR"
        assert bot._extract_destination("F BRE - MAO", "BRE") == "MAO"
        assert bot._extract_destination("A MUN - BOH", "MUN") == "BOH"

    def test_extract_hold(self, bot: DumbBot):
        """Extract destination from hold orders."""
        assert bot._extract_destination("A PAR H", "PAR") == "PAR"
        assert bot._extract_destination("F LON HOLD", "LON") == "LON"

    def test_extract_support(self, bot: DumbBot):
        """Extract destination from support orders."""
        # Support move
        assert bot._extract_destination("A MUN S A PAR - BUR", "MUN") == "BUR"
        # Support hold
        assert bot._extract_destination("A MUN S A BUR", "MUN") == "BUR"

    def test_extract_convoy(self, bot: DumbBot):
        """Extract destination from convoy orders."""
        # Convoy orders stay in place
        assert bot._extract_destination("F NTH C A LON - NWY", "NTH") == "NTH"

    def test_extract_with_coast(self, bot: DumbBot):
        """Extract destination strips coast notation."""
        assert bot._extract_destination("F MAO - SPA/NC", "MAO") == "SPA"
        assert bot._extract_destination("F GOL - SPA/SC", "GOL") == "SPA"
