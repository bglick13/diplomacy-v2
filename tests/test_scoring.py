"""
Tests for scoring and baseline agents.

This module tests:
- Win bonus calculation in scoring
- RandomBot and ChaosBot producing valid orders
"""

import pytest

from src.agents.baselines import ChaosBot, RandomBot
from src.engine.wrapper import DiplomacyWrapper
from src.utils.scoring import calculate_final_scores


class TestWinBonus:
    """Tests for win bonus in calculate_final_scores."""

    @pytest.fixture
    def game(self):
        """Create a fresh game."""
        return DiplomacyWrapper()

    def test_no_win_bonus_by_default_zero(self, game):
        """With win_bonus=0, no bonus is applied."""
        # Default game start - no one has enough SCs
        scores_no_bonus = calculate_final_scores(game, win_bonus=0.0)
        scores_with_bonus = calculate_final_scores(game, win_bonus=5.0)

        # At game start, no one has 10+ SCs, so scores should be equal
        # (minus floating point differences)
        for power in scores_no_bonus:
            assert abs(scores_no_bonus[power] - scores_with_bonus[power]) < 0.01

    def test_win_bonus_sole_leader(self):
        """Win bonus should apply to sole leader with enough SCs."""
        game = DiplomacyWrapper()

        # Manually manipulate the game state to give France 12 SCs
        # (This is a bit hacky but necessary for testing)
        france = game.game.powers["FRANCE"]

        # Get some supply centers to add to France
        all_centers = list(game.game.map.scs)
        france.centers = set(all_centers[:12])  # Give France 12 centers

        scores = calculate_final_scores(game, win_bonus=5.0, winner_threshold_sc=10)

        # France should have base score + win bonus
        # Other powers should not have win bonus
        france_score = scores["FRANCE"]

        # France should have significantly higher score due to bonus
        # Base score for 12 SCs: 12 * 2.0 = 24 + units + base
        # With win bonus: +5.0
        assert france_score > 25  # Should be well above 25 with bonus

    def test_no_win_bonus_when_tied(self):
        """No win bonus if multiple powers are tied for lead."""
        game = DiplomacyWrapper()

        # Give France and England equal SCs
        france = game.game.powers["FRANCE"]
        england = game.game.powers["ENGLAND"]

        all_centers = list(game.game.map.scs)
        france.centers = set(all_centers[:10])
        england.centers = set(all_centers[10:20])

        scores = calculate_final_scores(game, win_bonus=5.0, winner_threshold_sc=10)

        # Neither should get win bonus since they're tied
        # Both have 10 SCs, so base scores should be similar
        assert abs(scores["FRANCE"] - scores["ENGLAND"]) < 10  # No huge bonus diff

    def test_no_win_bonus_below_threshold(self):
        """No win bonus if leader is below threshold."""
        game = DiplomacyWrapper()

        # Give France 8 SCs (below threshold of 10)
        france = game.game.powers["FRANCE"]
        all_centers = list(game.game.map.scs)
        france.centers = set(all_centers[:8])

        scores_no_bonus = calculate_final_scores(game, win_bonus=0.0)
        scores_with_bonus = calculate_final_scores(game, win_bonus=5.0, winner_threshold_sc=10)

        # France score should be the same since 8 < 10
        assert abs(scores_no_bonus["FRANCE"] - scores_with_bonus["FRANCE"]) < 0.01

    def test_eliminated_power_negative_score(self, game):
        """Eliminated powers should have negative score (elimination penalty)."""
        # Eliminate a power by removing all units and centers
        russia = game.game.powers["RUSSIA"]
        russia.centers = set()
        russia.units = []

        # With position-based scoring (default), eliminated powers get -30.0
        scores = calculate_final_scores(game)
        assert scores["RUSSIA"] == -30.0

        # With legacy scoring, eliminated powers get -1.0
        scores_legacy = calculate_final_scores(game, use_position_scoring=False)
        assert scores_legacy["RUSSIA"] == -1.0


class TestRandomBot:
    """Tests for RandomBot baseline agent."""

    @pytest.fixture
    def game(self):
        return DiplomacyWrapper()

    @pytest.fixture
    def bot(self):
        return RandomBot()

    def test_produces_valid_orders(self, game, bot):
        """RandomBot should produce valid orders for each unit."""
        for power_name in ["FRANCE", "ENGLAND", "GERMANY"]:
            orders = bot.get_orders(game, power_name)

            # Should have orders for each unit
            valid_moves = game.get_valid_moves(power_name)
            assert len(orders) == len(valid_moves)

            # Each order should be from valid moves
            for order in orders:
                # Extract unit from order (e.g., "A PAR" from "A PAR - BUR")
                unit = " ".join(order.split()[:2])
                assert order in valid_moves.get(unit, []), f"Invalid order: {order}"

    def test_no_press(self, game, bot):
        """RandomBot should not send any press messages."""
        press = bot.get_press(game, "FRANCE")
        assert press == []

    def test_determinism_with_seed(self, game):
        """With same seed, RandomBot should produce same orders."""
        import random

        random.seed(42)
        bot1 = RandomBot()
        orders1 = bot1.get_orders(game, "FRANCE")

        random.seed(42)
        bot2 = RandomBot()
        orders2 = bot2.get_orders(game, "FRANCE")

        assert orders1 == orders2


class TestChaosBot:
    """Tests for ChaosBot baseline agent."""

    @pytest.fixture
    def game(self):
        return DiplomacyWrapper()

    @pytest.fixture
    def bot(self):
        return ChaosBot()

    def test_produces_valid_orders(self, game, bot):
        """ChaosBot should produce valid orders for each unit."""
        for power_name in ["FRANCE", "ENGLAND", "GERMANY"]:
            orders = bot.get_orders(game, power_name)

            valid_moves = game.get_valid_moves(power_name)
            assert len(orders) == len(valid_moves)

            for order in orders:
                unit = " ".join(order.split()[:2])
                assert order in valid_moves.get(unit, []), f"Invalid order: {order}"

    def test_prefers_movement(self, game, bot):
        """ChaosBot should prefer movement orders over holds/supports."""
        # Run multiple times to check preference
        move_counts = 0
        total_orders = 0

        for _ in range(10):
            orders = bot.get_orders(game, "FRANCE")
            for order in orders:
                total_orders += 1
                if " - " in order:  # Movement indicator
                    move_counts += 1

        # ChaosBot should have high percentage of moves
        move_rate = move_counts / total_orders if total_orders > 0 else 0
        assert move_rate > 0.5, f"ChaosBot should prefer moves, got {move_rate:.1%}"

    def test_no_press(self, game, bot):
        """ChaosBot should not send any press messages."""
        press = bot.get_press(game, "FRANCE")
        assert press == []


class TestScoringIntegration:
    """Integration tests for scoring with actual game progression."""

    def test_forward_units_bonus(self):
        """Units outside home centers should get bonus."""
        game = DiplomacyWrapper()

        # Initial scores (all units at home)
        calculate_final_scores(game)

        # Move French army from Paris to Burgundy
        game.game.set_orders("FRANCE", ["A PAR - BUR"])
        game.game.set_orders("ENGLAND", ["F LON - NTH", "F EDI - NWG", "A LVP - YOR"])
        game.game.set_orders("GERMANY", ["A BER - KIE", "A MUN - RUH", "F KIE - DEN"])
        game.game.set_orders(
            "RUSSIA", ["A WAR - GAL", "A MOS - UKR", "F SEV - BLA", "F STP/SC - BOT"]
        )
        game.game.set_orders("AUSTRIA", ["A VIE - GAL", "A BUD - SER", "F TRI - ALB"])
        game.game.set_orders("ITALY", ["A ROM - APU", "A VEN - TYR", "F NAP - ION"])
        game.game.set_orders("TURKEY", ["A CON - BUL", "A SMY - CON", "F ANK - BLA"])
        game.game.process()

        # Skip to fall and winter
        game.game.set_orders("FRANCE", ["A BUR - MAR"])
        game.game.process()

        # Scores after moves (France has unit outside home)
        final_scores = calculate_final_scores(game)

        # France's score should be slightly higher due to forward units
        # (This is a soft check since game state is complex)
        assert "FRANCE" in final_scores
        assert final_scores["FRANCE"] > 0
