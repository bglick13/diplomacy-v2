"""
Tests for the League Training module.

Tests cover:
- LeagueRegistry CRUD operations
- Elo calculation (multi-player)
- Checkpoint promotion logic
- Type serialization/deserialization
"""

import tempfile
from pathlib import Path

import pytest

from src.league import (
    AgentInfo,
    AgentType,
    LeagueRegistry,
    MatchResult,
    OpponentType,
    get_checkpoint_name,
    should_add_to_league,
    update_elo_from_match,
    update_elo_multiplayer,
)
from src.league.elo import calculate_expected_score, elo_diff_for_win_rate


class TestAgentInfo:
    """Tests for AgentInfo dataclass."""

    def test_create_baseline(self):
        agent = AgentInfo.create_baseline("random_bot", elo=800.0)

        assert agent.name == "random_bot"
        assert agent.agent_type == AgentType.BASELINE
        assert agent.elo == 800.0
        assert agent.path is None
        assert agent.step is None

    def test_create_checkpoint(self):
        agent = AgentInfo.create_checkpoint(
            name="adapter_v50",
            path="run/adapter_v50",
            step=50,
            parent="adapter_v40",
            elo=1200.0,
        )

        assert agent.name == "adapter_v50"
        assert agent.agent_type == AgentType.CHECKPOINT
        assert agent.path == "run/adapter_v50"
        assert agent.step == 50
        assert agent.parent == "adapter_v40"
        assert agent.elo == 1200.0
        assert agent.created_at is not None

    def test_serialization_roundtrip(self):
        agent = AgentInfo.create_checkpoint(
            name="adapter_v100",
            path="run/adapter_v100",
            step=100,
            elo=1150.0,
        )

        # Serialize
        data = agent.to_dict()
        assert data["type"] == "checkpoint"
        assert data["path"] == "run/adapter_v100"

        # Deserialize
        restored = AgentInfo.from_dict("adapter_v100", data)
        assert restored.name == agent.name
        assert restored.agent_type == agent.agent_type
        assert restored.path == agent.path
        assert restored.step == agent.step
        assert restored.elo == agent.elo


class TestMatchResult:
    """Tests for MatchResult dataclass."""

    def test_serialization_roundtrip(self):
        match = MatchResult(
            game_id="game-123",
            step=50,
            power_agents={
                "FRANCE": "adapter_v50",
                "ENGLAND": "random_bot",
                "GERMANY": "chaos_bot",
            },
            scores={"FRANCE": 12.0, "ENGLAND": 3.0, "GERMANY": 5.0},
            rankings={"FRANCE": 1, "GERMANY": 2, "ENGLAND": 3},
            num_years=8,
            winner="adapter_v50",
        )

        data = match.to_dict()
        restored = MatchResult.from_dict(data)

        assert restored.game_id == match.game_id
        assert restored.step == match.step
        assert restored.power_agents == match.power_agents
        assert restored.scores == match.scores
        assert restored.winner == match.winner


class TestLeagueRegistry:
    """Tests for LeagueRegistry class."""

    @pytest.fixture
    def temp_registry_path(self):
        """Create a temporary file for the registry."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            yield Path(f.name)
        # Cleanup
        Path(f.name).unlink(missing_ok=True)

    def test_initialize_new_registry(self, temp_registry_path):
        """New registry should have default baselines."""
        registry = LeagueRegistry(temp_registry_path, run_name="test-run")

        # Should have 6 default baselines (removed random_bot, added stronger bots + dumb_bot)
        baselines = registry.get_baselines()
        assert len(baselines) == 6

        baseline_names = {b.name for b in baselines}
        assert baseline_names == {
            "base_model",
            "chaos_bot",
            "defensive_bot",
            "territorial_bot",
            "coordinated_bot",
            "dumb_bot",
        }

        # Metadata should be initialized
        assert registry.metadata is not None
        assert registry.metadata.run_name == "test-run"

    def test_add_checkpoint(self, temp_registry_path):
        """Test adding a checkpoint to the registry."""
        registry = LeagueRegistry(temp_registry_path, run_name="test-run")

        agent = registry.add_checkpoint(
            name="adapter_v10",
            path="test-run/adapter_v10",
            step=10,
            initial_elo=1050.0,
        )

        assert agent.name == "adapter_v10"
        assert registry.num_checkpoints == 1
        assert registry.latest_step == 10

        # Should be retrievable
        retrieved = registry.get_agent("adapter_v10")
        assert retrieved is not None
        assert retrieved.path == "test-run/adapter_v10"

    def test_add_checkpoint_inherits_parent_elo(self, temp_registry_path):
        """Checkpoint should inherit parent's Elo if not specified."""
        registry = LeagueRegistry(temp_registry_path, run_name="test-run")

        # Add parent with specific Elo
        registry.add_checkpoint("adapter_v10", "run/v10", step=10, initial_elo=1100.0)

        # Add child without specifying Elo
        child = registry.add_checkpoint("adapter_v20", "run/v20", step=20, parent="adapter_v10")

        assert child.elo == 1100.0  # Inherited from parent

    def test_persistence(self, temp_registry_path):
        """Registry should persist to disk and reload correctly."""
        # Create and populate
        registry1 = LeagueRegistry(temp_registry_path, run_name="test-run")
        registry1.add_checkpoint("adapter_v10", "run/v10", step=10, initial_elo=1100.0)
        registry1.update_elo("adapter_v10", 1150.0)

        # Create new instance from same file
        registry2 = LeagueRegistry(temp_registry_path)

        # Should have same data
        assert registry2.num_checkpoints == 1
        agent = registry2.get_agent("adapter_v10")
        assert agent is not None
        assert agent.elo == 1150.0

    def test_update_elo(self, temp_registry_path):
        """Test Elo updates."""
        registry = LeagueRegistry(temp_registry_path, run_name="test-run")
        registry.add_checkpoint("adapter_v10", "run/v10", step=10, initial_elo=1000.0)

        registry.update_elo("adapter_v10", 1200.0)

        agent = registry.get_agent("adapter_v10")
        assert agent is not None
        assert agent.elo == 1200.0
        assert agent.matches == 1

        # Best agent should update
        assert registry.best_elo == 1200.0
        assert registry.best_agent == "adapter_v10"

    def test_bulk_update_elos(self, temp_registry_path):
        """Test bulk Elo updates."""
        registry = LeagueRegistry(temp_registry_path, run_name="test-run")
        registry.add_checkpoint("adapter_v10", "run/v10", step=10, initial_elo=1000.0)
        registry.add_checkpoint("adapter_v20", "run/v20", step=20, initial_elo=1000.0)

        registry.bulk_update_elos({"adapter_v10": 1100.0, "adapter_v20": 1200.0})

        agent1 = registry.get_agent("adapter_v10")
        assert agent1 is not None
        assert agent1.elo == 1100.0
        agent2 = registry.get_agent("adapter_v20")
        assert agent2 is not None
        assert agent2.elo == 1200.0
        assert registry.best_agent == "adapter_v20"

    def test_match_history(self, temp_registry_path):
        """Test match history tracking."""
        registry = LeagueRegistry(temp_registry_path, run_name="test-run")

        match = MatchResult(
            game_id="game-1",
            step=10,
            power_agents={"FRANCE": "adapter_v10", "ENGLAND": "random_bot"},
            scores={"FRANCE": 10.0, "ENGLAND": 5.0},
            rankings={"FRANCE": 1, "ENGLAND": 2},
            num_years=5,
        )

        registry.add_match(match)
        registry.save_history()

        recent = registry.get_recent_matches(10)
        assert len(recent) == 1
        assert recent[0].game_id == "game-1"


class TestShouldAddToLeague:
    """Tests for checkpoint promotion logic."""

    @pytest.fixture
    def registry(self, tmp_path):
        return LeagueRegistry(tmp_path / "league.json", run_name="test")

    def test_first_checkpoint_always_added(self, registry):
        """First checkpoint should always be added."""
        assert should_add_to_league(step=1, registry=registry)

    def test_early_training_every_5_steps(self, registry):
        """Early training (step < 50) should add every 5 steps."""
        # Add a checkpoint first
        registry.add_checkpoint("adapter_v0", "run/v0", step=0)

        # Step 5 should be added (early training: every 5 steps)
        assert should_add_to_league(step=5, registry=registry)

        # Step 7 should not (not divisible by 5)
        assert not should_add_to_league(step=7, registry=registry)

        # Step 15 should be added (15 % 5 == 0)
        assert should_add_to_league(step=15, registry=registry)

    def test_mid_training_every_10_steps(self, registry):
        """Mid training (50 <= step < 200) should add every 10 steps."""
        # Add a checkpoint first
        registry.add_checkpoint("adapter_v0", "run/v0", step=0)

        # Step 50 should be added (50 % 10 == 0)
        assert should_add_to_league(step=50, registry=registry)

        # Step 55 should not (55 % 10 != 0)
        assert not should_add_to_league(step=55, registry=registry)

        # Step 60 should be added
        assert should_add_to_league(step=60, registry=registry)

    def test_every_100_steps_anchor(self, registry):
        """Every 100 steps should always be added."""
        registry.add_checkpoint("adapter_v0", "run/v0", step=0)

        assert should_add_to_league(step=100, registry=registry)
        assert should_add_to_league(step=200, registry=registry)

    def test_new_best_display_rating(self, registry):
        """New best display_rating should trigger checkpoint."""
        registry.add_checkpoint("adapter_v0", "run/v0", step=0)

        # Step 53 doesn't match any interval rule (not 5, 10, 20, or 100 divisor)
        # Not a new best
        assert not should_add_to_league(step=53, registry=registry, current_display_rating=-5.0)

        # New best!
        assert should_add_to_league(step=53, registry=registry, current_display_rating=10.0)


class TestEloCalculation:
    """Tests for Elo calculation functions."""

    def test_expected_score_equal_ratings(self):
        """Equal ratings should give 0.5 expected score."""
        expected = calculate_expected_score(1000.0, 1000.0)
        assert abs(expected - 0.5) < 0.001

    def test_expected_score_higher_rating_favored(self):
        """Higher rating should give > 0.5 expected score."""
        expected = calculate_expected_score(1200.0, 1000.0)
        assert expected > 0.5

        # 200 point difference should give ~0.76
        assert 0.75 < expected < 0.77

    def test_update_elo_multiplayer_symmetric(self):
        """Symmetric game results should preserve total Elo."""
        initial_elos = {"a": 1000.0, "b": 1000.0, "c": 1000.0}
        game_results = {"a": 10.0, "b": 5.0, "c": 3.0}

        new_elos = update_elo_multiplayer(game_results, initial_elos)

        # Winner should gain, losers should lose
        assert new_elos["a"] > 1000.0
        assert new_elos["c"] < 1000.0

        # Total Elo should be approximately preserved
        total_before = sum(initial_elos.values())
        total_after = sum(new_elos.values())
        assert abs(total_before - total_after) < 0.01

    def test_update_elo_from_match(self):
        """Test convenience wrapper for match-based Elo updates."""
        power_agents = {
            "FRANCE": "adapter_v50",
            "ENGLAND": "random_bot",
            "GERMANY": "chaos_bot",
        }
        power_scores = {"FRANCE": 12.0, "ENGLAND": 3.0, "GERMANY": 5.0}
        agent_elos = {"adapter_v50": 1200.0, "random_bot": 800.0, "chaos_bot": 900.0}

        new_elos = update_elo_from_match(power_agents, power_scores, agent_elos)

        # adapter_v50 won, should gain (but not much since expected to win)
        assert new_elos["adapter_v50"] >= agent_elos["adapter_v50"]

        # random_bot lost badly, should lose Elo
        assert new_elos["random_bot"] < agent_elos["random_bot"]

    def test_elo_diff_for_win_rate(self):
        """Test Elo difference calculation for target win rates."""
        # 75% win rate should be ~191 Elo
        diff_75 = elo_diff_for_win_rate(0.75)
        assert 180 < diff_75 < 200

        # 90% win rate should be ~382 Elo
        diff_90 = elo_diff_for_win_rate(0.90)
        assert 370 < diff_90 < 400


class TestOpponentType:
    """Tests for OpponentType enum."""

    def test_opponent_type_values(self):
        """OpponentType should have expected values."""
        assert OpponentType.RANDOM.value == "random"
        assert OpponentType.CHAOS.value == "chaos"
        assert OpponentType.BASE_MODEL.value == "base_model"
        assert OpponentType.CHECKPOINT.value == "checkpoint"
        assert OpponentType.SELF.value == "self"

    def test_opponent_to_agent_name(self):
        """Should convert OpponentType to agent registry name."""
        from src.league import opponent_type_to_agent_name

        assert opponent_type_to_agent_name(OpponentType.RANDOM) == "random_bot"
        assert opponent_type_to_agent_name(OpponentType.CHAOS) == "chaos_bot"
        assert opponent_type_to_agent_name(OpponentType.BASE_MODEL) == "base_model"


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_checkpoint_name(self):
        """Test checkpoint naming convention."""
        name = get_checkpoint_name("grpo-20251206", 50)
        assert name == "grpo-20251206/adapter_v50"
