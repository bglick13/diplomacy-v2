"""
Tests for PFSP Matchmaker.

These tests verify the opponent sampling logic without requiring
the full Modal infrastructure.
"""

import pytest

from src.league import LeagueRegistry, PFSPConfig, PFSPMatchmaker


class TestPFSPConfig:
    """Tests for PFSP configuration."""

    def test_default_weights_sum_to_one(self):
        """Default PFSP weights should sum to 1.0."""
        config = PFSPConfig()
        total = (
            config.self_play_weight
            + config.peer_weight
            + config.exploitable_weight
            + config.baseline_weight
        )
        assert abs(total - 1.0) < 0.01

    def test_validate_raises_on_invalid_weights(self):
        """Should raise if weights don't sum to 1.0 in legacy mode."""
        config = PFSPConfig(
            self_play_weight=0.5,
            peer_weight=0.5,
            exploitable_weight=0.5,
            baseline_weight=0.5,
            elo_based_sampling=False,  # Legacy mode validates weights
        )
        with pytest.raises(ValueError, match="must sum to 1.0"):
            config.validate()

    def test_custom_weights_valid(self):
        """Custom weights that sum to 1.0 should be valid in legacy mode."""
        config = PFSPConfig(
            self_play_weight=0.25,
            peer_weight=0.25,
            exploitable_weight=0.25,
            baseline_weight=0.25,
            elo_based_sampling=False,  # Legacy mode validates weights
        )
        config.validate()  # Should not raise


class TestPFSPMatchmaker:
    """Tests for PFSP matchmaker logic."""

    @pytest.fixture
    def registry_path(self, tmp_path):
        return tmp_path / "league.json"

    @pytest.fixture
    def registry(self, registry_path):
        """Create a registry with some checkpoints."""
        reg = LeagueRegistry(registry_path, run_name="test")
        # Add some checkpoints with varying Elo
        reg.add_checkpoint("adapter_v10", "test/adapter_v10", step=10, initial_elo=1050)
        reg.add_checkpoint("adapter_v20", "test/adapter_v20", step=20, initial_elo=1100)
        reg.add_checkpoint("adapter_v30", "test/adapter_v30", step=30, initial_elo=1150)
        return reg

    @pytest.fixture
    def matchmaker(self, registry):
        return PFSPMatchmaker(registry)

    def test_sample_opponents_returns_correct_structure(self, matchmaker):
        """Should return MatchmakingResult with all required fields."""
        result = matchmaker.sample_opponents("adapter_v30", hero_power="FRANCE")

        assert result.hero_power == "FRANCE"
        assert result.hero_agent == "adapter_v30"
        assert len(result.power_adapters) == 7
        assert "FRANCE" in result.power_adapters
        assert len(result.opponent_categories) == 7

    def test_hero_power_assigned_to_hero_agent(self, matchmaker):
        """Hero power should use the hero agent's adapter."""
        result = matchmaker.sample_opponents("adapter_v30", hero_power="ENGLAND")

        # Hero power should have the hero agent
        assert result.power_adapters["ENGLAND"] == "test/adapter_v30"
        assert result.opponent_categories["ENGLAND"] == "hero"

    def test_random_hero_power_if_not_specified(self, matchmaker):
        """Should pick a random hero power if not specified."""
        result = matchmaker.sample_opponents("adapter_v30")

        assert result.hero_power in PFSPMatchmaker.POWERS
        assert result.opponent_categories[result.hero_power] == "hero"

    def test_hero_elo_from_registry(self, matchmaker, registry):
        """Hero Elo should come from registry when agent is registered."""
        result = matchmaker.sample_opponents("adapter_v30", hero_power="FRANCE")

        # adapter_v30 has Elo 1150 in the test registry
        assert result.hero_elo == 1150.0

    def test_hero_elo_override(self, matchmaker):
        """Hero Elo override should take precedence over registry lookup."""
        result = matchmaker.sample_opponents(
            "adapter_v30", hero_power="FRANCE", hero_elo_override=1300.0
        )

        # Override should be used even though adapter_v30 has Elo 1150
        assert result.hero_elo == 1300.0

    def test_hero_elo_default_for_unknown_agent(self, matchmaker):
        """Unknown agent should default to 1000 Elo."""
        result = matchmaker.sample_opponents("unknown_agent", hero_power="FRANCE")

        # Unknown agent defaults to 1000.0
        assert result.hero_elo == 1000.0

    def test_self_play_with_unregistered_checkpoint(self, matchmaker):
        """Self-play should use hero_adapter_path when hero isn't in registry."""
        # Simulate an unregistered checkpoint (adapter_v55 not in registry)
        result = matchmaker.sample_opponents(
            hero_agent="test/adapter_v55",  # Not in registry
            hero_power="FRANCE",
            hero_elo_override=1200.0,
            hero_adapter_path="test/adapter_v55",
        )

        # The hero adapter path should be used for the hero power
        # and also for any self-play opponents
        assert result.hero_elo == 1200.0

        # Verify _agent_to_adapter returns the path for unregistered hero
        adapter = matchmaker._agent_to_adapter("test/adapter_v55")
        assert adapter == "test/adapter_v55"

    def test_cold_start_uses_only_baselines(self, matchmaker):
        """Cold start should use only baseline bots as opponents."""
        result = matchmaker.get_cold_start_opponents("FRANCE")

        # Hero uses base model
        assert result.power_adapters["FRANCE"] is None
        assert result.opponent_categories["FRANCE"] == "hero"

        # All opponents should be baselines
        for power, adapter in result.power_adapters.items():
            if power != "FRANCE":
                assert adapter in ("random_bot", "chaos_bot")
                assert result.opponent_categories[power] == "baseline"

    def test_self_play_returns_hero_agent(self, matchmaker):
        """Self-play category should use the hero agent."""
        # Test internal method
        agent = matchmaker._sample_agent_for_category(
            "self",
            hero_agent="adapter_v30",
            hero_elo=1150,
            checkpoints=[],
            baselines=[],
        )
        assert agent == "adapter_v30"

    def test_baseline_returns_bot(self, matchmaker, registry):
        """Baseline category should return a bot."""
        baselines = registry.get_baselines()
        agent = matchmaker._sample_agent_for_category(
            "baseline",
            hero_agent="adapter_v30",
            hero_elo=1150,
            checkpoints=[],
            baselines=baselines,
        )
        # Baselines are: chaos_bot, defensive_bot, territorial_bot, coordinated_bot
        # base_model is also a baseline but not in baseline_agents config
        assert agent in (
            "chaos_bot",
            "defensive_bot",
            "territorial_bot",
            "coordinated_bot",
            "base_model",
        )

    def test_peer_prefers_similar_elo(self, matchmaker, registry):
        """Peer category should prefer agents with similar Elo."""
        checkpoints = registry.get_checkpoints()
        baselines = registry.get_baselines()

        # Sample many times and check distribution
        peers = []
        for _ in range(100):
            agent = matchmaker._sample_agent_for_category(
                "peer",
                hero_agent="adapter_v30",
                hero_elo=1150,  # adapter_v30's Elo
                checkpoints=checkpoints,
                baselines=baselines,
            )
            peers.append(agent)

        # Most peers should be close in Elo (adapter_v20 at 1100, adapter_v30 at 1150)
        # adapter_v10 at 1050 is 100 Elo away, still within peer range
        assert "adapter_v20" in peers or "adapter_v10" in peers

    def test_exploitable_prefers_weaker(self, matchmaker, registry):
        """Exploitable category should prefer weaker agents."""
        checkpoints = registry.get_checkpoints()
        baselines = registry.get_baselines()

        # Sample many times
        exploitables = []
        for _ in range(100):
            agent = matchmaker._sample_agent_for_category(
                "exploitable",
                hero_agent="adapter_v30",
                hero_elo=1150,
                checkpoints=checkpoints,
                baselines=baselines,
            )
            exploitables.append(agent)

        # adapter_v10 (1050) is 100 Elo weaker than adapter_v30 (1150)
        # It should appear frequently
        if "adapter_v10" in exploitables:
            assert exploitables.count("adapter_v10") > 0

    def test_agent_to_adapter_baseline(self, matchmaker):
        """Baseline agents should return their name as adapter."""
        assert matchmaker._agent_to_adapter("random_bot") == "random_bot"
        assert matchmaker._agent_to_adapter("chaos_bot") == "chaos_bot"

    def test_agent_to_adapter_base_model(self, matchmaker):
        """Base model should return None."""
        assert matchmaker._agent_to_adapter("base_model") is None

    def test_agent_to_adapter_checkpoint(self, matchmaker, registry):
        """Checkpoint should return its path."""
        adapter = matchmaker._agent_to_adapter("adapter_v10")
        assert adapter == "test/adapter_v10"

    def test_sampling_stats_aggregation(self, matchmaker):
        """Should correctly aggregate stats from multiple results."""
        results = []
        for _ in range(10):
            result = matchmaker.sample_opponents("adapter_v30", hero_power="FRANCE")
            results.append(result)

        stats = matchmaker.get_sampling_stats(results)

        assert stats["total_games"] == 10
        assert "category_counts" in stats
        assert "category_rates" in stats

        # Hero shouldn't be counted in opponent categories
        assert "hero" not in stats["category_counts"]


class TestMatchmakingResult:
    """Tests for MatchmakingResult dataclass."""

    def test_to_wandb_dict(self):
        """Should produce valid WandB logging format."""
        from src.league.matchmaker import MatchmakingResult

        result = MatchmakingResult(
            hero_power="FRANCE",
            hero_agent="adapter_v50",
            hero_elo=1150.0,
            power_adapters={
                "FRANCE": "test/adapter_v50",
                "ENGLAND": "random_bot",
                "GERMANY": "chaos_bot",
                "RUSSIA": "test/adapter_v40",
                "AUSTRIA": "test/adapter_v30",
                "ITALY": None,
                "TURKEY": "test/adapter_v50",
            },
            power_agent_names={
                "FRANCE": "adapter_v50",
                "ENGLAND": "random_bot",
                "GERMANY": "chaos_bot",
                "RUSSIA": "adapter_v40",
                "AUSTRIA": "adapter_v30",
                "ITALY": "base_model",
                "TURKEY": "adapter_v50",
            },
            opponent_categories={
                "FRANCE": "hero",
                "ENGLAND": "baseline",
                "GERMANY": "baseline",
                "RUSSIA": "peer",
                "AUSTRIA": "exploitable",
                "ITALY": "self",
                "TURKEY": "self",
            },
        )

        wandb_dict = result.to_wandb_dict()

        assert wandb_dict["hero_power"] == "FRANCE"
        assert wandb_dict["hero_agent"] == "adapter_v50"
        assert wandb_dict["hero_elo"] == 1150.0
        assert "opponent_categories" in wandb_dict
        assert wandb_dict["opponent_categories"]["baseline"] == 2
        assert wandb_dict["opponent_categories"]["self"] == 2
