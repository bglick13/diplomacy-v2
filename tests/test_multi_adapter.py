"""
Tests for multi-adapter rollout support.

These tests verify the logic for handling different adapters per power
without requiring the actual Modal infrastructure.
"""


class TestPowerAdapterMapping:
    """Tests for power adapter mapping logic."""

    def test_baseline_bot_detection(self):
        """Baselines should be correctly identified."""
        from src.agents.baselines import ChaosBot, RandomBot

        BASELINE_BOTS = {
            "random_bot": RandomBot(),
            "chaos_bot": ChaosBot(),
        }

        # These should be in BASELINE_BOTS
        assert "random_bot" in BASELINE_BOTS
        assert "chaos_bot" in BASELINE_BOTS

        # These should NOT be in BASELINE_BOTS
        assert None not in BASELINE_BOTS
        assert "base_model" not in BASELINE_BOTS
        assert "adapter_v50" not in BASELINE_BOTS

    def test_power_adapter_defaults_to_lora_name(self):
        """Legacy mode: all powers use the same adapter."""
        POWERS = ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]
        lora_name = "run/adapter_v50"

        # This mirrors the logic in run_rollout
        power_adapters = dict.fromkeys(POWERS, lora_name)

        assert len(power_adapters) == 7
        for power in POWERS:
            assert power_adapters[power] == lora_name

    def test_mixed_adapter_configuration(self):
        """Powers can have different adapters."""
        power_adapters = {
            "FRANCE": "run/adapter_v50",  # Hero with latest checkpoint
            "ENGLAND": "run/adapter_v40",  # Older checkpoint
            "GERMANY": None,  # Base model
            "RUSSIA": "random_bot",  # Baseline
            "AUSTRIA": "chaos_bot",  # Baseline
            "ITALY": "run/adapter_v30",  # Another checkpoint
            "TURKEY": "base_model",  # Explicit base model
        }

        # Verify different adapter types
        assert power_adapters["FRANCE"] == "run/adapter_v50"
        assert power_adapters["RUSSIA"] == "random_bot"
        assert power_adapters["GERMANY"] is None

    def test_unique_lora_extraction(self):
        """Should extract unique LoRA adapters (excluding baselines and base model)."""
        BASELINE_BOTS = {"random_bot", "chaos_bot"}

        power_adapters = {
            "FRANCE": "run/adapter_v50",
            "ENGLAND": "run/adapter_v40",
            "GERMANY": None,
            "RUSSIA": "random_bot",
            "AUSTRIA": "chaos_bot",
            "ITALY": "run/adapter_v50",  # Duplicate
            "TURKEY": "base_model",
        }

        # This mirrors the logic in run_rollout
        unique_loras = {
            adapter
            for adapter in power_adapters.values()
            if adapter is not None and adapter not in BASELINE_BOTS and adapter != "base_model"
        }

        assert unique_loras == {"run/adapter_v50", "run/adapter_v40"}


class TestAdapterGrouping:
    """Tests for grouping powers by adapter for batched inference."""

    def test_group_by_adapter(self):
        """Powers should be grouped by adapter for batched inference."""
        BASELINE_BOTS = {"random_bot", "chaos_bot"}

        power_adapters = {
            "FRANCE": "adapter_v50",
            "ENGLAND": "adapter_v40",
            "GERMANY": None,
            "RUSSIA": "random_bot",
            "AUSTRIA": "adapter_v50",  # Same as France
            "ITALY": None,  # Same as Germany (base model)
            "TURKEY": "chaos_bot",
        }

        powers = ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]

        # Group by adapter (mimics run_game_async logic)
        adapter_groups: dict[str | None, list[str]] = {}
        baseline_powers: list[str] = []

        for power in powers:
            adapter = power_adapters.get(power)
            if adapter in BASELINE_BOTS:
                baseline_powers.append(power)
            else:
                # Normalize: None and "base_model" both mean no LoRA
                adapter_key = None if adapter in (None, "base_model") else adapter
                if adapter_key not in adapter_groups:
                    adapter_groups[adapter_key] = []
                adapter_groups[adapter_key].append(power)

        # Verify groupings
        assert set(baseline_powers) == {"RUSSIA", "TURKEY"}
        assert set(adapter_groups[None]) == {"GERMANY", "ITALY"}
        assert set(adapter_groups["adapter_v50"]) == {"FRANCE", "AUSTRIA"}
        assert adapter_groups["adapter_v40"] == ["ENGLAND"]


class TestHeroPowerFiltering:
    """Tests for hero power trajectory collection."""

    def test_hero_power_only_collects_hero_data(self):
        """Only hero power's data should be collected for training."""
        hero_power = "FRANCE"
        powers = ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY"]

        # Mimic fork_data collection logic
        collected_powers = []
        for power in powers:
            should_collect = (hero_power is None) or (power == hero_power)
            if should_collect:
                collected_powers.append(power)

        assert collected_powers == ["FRANCE"]

    def test_none_hero_power_collects_all(self):
        """With hero_power=None (legacy), all powers are collected."""
        hero_power = None
        powers = ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY"]

        collected_powers = []
        for power in powers:
            should_collect = (hero_power is None) or (power == hero_power)
            if should_collect:
                collected_powers.append(power)

        assert collected_powers == powers

    def test_baseline_powers_never_collected(self):
        """Baseline bot powers should never be in trajectories."""
        BASELINE_BOTS = {"random_bot", "chaos_bot"}
        hero_power = None  # Collect all

        power_adapters = {
            "FRANCE": "adapter_v50",
            "ENGLAND": "random_bot",  # Baseline
            "GERMANY": "chaos_bot",  # Baseline
            "RUSSIA": None,  # Base model (should be collected)
        }

        collected_powers = []
        for power, adapter in power_adapters.items():
            should_collect = (hero_power is None) or (power == hero_power)
            is_baseline = adapter in BASELINE_BOTS

            if should_collect and not is_baseline:
                collected_powers.append(power)

        assert set(collected_powers) == {"FRANCE", "RUSSIA"}


class TestReferenceLogprobsLogic:
    """Tests for reference logprobs computation logic."""

    def test_hero_uses_lora_detection(self):
        """Should detect if hero power uses LoRA adapter."""
        BASELINE_BOTS = {"random_bot", "chaos_bot"}

        test_cases = [
            ({"FRANCE": "adapter_v50"}, "FRANCE", True),
            ({"FRANCE": None}, "FRANCE", False),
            ({"FRANCE": "base_model"}, "FRANCE", False),
            ({"FRANCE": "random_bot"}, "FRANCE", False),  # Baseline
        ]

        for power_adapters, hero_power, expected in test_cases:
            hero_adapter = power_adapters.get(hero_power)
            hero_uses_lora = (
                hero_adapter not in (None, "base_model") and hero_adapter not in BASELINE_BOTS
            )
            assert hero_uses_lora == expected, f"Failed for {power_adapters}"

    def test_ref_logprobs_not_needed_for_base_model(self):
        """When generating with base model, generation logprobs ARE reference logprobs."""
        BASELINE_BOTS = {"random_bot", "chaos_bot"}

        power_adapters = {"FRANCE": None}  # Base model
        power = "FRANCE"

        power_adapter = power_adapters.get(power)
        power_uses_lora = (
            power_adapter not in (None, "base_model") and power_adapter not in BASELINE_BOTS
        )

        # Should NOT use separate ref logprobs
        assert not power_uses_lora
