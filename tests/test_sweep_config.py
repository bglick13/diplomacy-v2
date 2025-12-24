# tests/test_sweep_config.py
"""Tests for sweep configuration loading and validation."""

import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.utils.sweep_config import (
    AnalysisConfig,
    RunConfig,
    SweepConfig,
    SweepMetadata,
    SweepState,
    _resolve_interpolation,
)


class TestResolveInterpolation:
    """Tests for variable interpolation in YAML."""

    def test_simple_interpolation(self):
        """Test basic ${path} interpolation."""
        data = {
            "metadata": {"name": "test-sweep", "prefix": "test"},
            "runs": {"A": {"tag": "${metadata.prefix}-A"}},
        }
        result = _resolve_interpolation(data)
        assert result["runs"]["A"]["tag"] == "test-A"

    def test_nested_interpolation(self):
        """Test interpolation with nested paths."""
        data = {
            "metadata": {"nested": {"value": "hello"}},
            "field": "${metadata.nested.value}-world",
        }
        result = _resolve_interpolation(data)
        assert result["field"] == "hello-world"

    def test_no_interpolation_needed(self):
        """Test data without ${} patterns passes through."""
        data = {"key": "value", "nested": {"a": 1}}
        result = _resolve_interpolation(data)
        assert result == data

    def test_missing_path_unchanged(self):
        """Test missing paths are left unchanged."""
        data = {"field": "${nonexistent.path}"}
        result = _resolve_interpolation(data)
        assert result["field"] == "${nonexistent.path}"

    def test_list_interpolation(self):
        """Test interpolation works in lists."""
        data = {
            "metadata": {"prefix": "test"},
            "items": ["${metadata.prefix}-1", "${metadata.prefix}-2"],
        }
        result = _resolve_interpolation(data)
        assert result["items"] == ["test-1", "test-2"]


class TestSweepMetadata:
    """Tests for SweepMetadata model."""

    def test_required_fields(self):
        """Test that required fields are enforced."""
        with pytest.raises(ValidationError):
            SweepMetadata()  # type: ignore

    def test_minimal_metadata(self):
        """Test minimal valid metadata."""
        meta = SweepMetadata(name="test", experiment_tag_prefix="test-prefix")
        assert meta.name == "test"
        assert meta.experiment_tag_prefix == "test-prefix"
        assert meta.description == ""
        assert meta.hypothesis == ""

    def test_full_metadata(self):
        """Test metadata with all fields."""
        meta = SweepMetadata(
            name="test-sweep",
            description="Test description",
            hypothesis="Test hypothesis",
            experiment_tag_prefix="test",
            created="2024-01-01",
            author="tester",
        )
        assert meta.author == "tester"
        assert meta.created == "2024-01-01"


class TestRunConfig:
    """Tests for RunConfig model."""

    def test_required_fields(self):
        """Test that name is required."""
        with pytest.raises(ValidationError):
            RunConfig()  # type: ignore

    def test_minimal_run_config(self):
        """Test minimal run config."""
        run = RunConfig(name="baseline")
        assert run.name == "baseline"
        assert run.description == ""
        assert run.config == {}

    def test_run_with_config_overrides(self):
        """Test run with config overrides."""
        run = RunConfig(
            name="treatment",
            description="With longer horizon",
            config={"rollout_horizon_years": 8, "learning_rate": 1e-5},
        )
        assert run.config["rollout_horizon_years"] == 8
        assert run.config["learning_rate"] == 1e-5


class TestSweepConfig:
    """Tests for SweepConfig model."""

    def test_minimal_sweep_config(self):
        """Test minimal valid sweep config."""
        config = SweepConfig(
            metadata=SweepMetadata(name="test", experiment_tag_prefix="test"),
            runs={"A": RunConfig(name="baseline")},
        )
        assert len(config.runs) == 1
        assert "A" in config.runs

    def test_empty_runs_rejected(self):
        """Test that empty runs dict is rejected."""
        with pytest.raises(ValueError):
            SweepConfig(
                metadata=SweepMetadata(name="test", experiment_tag_prefix="test"),
                runs={},
            )

    def test_get_run_ids(self):
        """Test get_run_ids returns keys in order."""
        config = SweepConfig(
            metadata=SweepMetadata(name="test", experiment_tag_prefix="test"),
            runs={
                "A": RunConfig(name="first"),
                "B": RunConfig(name="second"),
                "C": RunConfig(name="third"),
            },
        )
        assert config.get_run_ids() == ["A", "B", "C"]

    def test_build_experiment_config(self):
        """Test building ExperimentConfig from run."""
        config = SweepConfig(
            metadata=SweepMetadata(name="test-sweep", experiment_tag_prefix="test"),
            defaults={"total_steps": 50},
            runs={
                "A": RunConfig(name="baseline", config={"learning_rate": 1e-5}),
            },
        )
        exp_cfg = config.build_experiment_config("A", timestamp="20240101-120000")

        # Check defaults applied
        assert exp_cfg.total_steps == 50
        # Check run override applied
        assert exp_cfg.learning_rate == 1e-5
        # Check auto-generated values
        assert exp_cfg.run_name == "test-sweep-A-20240101-120000"
        assert exp_cfg.experiment_tag == "test-A"

    def test_build_experiment_config_override_defaults(self):
        """Test that run config overrides defaults."""
        config = SweepConfig(
            metadata=SweepMetadata(name="test", experiment_tag_prefix="test"),
            defaults={"total_steps": 100, "learning_rate": 5e-6},
            runs={
                "A": RunConfig(
                    name="treatment",
                    config={"learning_rate": 1e-5},  # Override default
                ),
            },
        )
        exp_cfg = config.build_experiment_config("A")

        assert exp_cfg.total_steps == 100  # From defaults
        assert exp_cfg.learning_rate == 1e-5  # Overridden by run

    def test_build_experiment_config_invalid_run(self):
        """Test that invalid run ID raises error."""
        config = SweepConfig(
            metadata=SweepMetadata(name="test", experiment_tag_prefix="test"),
            runs={"A": RunConfig(name="baseline")},
        )
        with pytest.raises(ValueError, match="Unknown run ID"):
            config.build_experiment_config("B")


class TestSweepConfigYAML:
    """Tests for YAML loading."""

    def test_load_from_yaml(self):
        """Test loading sweep config from YAML file."""
        yaml_content = """
metadata:
  name: "test-sweep"
  description: "Test description"
  experiment_tag_prefix: "test"

defaults:
  total_steps: 100

runs:
  A:
    name: "baseline"
    config:
      experiment_tag: "test-A"
  B:
    name: "treatment"
    config:
      rollout_horizon_years: 8
      experiment_tag: "test-B"
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "sweep.yaml"
            yaml_path.write_text(yaml_content)

            config = SweepConfig.from_yaml(yaml_path)

            assert config.metadata.name == "test-sweep"
            assert len(config.runs) == 2
            assert config.defaults["total_steps"] == 100

    def test_load_from_directory(self):
        """Test loading from directory containing sweep.yaml."""
        yaml_content = """
metadata:
  name: "dir-test"
  experiment_tag_prefix: "test"

runs:
  A:
    name: "only-run"
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "sweep.yaml"
            yaml_path.write_text(yaml_content)

            # Pass directory, not file
            config = SweepConfig.from_yaml(Path(tmpdir))

            assert config.metadata.name == "dir-test"

    def test_load_with_interpolation(self):
        """Test that variable interpolation works in YAML."""
        yaml_content = """
metadata:
  name: "interp-test"
  experiment_tag_prefix: "interp"

runs:
  A:
    name: "baseline"
    config:
      experiment_tag: "${metadata.experiment_tag_prefix}-A"
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "sweep.yaml"
            yaml_path.write_text(yaml_content)

            config = SweepConfig.from_yaml(yaml_path)

            assert config.runs["A"].config["experiment_tag"] == "interp-A"

    def test_load_missing_file(self):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            SweepConfig.from_yaml(Path("/nonexistent/sweep.yaml"))


class TestSweepState:
    """Tests for SweepState model."""

    def test_initial_state(self):
        """Test creating initial state."""
        state = SweepState(
            sweep_name="test",
            sweep_config_dict={"metadata": {"name": "test"}},
        )
        assert state.sweep_name == "test"
        assert state.completed_runs == []
        assert state.current_run is None

    def test_mark_run_started(self):
        """Test marking a run as started."""
        state = SweepState(
            sweep_name="test",
            sweep_config_dict={},
        )
        state.mark_run_started("A", "test-sweep-A-123")

        assert state.current_run == "A"
        assert state.run_names["A"] == "test-sweep-A-123"

    def test_mark_run_completed(self):
        """Test marking a run as completed."""
        state = SweepState(
            sweep_name="test",
            sweep_config_dict={},
        )
        state.current_run = "A"
        state.mark_run_completed("A", {"final_reward": 5.0})

        assert "A" in state.completed_runs
        assert state.run_results["A"]["final_reward"] == 5.0
        assert state.current_run is None

    def test_is_run_completed(self):
        """Test checking if run is completed."""
        state = SweepState(
            sweep_name="test",
            sweep_config_dict={},
            completed_runs=["A", "B"],
        )
        assert state.is_run_completed("A") is True
        assert state.is_run_completed("B") is True
        assert state.is_run_completed("C") is False

    def test_get_pending_runs(self):
        """Test getting pending runs."""
        config = SweepConfig(
            metadata=SweepMetadata(name="test", experiment_tag_prefix="test"),
            runs={
                "A": RunConfig(name="a"),
                "B": RunConfig(name="b"),
                "C": RunConfig(name="c"),
            },
        )
        state = SweepState(
            sweep_name="test",
            sweep_config_dict=config.model_dump(),
            completed_runs=["A"],
        )
        pending = state.get_pending_runs()
        assert pending == ["B", "C"]

    def test_get_pending_runs_with_filter(self):
        """Test getting pending runs with filter."""
        config = SweepConfig(
            metadata=SweepMetadata(name="test", experiment_tag_prefix="test"),
            runs={
                "A": RunConfig(name="a"),
                "B": RunConfig(name="b"),
                "C": RunConfig(name="c"),
            },
        )
        state = SweepState(
            sweep_name="test",
            sweep_config_dict=config.model_dump(),
            completed_runs=["A"],
        )
        # Only want B and C, A already done
        pending = state.get_pending_runs(["A", "B"])
        assert pending == ["B"]

    def test_save_and_load(self):
        """Test saving and loading state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "state.json"

            state = SweepState(
                sweep_name="test",
                sweep_config_dict={"metadata": {"name": "test"}},
                completed_runs=["A"],
                run_results={"A": {"reward": 5.0}},
            )
            state.save(path)

            loaded = SweepState.load("test", path)
            assert loaded is not None
            assert loaded.sweep_name == "test"
            assert loaded.completed_runs == ["A"]
            assert loaded.run_results["A"]["reward"] == 5.0

    def test_load_missing_returns_none(self):
        """Test that loading missing state returns None."""
        result = SweepState.load("nonexistent", Path("/nonexistent/path.json"))
        assert result is None


class TestAnalysisConfig:
    """Tests for AnalysisConfig model."""

    def test_default_values(self):
        """Test default analysis config values."""
        config = AnalysisConfig()
        assert config.primary_metric == "trueskill/display_rating"
        assert len(config.secondary_metrics) > 0
        assert config.expected_ranking == []

    def test_custom_values(self):
        """Test custom analysis config."""
        config = AnalysisConfig(
            primary_metric="benchmark/reward_mean",
            secondary_metrics=["game/win_rate"],
            expected_ranking=["B", "A"],
            success_criteria=["B beats A"],
        )
        assert config.primary_metric == "benchmark/reward_mean"
        assert config.expected_ranking == ["B", "A"]


class TestExistingSweeConfig:
    """Test loading the actual sweep config in the repo."""

    def test_load_ablation_sweep(self):
        """Test loading the ablation sweep config."""
        path = Path("experiments/sweeps/longer-horizon-inverted-weight-ablation/")
        if not path.exists():
            pytest.skip("Ablation sweep not found")

        config = SweepConfig.from_yaml(path)

        assert config.metadata.name == "longer-horizon-inverted-weight-ablation"
        assert len(config.runs) == 4
        assert "A" in config.runs
        assert "D" in config.runs

        # Check interpolation worked
        assert config.runs["A"].config["experiment_tag"] == "ablation-scoring-A-baseline"

        # Check we can build experiment configs
        for run_id in config.runs:
            exp_cfg = config.build_experiment_config(run_id)
            assert exp_cfg.run_name.startswith("longer-horizon-inverted-weight-ablation")
