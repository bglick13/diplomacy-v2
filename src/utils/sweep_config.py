# src/utils/sweep_config.py
"""
Configuration models for sweep/ablation experiments.

This module provides:
- SweepConfig: YAML-based configuration for multi-run experiments
- SweepState: Persisted state for resumable sweeps
- Helper functions for loading and validating sweep configs
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator

from src.utils.config import ExperimentConfig


class SweepMetadata(BaseModel):
    """Metadata for a sweep experiment."""

    name: str = Field(..., description="Unique identifier for this sweep")
    description: str = Field(default="", description="Human-readable description of the sweep")
    hypothesis: str = Field(default="", description="The hypothesis being tested")
    experiment_tag_prefix: str = Field(
        ..., description="Prefix for WandB experiment tags (e.g., 'ablation-scoring')"
    )
    created: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d"),
        description="Creation date (YYYY-MM-DD)",
    )
    author: str = Field(default="", description="Author of the sweep")


class RunConfig(BaseModel):
    """Configuration for a single run within a sweep."""

    name: str = Field(..., description="Short name for this run (e.g., 'baseline', 'long-horizon')")
    description: str = Field(default="", description="Description of what this run tests")
    config: dict[str, Any] = Field(
        default_factory=dict,
        description="ExperimentConfig overrides for this run",
    )


class AnalysisConfig(BaseModel):
    """Configuration for post-sweep analysis."""

    primary_metric: str = Field(
        default="trueskill/display_rating",
        description="Primary metric to compare across runs",
    )
    secondary_metrics: list[str] = Field(
        default_factory=lambda: [
            "game/win_bonus_rate",
            "benchmark/reward_mean",
            "game/avg_sc_count",
        ],
        description="Additional metrics to track",
    )
    expected_ranking: list[str] = Field(
        default_factory=list,
        description="Expected ordering of runs by performance (e.g., ['D', 'C', 'B', 'A'])",
    )
    success_criteria: list[str] = Field(
        default_factory=list,
        description="Criteria for determining sweep success",
    )


class SweepConfig(BaseModel):
    """
    Configuration for a sweep experiment.

    A sweep is a collection of related training runs that test a hypothesis.
    Each run uses the same defaults but with specific config overrides.

    Example YAML:
        metadata:
          name: "horizon-ablation"
          experiment_tag_prefix: "horizon-sweep"

        defaults:
          total_steps: 100

        runs:
          A:
            name: "short-horizon"
            config:
              rollout_horizon_years: 4
          B:
            name: "long-horizon"
            config:
              rollout_horizon_years: 8
    """

    metadata: SweepMetadata
    defaults: dict[str, Any] = Field(
        default_factory=dict,
        description="Default ExperimentConfig values shared across all runs",
    )
    runs: dict[str, RunConfig] = Field(
        ..., description="Map of run IDs (e.g., 'A', 'B') to run configurations"
    )
    analysis: AnalysisConfig = Field(
        default_factory=AnalysisConfig,
        description="Configuration for post-sweep analysis",
    )

    @model_validator(mode="after")
    def validate_runs(self) -> SweepConfig:
        """Validate that runs are properly configured."""
        if not self.runs:
            raise ValueError("At least one run must be defined")
        return self

    @classmethod
    def from_yaml(cls, path: Path | str) -> SweepConfig:
        """
        Load sweep configuration from a YAML file.

        Args:
            path: Path to sweep.yaml file or directory containing it

        Returns:
            Parsed SweepConfig
        """
        path = Path(path)
        if path.is_dir():
            path = path / "sweep.yaml"

        if not path.exists():
            raise FileNotFoundError(f"Sweep config not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        # Resolve variable interpolation
        data = _resolve_interpolation(data)

        return cls(**data)

    def build_experiment_config(
        self, run_id: str, timestamp: str | None = None
    ) -> ExperimentConfig:
        """
        Build an ExperimentConfig for a specific run.

        Args:
            run_id: The run identifier (e.g., 'A', 'B')
            timestamp: Optional timestamp for run_name (defaults to current time)

        Returns:
            ExperimentConfig with merged defaults and run-specific overrides
        """
        if run_id not in self.runs:
            raise ValueError(f"Unknown run ID: {run_id}. Available: {list(self.runs.keys())}")

        run = self.runs[run_id]
        timestamp = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")

        # Merge: defaults <- run-specific config
        merged = {**self.defaults, **run.config}

        # Generate unique run_name if not explicitly set
        if "run_name" not in merged:
            merged["run_name"] = f"{self.metadata.name}-{run_id}-{timestamp}"

        # Ensure experiment_tag is set (for WandB grouping)
        if "experiment_tag" not in merged:
            merged["experiment_tag"] = f"{self.metadata.experiment_tag_prefix}-{run_id}"

        return ExperimentConfig(**merged)

    def get_run_ids(self) -> list[str]:
        """Get all run IDs in definition order."""
        return list(self.runs.keys())

    def describe(self) -> str:
        """Return a human-readable description of the sweep."""
        lines = [
            f"Sweep: {self.metadata.name}",
            f"  {self.metadata.description}",
            "",
            "Runs:",
        ]
        for run_id, run in self.runs.items():
            lines.append(f"  [{run_id}] {run.name}: {run.description}")
        return "\n".join(lines)


class SweepState(BaseModel):
    """
    Persisted state for resumable sweeps.

    Saved to /data/sweeps/<sweep_name>/state.json on Modal volume.
    """

    sweep_name: str = Field(..., description="Name of the sweep")
    sweep_config_dict: dict[str, Any] = Field(
        ..., description="Serialized SweepConfig for reference"
    )
    started_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="ISO timestamp when sweep started",
    )
    completed_runs: list[str] = Field(
        default_factory=list,
        description="List of completed run IDs (e.g., ['A', 'B'])",
    )
    run_results: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Results from completed runs keyed by run ID",
    )
    current_run: str | None = Field(
        default=None,
        description="Currently executing run ID (for crash recovery)",
    )
    run_names: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of run IDs to WandB run names",
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y%m%d-%H%M%S"),
        description="Shared timestamp for all runs in this sweep instance",
    )

    def is_run_completed(self, run_id: str) -> bool:
        """Check if a run has been completed."""
        return run_id in self.completed_runs

    def mark_run_started(self, run_id: str, run_name: str) -> None:
        """Mark a run as started."""
        self.current_run = run_id
        self.run_names[run_id] = run_name

    def mark_run_completed(self, run_id: str, result: dict[str, Any]) -> None:
        """Mark a run as completed with its result."""
        self.completed_runs.append(run_id)
        self.run_results[run_id] = result
        self.current_run = None

    def get_pending_runs(self, requested_runs: list[str] | None = None) -> list[str]:
        """Get list of runs that haven't been completed yet."""
        all_runs = requested_runs or list(SweepConfig(**self.sweep_config_dict).runs.keys())
        return [r for r in all_runs if r not in self.completed_runs]

    @classmethod
    def get_state_path(cls, sweep_name: str) -> Path:
        """Get the path for sweep state on Modal volume."""
        return Path(f"/data/sweeps/{sweep_name}/state.json")

    def save(self, path: Path | None = None) -> None:
        """Save state to disk."""
        path = path or self.get_state_path(self.sweep_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, sweep_name: str, path: Path | None = None) -> SweepState | None:
        """Load state from disk, or return None if not found."""
        path = path or cls.get_state_path(sweep_name)
        if not path.exists():
            return None
        with open(path) as f:
            return cls.model_validate_json(f.read())

    @classmethod
    def load_or_create(
        cls,
        sweep_config: SweepConfig,
        path: Path | None = None,
    ) -> SweepState:
        """Load existing state or create new state for a sweep."""
        existing = cls.load(sweep_config.metadata.name, path)
        if existing:
            return existing

        return cls(
            sweep_name=sweep_config.metadata.name,
            sweep_config_dict=sweep_config.model_dump(),
        )


def _resolve_interpolation(data: Any, root: dict | None = None) -> Any:
    """
    Resolve ${...} variable interpolation in YAML data.

    Supports:
        ${metadata.name} -> data["metadata"]["name"]
        ${metadata.experiment_tag_prefix} -> data["metadata"]["experiment_tag_prefix"]

    Args:
        data: The data structure to process
        root: The root data structure for variable resolution

    Returns:
        Data with variables resolved
    """
    if root is None:
        root = data

    if isinstance(data, dict):
        return {k: _resolve_interpolation(v, root) for k, v in data.items()}
    elif isinstance(data, list):
        return [_resolve_interpolation(v, root) for v in data]
    elif isinstance(data, str):
        # Find ${...} patterns
        pattern = r"\$\{([^}]+)\}"
        matches = re.findall(pattern, data)
        for match in matches:
            # Resolve dotted path (e.g., "metadata.name")
            value = root
            for key in match.split("."):
                if isinstance(value, dict):
                    value = value.get(key)
                else:
                    value = None
                    break
            if value is not None:
                # Replace in string
                data = data.replace(f"${{{match}}}", str(value))
        return data
    return data
