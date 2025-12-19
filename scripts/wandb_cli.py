#!/usr/bin/env python3
"""
WandB CLI for experiment analysis and hypothesis generation.

This CLI provides programmatic access to WandB experiment data for automated
analysis and hypothesis generation by coding agents.

Usage:
    # List runs with filters
    wandb-cli list --project diplomacy-grpo --tag power-laws --limit 10

    # Get run details (config, summary metrics)
    wandb-cli get-run --run-id <run_id_or_name> --project diplomacy-grpo

    # Get metrics over time
    wandb-cli get-metrics --run-id <run_id_or_name> --project diplomacy-grpo --metrics elo/challenger benchmark/loss

    # Get all metrics for a run
    wandb-cli get-metrics --run-id <run_id_or_name> --project diplomacy-grpo --all-metrics

    # Get artifacts
    wandb-cli get-artifacts --run-id <run_id_or_name> --project diplomacy-grpo

    # Compare multiple runs
    wandb-cli compare --run-ids <id1> <id2> <id3> --project diplomacy-grpo --metrics elo/challenger benchmark/loss

    # Export run data as JSON (for programmatic use)
    wandb-cli export --run-id <run_id_or_name> --project diplomacy-grpo --output run_data.json

    # Search runs by config values
    wandb-cli search --project diplomacy-grpo --config "lora_rank=16" --config "base_model_id=Qwen/Qwen2.5-7B-Instruct"
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import click
import wandb


def format_datetime(dt: datetime | str | None) -> str | None:
    """Format datetime for JSON serialization."""
    if dt is None:
        return None
    if isinstance(dt, str):
        return dt
    return dt.isoformat()


def resolve_run(
    api: wandb.Api,
    run_identifier: str,
    project: str,
    entity: str | None = None,
) -> Any:
    """
    Resolve a run by ID or display name.

    Args:
        api: WandB API instance
        run_identifier: Run ID (8-char) or display name
        project: WandB project name
        entity: WandB entity (team/username)

    Returns:
        WandB Run object

    Raises:
        click.ClickException: If run cannot be found
    """
    project_path = f"{entity}/{project}" if entity else project

    # First, try direct lookup by run ID
    try:
        run_path = (
            f"{entity}/{project}/{run_identifier}" if entity else f"{project}/{run_identifier}"
        )
        run = api.run(run_path)
        # Verify it's a valid run (not "not found")
        _ = run.state  # This will raise if run doesn't exist
        return run
    except Exception:
        pass  # Run ID lookup failed, try by display name

    # Search by display name
    try:
        runs = api.runs(
            project_path,
            filters={"displayName": run_identifier},
            per_page=1,
        )
        runs_list = list(runs)
        if runs_list:
            return runs_list[0]
    except Exception:
        pass

    # Also try filtering by name field (some WandB versions use this)
    try:
        runs = api.runs(
            project_path,
            filters={"display_name": run_identifier},
            per_page=1,
        )
        runs_list = list(runs)
        if runs_list:
            return runs_list[0]
    except Exception:
        pass

    # Try with $eq operator
    try:
        runs = api.runs(
            project_path,
            filters={"displayName": {"$eq": run_identifier}},
            per_page=1,
        )
        runs_list = list(runs)
        if runs_list:
            return runs_list[0]
    except Exception:
        pass

    raise click.ClickException(
        f"Could not find run '{run_identifier}' in project '{project_path}'. "
        f"Tried lookup by run ID and display name."
    )


# Click group for CLI
@click.group()
@click.version_option(version="1.0.0", prog_name="wandb-cli")
def cli() -> None:
    """WandB CLI for experiment analysis and hypothesis generation."""
    pass


@cli.command("list")
@click.option(
    "--project",
    "-p",
    default="diplomacy-grpo",
    help="WandB project name",
    show_default=True,
)
@click.option("--entity", "-e", default=None, help="WandB entity (team/username)")
@click.option(
    "--tag", "-t", "tags", multiple=True, help="Filter by tags (can be used multiple times)"
)
@click.option("--state", "-s", default=None, help="Filter by state (running, finished, crashed)")
@click.option("--limit", "-l", default=50, help="Maximum runs to return", show_default=True)
@click.option(
    "--output-format",
    "-o",
    type=click.Choice(["json", "table"]),
    default="json",
    help="Output format",
    show_default=True,
)
def list_runs_cmd(
    project: str,
    entity: str | None,
    tags: tuple[str, ...],
    state: str | None,
    limit: int,
    output_format: str,
) -> None:
    """List runs matching filters."""
    result = list_runs(
        project=project,
        entity=entity,
        tags=list(tags) if tags else None,
        state=state,
        limit=limit,
        output_format=output_format,
    )
    if output_format == "json":
        click.echo(json.dumps(result, indent=2, default=str))


@cli.command("get-run")
@click.option(
    "--run-id",
    "-r",
    required=True,
    help="WandB run ID or display name",
)
@click.option(
    "--project",
    "-p",
    default="diplomacy-grpo",
    help="WandB project name",
    show_default=True,
)
@click.option("--entity", "-e", default=None, help="WandB entity (team/username)")
@click.option("--include-history", "-h", is_flag=True, help="Include full metric history")
@click.option("--include-artifacts", "-a", is_flag=True, help="Include artifact information")
@click.option(
    "--output-format",
    "-o",
    type=click.Choice(["json", "table"]),
    default="json",
    help="Output format",
    show_default=True,
)
def get_run_cmd(
    run_id: str,
    project: str,
    entity: str | None,
    include_history: bool,
    include_artifacts: bool,
    output_format: str,
) -> None:
    """Get detailed information about a specific run."""
    result = get_run(
        run_id=run_id,
        project=project,
        entity=entity,
        include_history=include_history,
        include_artifacts=include_artifacts,
        output_format=output_format,
    )
    if output_format == "json":
        click.echo(json.dumps(result, indent=2, default=str))


@cli.command("get-metrics")
@click.option(
    "--run-id",
    "-r",
    required=True,
    help="WandB run ID or display name",
)
@click.option(
    "--project",
    "-p",
    default="diplomacy-grpo",
    help="WandB project name",
    show_default=True,
)
@click.option("--entity", "-e", default=None, help="WandB entity (team/username)")
@click.option("--metrics", "-m", multiple=True, help="Specific metrics to retrieve")
@click.option("--all-metrics", is_flag=True, help="Retrieve all metrics")
@click.option(
    "--x-axis",
    type=click.Choice(["_step", "_timestamp", "_runtime"]),
    default="_step",
    help="X-axis for metrics",
    show_default=True,
)
@click.option(
    "--output-format",
    "-o",
    type=click.Choice(["json", "table"]),
    default="json",
    help="Output format",
    show_default=True,
)
def get_metrics_cmd(
    run_id: str,
    project: str,
    entity: str | None,
    metrics: tuple[str, ...],
    all_metrics: bool,
    x_axis: str,
    output_format: str,
) -> None:
    """Get metrics over time for a run."""
    result = get_metrics(
        run_id=run_id,
        project=project,
        entity=entity,
        metrics=list(metrics) if metrics else None,
        all_metrics=all_metrics,
        x_axis=x_axis,
        output_format=output_format,
    )
    if output_format == "json":
        click.echo(json.dumps(result, indent=2, default=str))


@cli.command("get-artifacts")
@click.option(
    "--run-id",
    "-r",
    required=True,
    help="WandB run ID or display name",
)
@click.option(
    "--project",
    "-p",
    default="diplomacy-grpo",
    help="WandB project name",
    show_default=True,
)
@click.option("--entity", "-e", default=None, help="WandB entity (team/username)")
@click.option("--artifact-type", default=None, help="Filter by artifact type")
@click.option(
    "--output-format",
    "-o",
    type=click.Choice(["json", "table"]),
    default="json",
    help="Output format",
    show_default=True,
)
def get_artifacts_cmd(
    run_id: str,
    project: str,
    entity: str | None,
    artifact_type: str | None,
    output_format: str,
) -> None:
    """Get artifacts for a run."""
    result = get_artifacts(
        run_id=run_id,
        project=project,
        entity=entity,
        artifact_type=artifact_type,
        output_format=output_format,
    )
    if output_format == "json":
        click.echo(json.dumps(result, indent=2, default=str))


@cli.command("compare")
@click.option(
    "--run-ids",
    "-r",
    multiple=True,
    required=True,
    help="Run IDs or display names to compare",
)
@click.option(
    "--project",
    "-p",
    default="diplomacy-grpo",
    help="WandB project name",
    show_default=True,
)
@click.option("--entity", "-e", default=None, help="WandB entity (team/username)")
@click.option("--metrics", "-m", multiple=True, help="Metrics to compare")
@click.option(
    "--output-format",
    "-o",
    type=click.Choice(["json", "table"]),
    default="json",
    help="Output format",
    show_default=True,
)
def compare_cmd(
    run_ids: tuple[str, ...],
    project: str,
    entity: str | None,
    metrics: tuple[str, ...],
    output_format: str,
) -> None:
    """Compare multiple runs across specified metrics."""
    result = compare_runs(
        run_ids=list(run_ids),
        project=project,
        entity=entity,
        metrics=list(metrics) if metrics else None,
        output_format=output_format,
    )
    if output_format == "json":
        click.echo(json.dumps(result, indent=2, default=str))


@cli.command("search")
@click.option(
    "--project",
    "-p",
    default="diplomacy-grpo",
    help="WandB project name",
    show_default=True,
)
@click.option("--entity", "-e", default=None, help="WandB entity (team/username)")
@click.option(
    "--config",
    "-c",
    "config_filters",
    multiple=True,
    help="Config filter (key=value)",
)
@click.option("--tag", "-t", "tag_filters", multiple=True, help="Tag filter")
@click.option("--limit", "-l", default=50, help="Maximum runs to return", show_default=True)
@click.option(
    "--output-format",
    "-o",
    type=click.Choice(["json", "table"]),
    default="json",
    help="Output format",
    show_default=True,
)
def search_cmd(
    project: str,
    entity: str | None,
    config_filters: tuple[str, ...],
    tag_filters: tuple[str, ...],
    limit: int,
    output_format: str,
) -> None:
    """Search runs by config values and tags."""
    result = search_runs(
        project=project,
        entity=entity,
        config_filters=list(config_filters) if config_filters else None,
        tag_filters=list(tag_filters) if tag_filters else None,
        limit=limit,
        output_format=output_format,
    )
    if output_format == "json":
        click.echo(json.dumps(result, indent=2, default=str))


@cli.command("export")
@click.option(
    "--run-id",
    "-r",
    required=True,
    help="WandB run ID or display name",
)
@click.option(
    "--project",
    "-p",
    default="diplomacy-grpo",
    help="WandB project name",
    show_default=True,
)
@click.option("--entity", "-e", default=None, help="WandB entity (team/username)")
@click.option(
    "--output",
    "-o",
    default="run_export.json",
    help="Output file path",
    show_default=True,
)
@click.option("--no-history", is_flag=True, help="Exclude metric history")
@click.option("--no-artifacts", is_flag=True, help="Exclude artifacts")
def export_cmd(
    run_id: str,
    project: str,
    entity: str | None,
    output: str,
    no_history: bool,
    no_artifacts: bool,
) -> None:
    """Export complete run data to JSON file."""
    result = export_run(
        run_id=run_id,
        project=project,
        entity=entity,
        output_path=output,
        include_history=not no_history,
        include_artifacts=not no_artifacts,
    )
    click.echo(json.dumps(result, indent=2, default=str))


@cli.command("get-cache-stats")
@click.option(
    "--run-id",
    "-r",
    required=True,
    help="WandB run ID or display name",
)
@click.option(
    "--project",
    "-p",
    default="diplomacy-grpo",
    help="WandB project name",
    show_default=True,
)
@click.option("--entity", "-e", default=None, help="WandB entity (team/username)")
@click.option(
    "--output-format",
    "-o",
    type=click.Choice(["json", "table"]),
    default="json",
    help="Output format",
    show_default=True,
)
def get_cache_stats_cmd(
    run_id: str,
    project: str,
    entity: str | None,
    output_format: str,
) -> None:
    """Get prefix cache statistics for a run."""
    result = get_cache_stats(
        run_id=run_id,
        project=project,
        entity=entity,
        output_format=output_format,
    )
    if output_format == "json":
        click.echo(json.dumps(result, indent=2, default=str))


# ============================================================================
# Core functions (unchanged logic, but now use resolve_run for lookups)
# ============================================================================


def list_runs(
    project: str,
    entity: str | None = None,
    tags: list[str] | None = None,
    state: str | None = None,
    limit: int = 50,
    output_format: str = "table",
) -> dict[str, Any]:
    """
    List runs matching filters.

    Args:
        project: WandB project name
        entity: WandB entity (team/username), defaults to current user
        tags: List of tags to filter by
        state: Run state filter (running, finished, crashed, etc.)
        limit: Maximum number of runs to return
        output_format: Output format (table, json)

    Returns:
        Dictionary with runs data
    """
    api = wandb.Api()
    filters = {}
    if tags:
        filters["tags"] = {"$in": tags}
    if state:
        filters["state"] = state

    runs = api.runs(
        f"{entity}/{project}" if entity else project,
        filters=filters if filters else None,
        per_page=limit,
    )

    runs_data = []
    for run in runs:
        runs_data.append(
            {
                "id": run.id,
                "name": run.name,
                "state": run.state,
                "created_at": format_datetime(getattr(run, "created_at", None)),
                "updated_at": format_datetime(getattr(run, "updated_at", None)),
                "tags": getattr(run, "tags", []),
                "config": dict(run.config) if run.config else {},
                "summary": dict(run.summary) if run.summary else {},
                "url": run.url,
            }
        )

    result = {"count": len(runs_data), "runs": runs_data}

    if output_format == "json":
        return result

    # Print table format
    click.echo(f"\nFound {len(runs_data)} runs in project '{project}':\n")
    click.echo(f"{'Run ID':<12} {'Name':<40} {'State':<12} {'Created':<20} {'Tags':<30}")
    click.echo("-" * 120)
    for run in runs_data:
        tags_str = ", ".join(run["tags"][:3]) if run["tags"] else ""
        if len(run["tags"]) > 3:
            tags_str += "..."
        created = run["created_at"][:19] if run["created_at"] else "N/A"
        click.echo(
            f"{run['id']:<12} {run['name']:<40} {run['state']:<12} {created:<20} {tags_str:<30}"
        )
    click.echo()

    return result


def get_run(
    run_id: str,
    project: str,
    entity: str | None = None,
    include_history: bool = False,
    include_artifacts: bool = False,
    output_format: str = "json",
) -> dict[str, Any]:
    """
    Get detailed information about a specific run.

    Args:
        run_id: WandB run ID or display name
        project: WandB project name
        entity: WandB entity (team/username)
        include_history: Whether to include full metric history
        include_artifacts: Whether to include artifact information
        output_format: Output format (table, json)

    Returns:
        Dictionary with run details
    """
    api = wandb.Api()
    run = resolve_run(api, run_id, project, entity)

    result: dict[str, Any] = {
        "id": run.id,
        "name": run.name,
        "state": run.state,
        "created_at": format_datetime(getattr(run, "created_at", None)),
        "updated_at": format_datetime(getattr(run, "updated_at", None)),
        "finished_at": format_datetime(getattr(run, "finished_at", None)),
        "tags": getattr(run, "tags", []),
        "config": dict(run.config) if run.config else {},
        "summary": dict(run.summary) if run.summary else {},
        "url": run.url,
        "group": getattr(run, "group", None),
        "job_type": getattr(run, "job_type", None),
    }

    if include_history:
        history = run.history()
        result["history"] = history.to_dict("records") if not history.empty else []

    if include_artifacts:
        artifacts = []
        for artifact in run.logged_artifacts():
            artifacts.append(
                {
                    "name": artifact.name,
                    "type": artifact.type,
                    "version": artifact.version,
                    "aliases": artifact.aliases,
                    "size": artifact.size,
                    "created_at": format_datetime(artifact.created_at),
                }
            )
        result["artifacts"] = artifacts

    if output_format == "json":
        return result

    # Print formatted output
    click.echo(f"\nRun: {run.name} ({run.id})")
    click.echo(f"State: {run.state}")
    click.echo(f"URL: {run.url}")
    click.echo("\nConfig:")
    for key, value in run.config.items():
        click.echo(f"  {key}: {value}")

    click.echo("\nSummary Metrics:")
    for key, value in run.summary.items():
        click.echo(f"  {key}: {value}")

    if include_artifacts and result.get("artifacts"):
        click.echo(f"\nArtifacts ({len(result['artifacts'])}):")
        for artifact in result["artifacts"]:
            click.echo(f"  - {artifact['name']} ({artifact['type']})")

    return result


def get_metrics(
    run_id: str,
    project: str,
    entity: str | None = None,
    metrics: list[str] | None = None,
    all_metrics: bool = False,
    x_axis: str = "_step",
    output_format: str = "json",
) -> dict[str, Any]:
    """
    Get metrics over time for a run.

    Args:
        run_id: WandB run ID or display name
        project: WandB project name
        entity: WandB entity (team/username)
        metrics: List of metric names to retrieve (e.g., ['elo/challenger', 'benchmark/loss'])
        all_metrics: If True, retrieve all metrics
        x_axis: X-axis for metrics (default: '_step', can be '_timestamp' or '_runtime')
        output_format: Output format (table, json)

    Returns:
        Dictionary with metrics data
    """
    api = wandb.Api()
    run = resolve_run(api, run_id, project, entity)

    # Get history
    history = run.history(x_axis=x_axis)

    if history.empty:
        return {"run_id": run.id, "run_name": run.name, "metrics": {}, "data_points": 0}

    # Get available metrics
    available_metrics = [col for col in history.columns if not col.startswith("_")]

    if all_metrics:
        metrics_to_get = available_metrics
    elif metrics:
        metrics_to_get = [m for m in metrics if m in available_metrics]
        missing = [m for m in metrics if m not in available_metrics]
        if missing:
            click.echo(f"Warning: Metrics not found: {missing}", err=True)
    else:
        metrics_to_get = available_metrics[:10]  # Default to first 10

    result: dict[str, Any] = {
        "run_id": run.id,
        "run_name": run.name,
        "x_axis": x_axis,
        "data_points": len(history),
        "metrics": {},
    }

    for metric in metrics_to_get:
        metric_data = history[metric].dropna().tolist()
        x_data = history[x_axis].dropna().tolist()
        # Align x and y data
        aligned_data = [
            {x_axis: x, "value": y}
            for x, y in zip(x_data, metric_data, strict=False)
            if not (isinstance(x, float) and (x != x)) and not (isinstance(y, float) and (y != y))
        ]

        result["metrics"][metric] = {
            "data_points": len(aligned_data),
            "values": aligned_data,
            "min": float(history[metric].min()) if not history[metric].empty else None,
            "max": float(history[metric].max()) if not history[metric].empty else None,
            "mean": float(history[metric].mean()) if not history[metric].empty else None,
            "std": float(history[metric].std()) if not history[metric].empty else None,
        }

    if output_format == "json":
        return result

    # Print formatted output
    click.echo(f"\nMetrics for run: {run.name} ({run.id})")
    click.echo(f"Data points: {len(history)}")
    click.echo(f"X-axis: {x_axis}\n")

    for metric_name, metric_info in result["metrics"].items():
        click.echo(f"{metric_name}:")
        click.echo(f"  Data points: {metric_info['data_points']}")
        if metric_info["min"] is not None:
            click.echo(f"  Min: {metric_info['min']:.4f}")
            click.echo(f"  Max: {metric_info['max']:.4f}")
            click.echo(f"  Mean: {metric_info['mean']:.4f}")
            click.echo(f"  Std: {metric_info['std']:.4f}")
        click.echo()

    return result


def get_artifacts(
    run_id: str,
    project: str,
    entity: str | None = None,
    artifact_type: str | None = None,
    output_format: str = "json",
) -> dict[str, Any]:
    """
    Get artifacts for a run.

    Args:
        run_id: WandB run ID or display name
        project: WandB project name
        entity: WandB entity (team/username)
        artifact_type: Filter by artifact type
        output_format: Output format (table, json)

    Returns:
        Dictionary with artifacts data
    """
    api = wandb.Api()
    run = resolve_run(api, run_id, project, entity)

    artifacts = []
    for artifact in run.logged_artifacts():
        if artifact_type and artifact.type != artifact_type:
            continue

        artifact_info: dict[str, Any] = {
            "name": artifact.name,
            "type": artifact.type,
            "version": artifact.version,
            "aliases": list(artifact.aliases),
            "size": artifact.size,
            "created_at": format_datetime(artifact.created_at),
            "updated_at": format_datetime(artifact.updated_at),
        }

        # Get artifact files if available
        try:
            files = []
            for file in artifact.files():
                files.append(
                    {
                        "name": file.name,
                        "size": file.size,
                        "url": file.url,
                    }
                )
            artifact_info["files"] = files
        except Exception:
            artifact_info["files"] = []

        artifacts.append(artifact_info)

    result = {"run_id": run.id, "run_name": run.name, "artifacts": artifacts}

    if output_format == "json":
        return result

    # Print formatted output
    click.echo(f"\nArtifacts for run: {run.name} ({run.id})")
    click.echo(f"Total artifacts: {len(artifacts)}\n")

    for artifact in artifacts:
        click.echo(f"Name: {artifact['name']}")
        click.echo(f"Type: {artifact['type']}")
        click.echo(f"Version: {artifact['version']}")
        click.echo(f"Size: {artifact['size']} bytes")
        click.echo(f"Files: {len(artifact['files'])}")
        if artifact["files"]:
            click.echo("  Files:")
            for file in artifact["files"][:5]:  # Show first 5 files
                click.echo(f"    - {file['name']} ({file['size']} bytes)")
            if len(artifact["files"]) > 5:
                click.echo(f"    ... and {len(artifact['files']) - 5} more")
        click.echo()

    return result


def compare_runs(
    run_ids: list[str],
    project: str,
    entity: str | None = None,
    metrics: list[str] | None = None,
    output_format: str = "json",
) -> dict[str, Any]:
    """
    Compare multiple runs across specified metrics.

    Args:
        run_ids: List of run IDs or display names to compare
        project: WandB project name
        entity: WandB entity (team/username)
        metrics: List of metric names to compare
        output_format: Output format (table, json)

    Returns:
        Dictionary with comparison data
    """
    api = wandb.Api()
    runs_data = []

    for run_id in run_ids:
        try:
            run = resolve_run(api, run_id, project, entity)
            runs_data.append(
                {
                    "id": run.id,
                    "name": run.name,
                    "config": dict(run.config) if run.config else {},
                    "summary": dict(run.summary) if run.summary else {},
                    "state": run.state,
                }
            )
        except click.ClickException as e:
            click.echo(f"Warning: {e.message}", err=True)

    if not runs_data:
        return {"error": "No valid runs found"}

    result: dict[str, Any] = {
        "runs": runs_data,
        "comparison": {},
    }

    # Compare summary metrics
    if metrics:
        for metric in metrics:
            comparison = []
            for run_data in runs_data:
                value = run_data["summary"].get(metric)
                comparison.append(
                    {
                        "run_id": run_data["id"],
                        "run_name": run_data["name"],
                        "value": value,
                    }
                )
            result["comparison"][metric] = comparison

    if output_format == "json":
        return result

    # Print formatted comparison
    click.echo(f"\nComparing {len(runs_data)} runs:\n")
    for run_data in runs_data:
        click.echo(f"{run_data['name']} ({run_data['id']})")
        click.echo(f"  State: {run_data['state']}")

    if result["comparison"]:
        click.echo("\nMetric Comparison:")
        for metric, values in result["comparison"].items():
            click.echo(f"\n{metric}:")
            for item in values:
                click.echo(f"  {item['run_name']}: {item['value']}")

    return result


def search_runs(
    project: str,
    entity: str | None = None,
    config_filters: list[str] | None = None,
    tag_filters: list[str] | None = None,
    limit: int = 50,
    output_format: str = "json",
) -> dict[str, Any]:
    """
    Search runs by config values and tags.

    Args:
        project: WandB project name
        entity: WandB entity (team/username)
        config_filters: List of config filters in format "key=value"
        tag_filters: List of tags to filter by
        limit: Maximum number of runs to return
        output_format: Output format (table, json)

    Returns:
        Dictionary with matching runs
    """
    api = wandb.Api()

    filters: dict[str, Any] = {}
    if tag_filters:
        filters["tags"] = {"$in": tag_filters}

    if config_filters:
        config_dict = {}
        for filter_str in config_filters:
            if "=" in filter_str:
                key, value = filter_str.split("=", 1)
                # Try to parse value as appropriate type
                try:
                    if value.lower() == "true":
                        value = True
                    elif value.lower() == "false":
                        value = False
                    elif value.isdigit():
                        value = int(value)
                    else:
                        try:
                            value = float(value)
                        except ValueError:
                            pass  # Keep as string
                except Exception:
                    pass
                config_dict[key] = value
        if config_dict:
            filters["config"] = config_dict

    runs = api.runs(
        f"{entity}/{project}" if entity else project,
        filters=filters if filters else None,
        per_page=limit,
    )

    runs_data = []
    for run in runs:
        runs_data.append(
            {
                "id": run.id,
                "name": run.name,
                "state": run.state,
                "config": dict(run.config) if run.config else {},
                "summary": dict(run.summary) if run.summary else {},
                "tags": getattr(run, "tags", []),
                "url": run.url,
            }
        )

    result = {"count": len(runs_data), "runs": runs_data}

    if output_format == "json":
        return result

    # Print formatted output
    click.echo(f"\nFound {len(runs_data)} matching runs:\n")
    for run in runs_data:
        click.echo(f"{run['name']} ({run['id']})")
        click.echo(f"  State: {run['state']}")
        click.echo(f"  URL: {run['url']}")
        click.echo()

    return result


def export_run(
    run_id: str,
    project: str,
    entity: str | None = None,
    output_path: str = "run_export.json",
    include_history: bool = True,
    include_artifacts: bool = True,
) -> dict[str, Any]:
    """
    Export complete run data to JSON file.

    Args:
        run_id: WandB run ID or display name
        project: WandB project name
        entity: WandB entity (team/username)
        output_path: Path to output JSON file
        include_history: Whether to include metric history
        include_artifacts: Whether to include artifact information

    Returns:
        Dictionary with export status
    """
    run_data = get_run(
        run_id=run_id,
        project=project,
        entity=entity,
        include_history=include_history,
        include_artifacts=include_artifacts,
        output_format="json",
    )

    # Add metrics if history was included
    if include_history:
        metrics_data = get_metrics(
            run_id=run_data["id"],  # Use resolved ID
            project=project,
            entity=entity,
            all_metrics=True,
            output_format="json",
        )
        run_data["metrics"] = metrics_data.get("metrics", {})

    # Write to file
    output_file = Path(output_path)
    with output_file.open("w") as f:
        json.dump(run_data, f, indent=2, default=str)

    return {
        "status": "success",
        "run_id": run_data["id"],
        "run_name": run_data["name"],
        "output_path": str(output_file.absolute()),
        "file_size": output_file.stat().st_size,
    }


def get_cache_stats(
    run_id: str,
    project: str,
    entity: str | None = None,
    output_format: str = "json",
) -> dict[str, Any]:
    """
    Get prefix cache statistics for a run.

    Args:
        run_id: WandB run ID or display name
        project: WandB project name
        entity: WandB entity (team/username)
        output_format: Output format (table, json)

    Returns:
        Dictionary with cache statistics
    """
    api = wandb.Api()
    run = resolve_run(api, run_id, project, entity)

    # Get history for cache metrics
    history = run.history(x_axis="_step")

    # Cache metrics to look for
    cache_metrics = [
        "cache/hit_rate",
        "cache/total_queries",
        "cache/total_hits",
        "cache/prompt_tokens",
        "cache/batches",
    ]

    result: dict[str, Any] = {
        "run_id": run.id,
        "run_name": run.name,
        "cache_enabled": False,
        "metrics": {},
        "summary": {},
    }

    # Check if cache metrics exist
    available_cols = [col for col in history.columns if col.startswith("cache/")]
    if not available_cols:
        result["cache_enabled"] = False
        result["message"] = "No cache metrics found. Run may not have prefix caching enabled."
        if output_format == "table":
            click.echo(f"\n⚠️ No cache metrics found for run: {run.name}")
            click.echo("   Run may not have prefix caching enabled (prefix_cache_optimized=False)")
        return result

    result["cache_enabled"] = True

    # Extract cache metrics
    for metric in cache_metrics:
        if metric in history.columns:
            values = history[metric].dropna().tolist()
            if values:
                result["metrics"][metric] = {
                    "data_points": len(values),
                    "first": values[0] if values else None,
                    "last": values[-1] if values else None,
                    "min": float(min(values)),
                    "max": float(max(values)),
                    "mean": float(sum(values) / len(values)),
                }

    # Calculate summary statistics
    if "cache/hit_rate" in result["metrics"]:
        hit_rate_data = result["metrics"]["cache/hit_rate"]
        result["summary"]["avg_hit_rate"] = hit_rate_data["mean"]
        result["summary"]["final_hit_rate"] = hit_rate_data["last"]
        result["summary"]["peak_hit_rate"] = hit_rate_data["max"]

    if "cache/prompt_tokens" in result["metrics"]:
        tokens_data = result["metrics"]["cache/prompt_tokens"]
        result["summary"]["total_prompt_tokens"] = tokens_data["last"]

    if "cache/total_hits" in result["metrics"] and "cache/total_queries" in result["metrics"]:
        hits = result["metrics"]["cache/total_hits"]["last"]
        queries = result["metrics"]["cache/total_queries"]["last"]
        if queries and queries > 0:
            result["summary"]["overall_hit_rate"] = hits / queries
            result["summary"]["total_cache_hits"] = hits
            result["summary"]["total_cache_queries"] = queries

    # Estimate tokens saved (rough calculation)
    if "cache/hit_rate" in result["metrics"] and "cache/prompt_tokens" in result["metrics"]:
        avg_hit_rate = result["metrics"]["cache/hit_rate"]["mean"]
        total_tokens = result["metrics"]["cache/prompt_tokens"]["last"]
        if total_tokens:
            result["summary"]["estimated_tokens_saved"] = int(total_tokens * avg_hit_rate)

    if output_format == "table":
        click.echo(f"\n📊 Prefix Cache Stats for: {run.name} ({run.id})")
        click.echo("-" * 60)

        if result["summary"]:
            click.echo("\nSummary:")
            if "avg_hit_rate" in result["summary"]:
                click.echo(f"  Average Hit Rate: {result['summary']['avg_hit_rate'] * 100:.1f}%")
            if "final_hit_rate" in result["summary"]:
                click.echo(f"  Final Hit Rate: {result['summary']['final_hit_rate'] * 100:.1f}%")
            if "peak_hit_rate" in result["summary"]:
                click.echo(f"  Peak Hit Rate: {result['summary']['peak_hit_rate'] * 100:.1f}%")
            if "total_cache_hits" in result["summary"]:
                click.echo(f"  Total Cache Hits: {result['summary']['total_cache_hits']:,}")
            if "estimated_tokens_saved" in result["summary"]:
                click.echo(f"  Est. Tokens Saved: {result['summary']['estimated_tokens_saved']:,}")

        click.echo("\nMetrics over time:")
        for metric_name, metric_data in result["metrics"].items():
            click.echo(f"\n  {metric_name}:")
            click.echo(f"    Data points: {metric_data['data_points']}")
            click.echo(f"    First → Last: {metric_data['first']:.4f} → {metric_data['last']:.4f}")
            if "hit_rate" in metric_name:
                click.echo(f"    Mean: {metric_data['mean'] * 100:.1f}%")
            else:
                click.echo(f"    Mean: {metric_data['mean']:.2f}")

    return result


@cli.command("get-traces")
@click.option(
    "--project",
    "-p",
    default="diplomacy-grpo",
    help="Weave project name",
    show_default=True,
)
@click.option(
    "--op-name",
    default="log_trajectory",
    help="Operation name to filter by",
    show_default=True,
)
@click.option("--limit", "-l", default=10, help="Maximum traces to return", show_default=True)
@click.option("--power", default=None, help="Filter by power (e.g., FRANCE)")
@click.option("--run-name", "-r", default=None, help="Filter by training run name")
@click.option(
    "--extraction-status",
    type=click.Choice(["full", "partial", "empty"]),
    default=None,
    help="Filter by extraction status",
)
@click.option("--min-reward", type=float, default=None, help="Minimum reward filter")
@click.option("--max-reward", type=float, default=None, help="Maximum reward filter")
@click.option(
    "--output-format",
    "-o",
    type=click.Choice(["json", "table"]),
    default="json",
    help="Output format",
    show_default=True,
)
def get_traces_cmd(
    project: str,
    op_name: str,
    limit: int,
    power: str | None,
    run_name: str | None,
    extraction_status: str | None,
    min_reward: float | None,
    max_reward: float | None,
    output_format: str,
) -> None:
    """Get trajectory traces from Weave."""
    result = get_traces(
        project=project,
        op_name=op_name,
        limit=limit,
        power=power,
        run_name=run_name,
        extraction_status=extraction_status,
        min_reward=min_reward,
        max_reward=max_reward,
        output_format=output_format,
    )
    if output_format == "json":
        click.echo(json.dumps(result, indent=2, default=str))


def get_traces(
    project: str,
    op_name: str = "log_trajectory",
    limit: int = 10,
    power: str | None = None,
    run_name: str | None = None,
    extraction_status: str | None = None,
    min_reward: float | None = None,
    max_reward: float | None = None,
    output_format: str = "json",
) -> dict[str, Any]:
    """
    Get trajectory traces from Weave.

    Args:
        project: Weave project name
        op_name: Operation name to filter by (default: log_trajectory)
        limit: Maximum number of traces to return
        power: Filter by power (e.g., FRANCE)
        run_name: Filter by training run name
        extraction_status: Filter by extraction status (full, partial, empty)
        min_reward: Minimum reward filter
        max_reward: Maximum reward filter
        output_format: Output format (json, table)

    Returns:
        Dictionary with traces data
    """
    try:
        import weave
    except ImportError:
        return {"error": "weave package not installed. Run: pip install weave"}

    try:
        # Initialize Weave client
        client = weave.init(project)

        # Get calls for the specified operation
        # Note: Weave API may vary across versions - this uses a defensive approach
        calls_result = client.get_calls(limit=limit)  # type: ignore[call-arg]
        calls = [c for c in calls_result if op_name in str(getattr(c, "op_name", ""))][:limit]

        traces = []
        for call in calls:
            # Extract input data from the call
            inputs = call.inputs if hasattr(call, "inputs") else {}
            trajectory = inputs.get("trajectory", {})

            # Apply filters
            if power and trajectory.get("power") != power:
                continue
            if run_name and trajectory.get("run_name") != run_name:
                continue
            if extraction_status and trajectory.get("extraction_status") != extraction_status:
                continue

            reward = trajectory.get("reward", 0)
            if min_reward is not None and reward < min_reward:
                continue
            if max_reward is not None and reward > max_reward:
                continue

            trace_data = {
                "id": str(call.id) if hasattr(call, "id") else None,
                "created_at": str(call.started_at) if hasattr(call, "started_at") else None,
                "trajectory": trajectory,
            }
            traces.append(trace_data)

        result: dict[str, Any] = {
            "project": project,
            "op_name": op_name,
            "count": len(traces),
            "traces": traces,
        }

        if output_format == "table" and traces:
            click.echo(f"\nTrajectory Traces from Weave ({project})")
            click.echo("-" * 80)
            click.echo(
                f"{'Power':<10} {'Year':<6} {'Reward':<10} {'Status':<10} "
                f"{'Orders':<15} {'Run':<20}"
            )
            click.echo("-" * 80)

            for trace in traces:
                traj = trace.get("trajectory", {})
                orders_str = f"{traj.get('orders_extracted', 0)}/{traj.get('orders_expected', 0)}"
                click.echo(
                    f"{traj.get('power', 'N/A'):<10} "
                    f"{traj.get('year', 'N/A'):<6} "
                    f"{traj.get('reward', 0):<10.2f} "
                    f"{traj.get('extraction_status', 'N/A'):<10} "
                    f"{orders_str:<15} "
                    f"{traj.get('run_name', 'N/A')[:20]:<20}"
                )

            click.echo(f"\nTotal: {len(traces)} traces")

        return result

    except Exception as e:
        return {
            "error": str(e),
            "project": project,
            "message": (
                "Failed to fetch traces from Weave. "
                "Make sure you have logged trajectories using log_trajectory()."
            ),
        }


def main() -> None:
    """Main CLI entry point."""
    cli()


if __name__ == "__main__":
    main()
