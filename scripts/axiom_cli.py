#!/usr/bin/env python3
"""
Axiom CLI for log analysis and debugging.

This CLI provides programmatic access to Axiom logs for automated
analysis by coding agents and developers.

Usage:
    # List recent events
    axiom-cli list --event rollout_complete --last 1h --limit 50

    # Get order extraction statistics
    axiom-cli extraction-stats --last 24h

    # Get rollout timing breakdown
    axiom-cli rollout-timing --last 6h

    # Get training step metrics
    axiom-cli training-steps --run-name grpo-20251219-121925 --last 24h

    # Get recent errors
    axiom-cli errors --last 1h

    # System health overview
    axiom-cli health --last 1h

    # Run custom APL query
    axiom-cli query "['diplomacy'] | where event == 'orders_empty' | take 10"

    # Get inference engine stats
    axiom-cli inference-stats --last 1h
"""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime, timedelta
from typing import Any

import click
import httpx

# ============================================================================
# AXIOM API CLIENT
# ============================================================================

AXIOM_API_URL = "https://api.axiom.co/v1/datasets/_apl"
DATASET_NAME = "diplomacy"


def get_axiom_token() -> str:
    """Get Axiom API token from environment."""
    token = os.environ.get("AXIOM_TOKEN")
    if not token:
        raise click.ClickException(
            "AXIOM_TOKEN environment variable not set. Set it with: export AXIOM_TOKEN=<your-token>"
        )
    return token


def parse_time_range(time_str: str) -> str:
    """
    Parse time range string into APL-compatible format.

    Supports formats like: 1h, 24h, 7d, 30m, 1w
    """
    match = re.match(r"^(\d+)([mhdw])$", time_str.lower())
    if not match:
        raise click.ClickException(
            f"Invalid time range: {time_str}. Use formats like: 30m, 1h, 24h, 7d, 1w"
        )

    value, unit = int(match.group(1)), match.group(2)

    unit_map = {
        "m": "m",  # minutes
        "h": "h",  # hours
        "d": "d",  # days
        "w": "d",  # weeks -> days
    }

    if unit == "w":
        value = value * 7

    return f"{value}{unit_map[unit]}"


def execute_apl(query: str, start_time: str | None = None, end_time: str | None = None) -> dict:
    """
    Execute APL query against Axiom API.

    Args:
        query: APL query string
        start_time: ISO8601 start time (optional)
        end_time: ISO8601 end time (optional)

    Returns:
        Query result dictionary
    """
    token = get_axiom_token()

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    payload: dict[str, Any] = {"apl": query}

    if start_time:
        payload["startTime"] = start_time
    if end_time:
        payload["endTime"] = end_time

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(AXIOM_API_URL, json=payload, headers=headers)

            if response.status_code == 401:
                raise click.ClickException("Invalid AXIOM_TOKEN. Check your API token.")
            elif response.status_code == 400:
                error_detail = response.json().get("message", response.text)
                raise click.ClickException(f"APL query error: {error_detail}")
            elif response.status_code != 200:
                raise click.ClickException(
                    f"Axiom API error ({response.status_code}): {response.text}"
                )

            return response.json()

    except httpx.ConnectError as e:
        raise click.ClickException(f"Could not connect to Axiom API: {e}") from e
    except httpx.TimeoutException:
        raise click.ClickException("Axiom API request timed out") from None


def get_time_bounds(last: str) -> tuple[str, str]:
    """Get ISO8601 time bounds from 'last' duration string."""
    now = datetime.now(UTC)
    parsed = parse_time_range(last)

    # Parse the duration
    value = int(parsed[:-1])
    unit = parsed[-1]

    if unit == "m":
        delta = timedelta(minutes=value)
    elif unit == "h":
        delta = timedelta(hours=value)
    elif unit == "d":
        delta = timedelta(days=value)
    else:
        delta = timedelta(hours=value)  # Default to hours

    start = now - delta
    return start.isoformat(), now.isoformat()


# ============================================================================
# CLI COMMANDS
# ============================================================================


@click.group()
@click.version_option(version="1.0.0", prog_name="axiom-cli")
def cli() -> None:
    """Axiom CLI for log analysis and debugging."""
    pass


@cli.command("query")
@click.argument("apl_query")
@click.option(
    "--last", "-l", default="1h", help="Time range (e.g., 1h, 24h, 7d)", show_default=True
)
@click.option("--raw", is_flag=True, help="Output raw API response")
def query_cmd(apl_query: str, last: str, raw: bool) -> None:
    """Execute raw APL query."""
    start_time, end_time = get_time_bounds(last)
    result = execute_apl(apl_query, start_time, end_time)

    if raw:
        click.echo(json.dumps(result, indent=2, default=str))
    else:
        # Extract and format matches
        matches = result.get("matches", [])
        click.echo(json.dumps({"count": len(matches), "matches": matches}, indent=2, default=str))


@cli.command("list")
@click.option("--event", "-e", default=None, help="Filter by event type")
@click.option(
    "--last", "-l", default="1h", help="Time range (e.g., 1h, 24h, 7d)", show_default=True
)
@click.option("--limit", default=50, help="Maximum results", show_default=True)
@click.option("--run-name", "-r", default=None, help="Filter by run name")
def list_cmd(event: str | None, last: str, limit: int, run_name: str | None) -> None:
    """List recent events."""
    # Build query
    filters = []
    if event:
        filters.append(f"event == '{event}'")
    if run_name:
        filters.append(f"run_name == '{run_name}'")

    where_clause = " and ".join(filters) if filters else "true"

    apl = f"""
    ['{DATASET_NAME}']
    | where {where_clause}
    | sort by _time desc
    | take {limit}
    """

    start_time, end_time = get_time_bounds(last)
    result = execute_apl(apl, start_time, end_time)

    matches = result.get("matches", [])
    output = {"count": len(matches), "events": [m.get("data", m) for m in matches]}
    click.echo(json.dumps(output, indent=2, default=str))


@cli.command("extraction-stats")
@click.option("--last", "-l", default="24h", help="Time range", show_default=True)
@click.option("--by-power", is_flag=True, help="Break down by power")
def extraction_stats_cmd(last: str, by_power: bool) -> None:
    """Get order extraction statistics."""
    if by_power:
        apl = f"""
        ['{DATASET_NAME}']
        | where event in ('orders_extracted', 'orders_empty', 'orders_partial')
        | summarize
            total = count(),
            extracted = countif(event == 'orders_extracted'),
            empty = countif(event == 'orders_empty'),
            partial = countif(event == 'orders_partial')
          by power
        | extend extraction_rate = round(toreal(extracted) / total * 100, 2)
        | sort by extraction_rate desc
        """
    else:
        apl = f"""
        ['{DATASET_NAME}']
        | where event in ('orders_extracted', 'orders_empty', 'orders_partial')
        | summarize
            total = count(),
            extracted = countif(event == 'orders_extracted'),
            empty = countif(event == 'orders_empty'),
            partial = countif(event == 'orders_partial')
        | extend extraction_rate = round(toreal(extracted) / total * 100, 2)
        """

    start_time, end_time = get_time_bounds(last)
    result = execute_apl(apl, start_time, end_time)

    # Format output
    buckets = result.get("buckets", {}).get("totals", [])
    if buckets:
        output = {"time_range": last, "stats": buckets}
    else:
        # Try matches as fallback
        matches = result.get("matches", [])
        output = {"time_range": last, "stats": [m.get("data", m) for m in matches]}

    click.echo(json.dumps(output, indent=2, default=str))


@cli.command("rollout-timing")
@click.option("--last", "-l", default="6h", help="Time range", show_default=True)
@click.option("--by-horizon", is_flag=True, help="Break down by horizon type")
def rollout_timing_cmd(last: str, by_horizon: bool) -> None:
    """Get rollout timing breakdown."""
    if by_horizon:
        apl = f"""
        ['{DATASET_NAME}']
        | where event == 'rollout_complete'
        | summarize
            count = count(),
            avg_duration_s = avg(duration_s),
            p50_duration = percentile(duration_s, 50),
            p95_duration = percentile(duration_s, 95),
            max_duration = max(duration_s)
          by horizon_type
        """
    else:
        apl = f"""
        ['{DATASET_NAME}']
        | where event == 'rollout_complete'
        | summarize
            count = count(),
            avg_duration_s = round(avg(duration_s), 2),
            p50_duration = round(percentile(duration_s, 50), 2),
            p95_duration = round(percentile(duration_s, 95), 2),
            max_duration = round(max(duration_s), 2),
            total_trajectories = sum(trajectory_count)
        """

    start_time, end_time = get_time_bounds(last)
    result = execute_apl(apl, start_time, end_time)

    buckets = result.get("buckets", {}).get("totals", [])
    if buckets:
        output = {"time_range": last, "timing": buckets}
    else:
        matches = result.get("matches", [])
        output = {"time_range": last, "timing": [m.get("data", m) for m in matches]}

    click.echo(json.dumps(output, indent=2, default=str))


@cli.command("training-steps")
@click.option(
    "--run-name", "-r", required=True, help="Training run name (e.g., grpo-20251219-121925)"
)
@click.option("--last", "-l", default="24h", help="Time range", show_default=True)
@click.option("--limit", default=100, help="Maximum steps to return", show_default=True)
def training_steps_cmd(run_name: str, last: str, limit: int) -> None:
    """Get training step metrics for a run."""
    apl = f"""
    ['{DATASET_NAME}']
    | where event == 'training_step' and run_name == '{run_name}'
    | project
        _time,
        step,
        loss,
        kl_divergence,
        grad_norm,
        learning_rate,
        batch_size,
        effective_batch_size
    | sort by step asc
    | take {limit}
    """

    start_time, end_time = get_time_bounds(last)
    result = execute_apl(apl, start_time, end_time)

    matches = result.get("matches", [])
    steps = [m.get("data", m) for m in matches]

    # Calculate summary stats if we have data
    summary = {}
    if steps:
        losses = [s.get("loss") for s in steps if s.get("loss") is not None]
        kls = [s.get("kl_divergence") for s in steps if s.get("kl_divergence") is not None]
        grads = [s.get("grad_norm") for s in steps if s.get("grad_norm") is not None]

        if losses:
            summary["loss"] = {
                "first": losses[0],
                "last": losses[-1],
                "min": min(losses),
                "max": max(losses),
                "mean": sum(losses) / len(losses),
            }
        if kls:
            summary["kl_divergence"] = {
                "first": kls[0],
                "last": kls[-1],
                "min": min(kls),
                "max": max(kls),
                "mean": sum(kls) / len(kls),
            }
        if grads:
            summary["grad_norm"] = {
                "first": grads[0],
                "last": grads[-1],
                "min": min(grads),
                "max": max(grads),
                "mean": sum(grads) / len(grads),
            }

    output = {
        "run_name": run_name,
        "time_range": last,
        "step_count": len(steps),
        "summary": summary,
        "steps": steps,
    }

    click.echo(json.dumps(output, indent=2, default=str))


@cli.command("errors")
@click.option("--last", "-l", default="1h", help="Time range", show_default=True)
@click.option("--limit", default=50, help="Maximum errors to return", show_default=True)
@click.option("--severity", "-s", default=None, help="Filter by severity (error, warning)")
def errors_cmd(last: str, limit: int, severity: str | None) -> None:
    """Get recent errors and warnings."""
    severity_filter = ""
    if severity:
        severity_filter = f"and level == '{severity}'"

    apl = f"""
    ['{DATASET_NAME}']
    | where level in ('error', 'warning') or event contains 'error' or event contains 'fail'
    {severity_filter}
    | sort by _time desc
    | take {limit}
    | project _time, event, level, message, error, run_name, rollout_id
    """

    start_time, end_time = get_time_bounds(last)
    result = execute_apl(apl, start_time, end_time)

    matches = result.get("matches", [])
    errors = [m.get("data", m) for m in matches]

    output = {"time_range": last, "count": len(errors), "errors": errors}
    click.echo(json.dumps(output, indent=2, default=str))


@cli.command("health")
@click.option("--last", "-l", default="1h", help="Time range", show_default=True)
def health_cmd(last: str) -> None:
    """Get system health overview."""
    # Run multiple queries for different health metrics
    start_time, end_time = get_time_bounds(last)

    health: dict[str, Any] = {"time_range": last, "status": "unknown", "metrics": {}}

    # 1. Event counts by type
    try:
        event_counts_apl = f"""
        ['{DATASET_NAME}']
        | summarize count = count() by event
        | sort by count desc
        | take 20
        """
        result = execute_apl(event_counts_apl, start_time, end_time)
        buckets = result.get("buckets", {}).get("totals", [])
        health["metrics"]["event_counts"] = buckets if buckets else []
    except Exception as e:
        health["metrics"]["event_counts"] = {"error": str(e)}

    # 2. Error rate
    try:
        error_rate_apl = f"""
        ['{DATASET_NAME}']
        | summarize
            total = count(),
            errors = countif(level == 'error' or event contains 'error')
        | extend error_rate = round(toreal(errors) / total * 100, 2)
        """
        result = execute_apl(error_rate_apl, start_time, end_time)
        buckets = result.get("buckets", {}).get("totals", [])
        health["metrics"]["error_rate"] = buckets[0] if buckets else {}
    except Exception as e:
        health["metrics"]["error_rate"] = {"error": str(e)}

    # 3. Rollout completion rate
    try:
        rollout_apl = f"""
        ['{DATASET_NAME}']
        | where event in ('rollout_start', 'rollout_complete', 'rollout_error')
        | summarize
            started = countif(event == 'rollout_start'),
            completed = countif(event == 'rollout_complete'),
            failed = countif(event == 'rollout_error')
        | extend completion_rate = round(toreal(completed) / started * 100, 2)
        """
        result = execute_apl(rollout_apl, start_time, end_time)
        buckets = result.get("buckets", {}).get("totals", [])
        health["metrics"]["rollouts"] = buckets[0] if buckets else {}
    except Exception as e:
        health["metrics"]["rollouts"] = {"error": str(e)}

    # 4. Inference performance
    try:
        inference_apl = f"""
        ['{DATASET_NAME}']
        | where event == 'inference_response'
        | summarize
            requests = count(),
            avg_duration_ms = avg(duration_ms),
            avg_tps = avg(tokens_per_second),
            avg_batch_size = avg(batch_size)
        """
        result = execute_apl(inference_apl, start_time, end_time)
        buckets = result.get("buckets", {}).get("totals", [])
        health["metrics"]["inference"] = buckets[0] if buckets else {}
    except Exception as e:
        health["metrics"]["inference"] = {"error": str(e)}

    # Determine overall status
    error_info = health["metrics"].get("error_rate", {})
    if isinstance(error_info, dict) and "error_rate" in error_info:
        error_rate = error_info.get("error_rate", 0)
        if error_rate < 1:
            health["status"] = "healthy"
        elif error_rate < 5:
            health["status"] = "degraded"
        else:
            health["status"] = "unhealthy"
    else:
        health["status"] = "unknown"

    click.echo(json.dumps(health, indent=2, default=str))


@cli.command("inference-stats")
@click.option("--last", "-l", default="1h", help="Time range", show_default=True)
@click.option("--by-adapter", is_flag=True, help="Break down by LoRA adapter")
def inference_stats_cmd(last: str, by_adapter: bool) -> None:
    """Get inference engine statistics."""
    if by_adapter:
        apl = f"""
        ['{DATASET_NAME}']
        | where event == 'inference_timing_breakdown'
        | summarize
            requests = count(),
            avg_total_time_s = round(avg(total_time_s), 3),
            avg_gen_time_s = round(avg(generation_time_s), 3),
            avg_adapter_load_s = round(avg(adapter_load_time_s), 3),
            avg_tps = round(avg(tokens_per_second), 1),
            total_tokens = sum(tokens_generated)
          by lora_name
        | sort by requests desc
        """
    else:
        apl = f"""
        ['{DATASET_NAME}']
        | where event == 'inference_timing_breakdown'
        | summarize
            requests = count(),
            avg_total_time_s = round(avg(total_time_s), 3),
            avg_gen_time_s = round(avg(generation_time_s), 3),
            avg_adapter_load_s = round(avg(adapter_load_time_s), 3),
            avg_input_tps = round(avg(input_tokens_per_second), 1),
            avg_output_tps = round(avg(output_tokens_per_second), 1),
            total_tokens = sum(tokens_generated),
            avg_batch_size = round(avg(batch_size), 1),
            avg_cache_hit_rate = round(avg(prefix_cache_hit_rate) * 100, 2)
        """

    start_time, end_time = get_time_bounds(last)
    result = execute_apl(apl, start_time, end_time)

    buckets = result.get("buckets", {}).get("totals", [])
    if buckets:
        output = {"time_range": last, "stats": buckets}
    else:
        matches = result.get("matches", [])
        output = {"time_range": last, "stats": [m.get("data", m) for m in matches]}

    click.echo(json.dumps(output, indent=2, default=str))


@cli.command("cache-stats")
@click.option("--last", "-l", default="1h", help="Time range", show_default=True)
def cache_stats_cmd(last: str) -> None:
    """Get prefix cache statistics from inference logs."""
    apl = f"""
    ['{DATASET_NAME}']
    | where event == 'prefix_cache_stats'
    | summarize
        samples = count(),
        avg_hit_rate = round(avg(hit_rate) * 100, 2),
        max_hit_rate = round(max(hit_rate) * 100, 2),
        total_queries = sum(queries),
        total_hits = sum(hits),
        total_prompt_tokens = sum(prompt_tokens_total)
    """

    start_time, end_time = get_time_bounds(last)
    result = execute_apl(apl, start_time, end_time)

    buckets = result.get("buckets", {}).get("totals", [])
    if buckets and buckets[0]:
        stats = buckets[0]
        # Calculate overall hit rate
        if stats.get("total_queries", 0) > 0:
            stats["overall_hit_rate"] = round(stats["total_hits"] / stats["total_queries"] * 100, 2)
        output = {"time_range": last, "cache_stats": stats}
    else:
        output = {
            "time_range": last,
            "cache_stats": None,
            "message": "No prefix cache stats found. Check if caching is enabled.",
        }

    click.echo(json.dumps(output, indent=2, default=str))


@cli.command("game-outcomes")
@click.option("--last", "-l", default="24h", help="Time range", show_default=True)
@click.option("--by-power", is_flag=True, help="Break down by power")
def game_outcomes_cmd(last: str, by_power: bool) -> None:
    """Get game outcome statistics."""
    if by_power:
        apl = f"""
        ['{DATASET_NAME}']
        | where event == 'rollout_complete'
        | mv-expand power = ['AUSTRIA', 'ENGLAND', 'FRANCE', 'GERMANY', 'ITALY', 'RUSSIA', 'TURKEY']
        | project power, final_score = extract_json_member(final_scores, power)
        | summarize
            games = count(),
            avg_score = round(avg(toreal(final_score)), 2),
            max_score = max(toreal(final_score))
          by power
        | sort by avg_score desc
        """
    else:
        apl = f"""
        ['{DATASET_NAME}']
        | where event == 'rollout_complete'
        | summarize
            games = count(),
            avg_trajectory_count = round(avg(trajectory_count), 1),
            avg_duration_s = round(avg(duration_s), 1)
        """

    start_time, end_time = get_time_bounds(last)
    result = execute_apl(apl, start_time, end_time)

    buckets = result.get("buckets", {}).get("totals", [])
    if buckets:
        output = {"time_range": last, "outcomes": buckets}
    else:
        matches = result.get("matches", [])
        output = {"time_range": last, "outcomes": [m.get("data", m) for m in matches]}

    click.echo(json.dumps(output, indent=2, default=str))


def main() -> None:
    """Main CLI entry point."""
    cli()


if __name__ == "__main__":
    main()
