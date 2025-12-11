# WandB CLI Reference

A command-line interface for programmatic access to WandB experiment data, designed for automated analysis and hypothesis generation by coding agents.

## Installation

The CLI uses the `wandb` package which is already in the project dependencies. Ensure you're authenticated:

```bash
wandb login
```

## Commands

### List Runs

List runs with optional filtering by tags, state, etc.

```bash
# List all runs
python scripts/wandb_cli.py list --project diplomacy-grpo

# Filter by tags
python scripts/wandb_cli.py list --project diplomacy-grpo --tags power-laws --limit 10

# Filter by state
python scripts/wandb_cli.py list --project diplomacy-grpo --state finished

# JSON output (for programmatic use)
python scripts/wandb_cli.py list --project diplomacy-grpo --output-format json
```

### Get Run Details

Get comprehensive information about a specific run including config, summary metrics, and optionally history/artifacts.

```bash
# Basic run info
python scripts/wandb_cli.py get-run --run-id <run_id> --project diplomacy-grpo

# Include full metric history
python scripts/wandb_cli.py get-run --run-id <run_id> --project diplomacy-grpo --include-history

# Include artifacts
python scripts/wandb_cli.py get-run --run-id <run_id> --project diplomacy-grpo --include-artifacts

# JSON output
python scripts/wandb_cli.py get-run --run-id <run_id> --project diplomacy-grpo --output-format json
```

### Get Metrics Over Time

Retrieve time-series data for specific metrics or all metrics.

```bash
# Specific metrics
python scripts/wandb_cli.py get-metrics --run-id <run_id> --project diplomacy-grpo \
    --metrics elo/challenger benchmark/loss benchmark/reward_mean

# All metrics
python scripts/wandb_cli.py get-metrics --run-id <run_id> --project diplomacy-grpo --all-metrics

# Use timestamp as x-axis instead of step
python scripts/wandb_cli.py get-metrics --run-id <run_id> --project diplomacy-grpo \
    --metrics elo/challenger --x-axis _timestamp

# JSON output
python scripts/wandb_cli.py get-metrics --run-id <run_id> --project diplomacy-grpo \
    --metrics elo/challenger --output-format json
```

### Get Artifacts

List artifacts associated with a run.

```bash
# All artifacts
python scripts/wandb_cli.py get-artifacts --run-id <run_id> --project diplomacy-grpo

# Filter by artifact type
python scripts/wandb_cli.py get-artifacts --run-id <run_id> --project diplomacy-grpo \
    --artifact-type dataset

# JSON output
python scripts/wandb_cli.py get-artifacts --run-id <run_id> --project diplomacy-grpo \
    --output-format json
```

### Compare Runs

Compare multiple runs across specified metrics.

```bash
# Compare summary metrics
python scripts/wandb_cli.py compare --run-ids <id1> <id2> <id3> --project diplomacy-grpo \
    --metrics elo/challenger benchmark/loss benchmark/reward_mean

# JSON output
python scripts/wandb_cli.py compare --run-ids <id1> <id2> --project diplomacy-grpo \
    --metrics elo/challenger --output-format json
```

### Search Runs

Search runs by config values and tags.

```bash
# Search by config values
python scripts/wandb_cli.py search --project diplomacy-grpo \
    --config "lora_rank=16" --config "base_model_id=Qwen/Qwen2.5-7B-Instruct"

# Search by tags
python scripts/wandb_cli.py search --project diplomacy-grpo --tag power-laws --tag baseline

# Combine filters
python scripts/wandb_cli.py search --project diplomacy-grpo \
    --config "lora_rank=16" --tag power-laws --limit 20

# JSON output
python scripts/wandb_cli.py search --project diplomacy-grpo \
    --config "lora_rank=16" --output-format json
```

### Export Run Data

Export complete run data to a JSON file for offline analysis.

```bash
# Full export (includes history and artifacts)
python scripts/wandb_cli.py export --run-id <run_id> --project diplomacy-grpo \
    --output run_data.json

# Exclude history (faster, smaller file)
python scripts/wandb_cli.py export --run-id <run_id> --project diplomacy-grpo \
    --output run_summary.json --no-history

# Exclude artifacts
python scripts/wandb_cli.py export --run-id <run_id> --project diplomacy-grpo \
    --output run_data.json --no-artifacts
```

## Programmatic Usage

The CLI is designed to be used by coding agents. All commands support `--output-format json` which returns structured JSON data that can be parsed programmatically.

### Example: Python Script

```python
import json
import subprocess

def get_run_metrics(run_id: str, project: str = "diplomacy-grpo") -> dict:
    """Get metrics for a run."""
    result = subprocess.run(
        [
            "python", "scripts/wandb_cli.py",
            "get-metrics",
            "--run-id", run_id,
            "--project", project,
            "--all-metrics",
            "--output-format", "json"
        ],
        capture_output=True,
        text=True
    )
    return json.loads(result.stdout)

# Usage
metrics = get_run_metrics("abc123")
print(f"Found {len(metrics['metrics'])} metrics")
for metric_name, metric_data in metrics['metrics'].items():
    print(f"{metric_name}: mean={metric_data['mean']:.4f}")
```

### Example: Shell Script

```bash
#!/bin/bash

# Get all runs with a specific tag
RUNS=$(python scripts/wandb_cli.py list \
    --project diplomacy-grpo \
    --tags power-laws \
    --output-format json)

# Extract run IDs
RUN_IDS=$(echo "$RUNS" | jq -r '.runs[].id')

# Compare metrics across runs
for run_id in $RUN_IDS; do
    echo "Analyzing run $run_id..."
    python scripts/wandb_cli.py get-metrics \
        --run-id "$run_id" \
        --project diplomacy-grpo \
        --metrics elo/challenger benchmark/loss \
        --output-format json > "metrics_${run_id}.json"
done
```

## Output Formats

### JSON Format

All commands support `--output-format json` which returns structured JSON:

- **List**: `{"count": N, "runs": [...]}`
- **Get Run**: `{"id": "...", "name": "...", "config": {...}, "summary": {...}, ...}`
- **Get Metrics**: `{"run_id": "...", "metrics": {"metric_name": {"values": [...], "min": ..., "max": ..., ...}}}`
- **Get Artifacts**: `{"run_id": "...", "artifacts": [...]}`
- **Compare**: `{"runs": [...], "comparison": {"metric": [...]}}`
- **Search**: `{"count": N, "runs": [...]}`

### Table Format

Default format for human-readable output. Shows formatted tables and summaries.

## Common Use Cases

### Hypothesis Generation

```bash
# 1. Find runs with different configs
python scripts/wandb_cli.py search --project diplomacy-grpo \
    --config "lora_rank=16" --output-format json > runs_rank16.json

python scripts/wandb_cli.py search --project diplomacy-grpo \
    --config "lora_rank=32" --output-format json > runs_rank32.json

# 2. Compare final Elo scores
python scripts/wandb_cli.py compare \
    --run-ids $(jq -r '.runs[].id' runs_rank16.json | head -5) \
    --project diplomacy-grpo \
    --metrics elo/challenger \
    --output-format json > comparison.json

# 3. Analyze metric trends
python scripts/wandb_cli.py get-metrics \
    --run-id <best_run_id> \
    --project diplomacy-grpo \
    --metrics elo/challenger benchmark/loss \
    --output-format json > trends.json
```

### Automated Analysis

```bash
# Export all recent runs for batch analysis
python scripts/wandb_cli.py list --project diplomacy-grpo --limit 50 --output-format json | \
    jq -r '.runs[].id' | \
    while read run_id; do
        python scripts/wandb_cli.py export \
            --run-id "$run_id" \
            --project diplomacy-grpo \
            --output "exports/${run_id}.json" \
            --no-history  # Faster export
    done
```

## Integration with Project

The CLI uses the default project name `diplomacy-grpo` which matches the project's WandB configuration. You can override this with `--project` or set a different default in the script.

## Error Handling

The CLI returns non-zero exit codes on errors and prints error messages to stderr. JSON output is always valid JSON, even on errors (may contain an "error" field).

## Performance Notes

- **History retrieval**: Can be slow for runs with many logged steps. Use `--no-history` when exporting if you only need summary metrics.
- **Artifact listing**: May take time for runs with many artifacts. Use `--no-artifacts` to skip.
- **Large exports**: Consider filtering metrics or excluding history for faster exports.
