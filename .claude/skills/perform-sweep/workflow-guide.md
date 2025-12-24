# Sweep Workflow Guide

Detailed guide for the complete hypothesis-to-analysis workflow.

## Phase 1: Hypothesis Formation

### Review Prior Work

Before designing a new sweep, understand what's been tried:

```bash
# Check experiment tracker
cat experiments/experiment-tracker.md

# List existing sweeps
ls experiments/sweeps/

# Review a specific prior sweep
cat experiments/sweeps/<name>/sweep.yaml
cat experiments/sweeps/<name>/results.md
```

### Identify the Question

Good sweep hypotheses are:
- **Specific**: Test one variable at a time
- **Measurable**: Clear metrics to evaluate
- **Falsifiable**: Can be proven wrong
- **Actionable**: Results inform next steps

Examples:
- "Longer horizons improve late-game play" (test with horizon sweep)
- "Strategic scoring reduces SC-hoarding" (test with scoring sweep)
- "Higher KL penalty prevents reward hacking" (test with KL sweep)

### Document Your Prediction

Before running, write down:
1. What you expect to happen
2. Why you expect it
3. What would disprove the hypothesis

This goes in the `hypothesis` field of sweep.yaml.

## Phase 2: Configuration

### Create Sweep Directory

```bash
mkdir -p experiments/sweeps/<sweep-name>
```

### Write sweep.yaml

Start with the template:

```yaml
metadata:
  name: "<sweep-name>"
  description: "<one-line description>"
  hypothesis: |
    <your hypothesis>
    <expected outcomes>
  experiment_tag_prefix: "<short-prefix>"

defaults:
  total_steps: 100
  # shared config

runs:
  A:
    name: "control"
    config:
      experiment_tag: "${metadata.experiment_tag_prefix}-A"
  B:
    name: "treatment"
    config:
      <your changes>
      experiment_tag: "${metadata.experiment_tag_prefix}-B"

analysis:
  expected_ranking: ["B", "A"]  # your prediction
```

### Validate Configuration

```bash
# Check syntax and structure
python scripts/launch_sweep.py experiments/sweeps/<name>/ --info

# Dry run (validates can build ExperimentConfig)
python scripts/launch_sweep.py experiments/sweeps/<name>/ --dry-run
```

Common errors:
- Invalid field names (typos in ExperimentConfig fields)
- YAML syntax errors (indentation, missing colons)
- Missing required fields (name, experiment_tag_prefix)

## Phase 3: Execution

### Launch the Sweep

```bash
# Launch all runs
python scripts/launch_sweep.py experiments/sweeps/<name>/

# Launch specific runs
python scripts/launch_sweep.py experiments/sweeps/<name>/ --run A B
```

This spawns a Modal orchestrator and exits. You can close your laptop.

### Monitor Progress

```bash
# Check status
python scripts/launch_sweep.py experiments/sweeps/<name>/ --status

# List all sweeps
python scripts/launch_sweep.py --list
```

Also monitor in:
- **Modal Dashboard**: https://modal.com/apps
- **WandB**: Filter by `experiment_tag` containing your prefix

### Handle Failures

If a run fails:
1. Check Modal logs for error
2. Fix the issue (config, code, etc.)
3. Re-run the sweep - it will skip completed runs

```bash
# Re-launch (skips already completed runs)
python scripts/launch_sweep.py experiments/sweeps/<name>/

# Force re-run specific runs
python scripts/launch_sweep.py experiments/sweeps/<name>/ --run B
```

## Phase 4: Analysis

### Per-Run Analysis

Use the `experiment-analysis` skill:

```bash
# Full Elo analysis for each run
uv run python .claude/skills/experiment-analysis/analyze_elo.py <run-name-A>
uv run python .claude/skills/experiment-analysis/analyze_elo.py <run-name-B>

# Get all metrics
uv run python scripts/wandb_cli.py get-metrics -r <run-name> --all-metrics
```

### Cross-Run Comparison

In WandB:
1. Go to project: `diplomacy-grpo`
2. Filter runs by `experiment_tag` containing your prefix
3. Create comparison chart with:
   - X-axis: `_step` or `power_law/cumulative_simulated_years`
   - Y-axis: Your primary metric (e.g., `trueskill/display_rating`)

### Evaluate Results

Check against your predictions:
1. Did the expected ranking hold? (D > C > B > A)
2. What magnitude were the differences?
3. Were there any surprises?
4. What did you learn?

### Document Findings

Create `experiments/sweeps/<name>/results.md`:

```markdown
# Results: <Sweep Name>

## Summary

<1-2 sentence summary of findings>

## Key Metrics

| Run | TrueSkill | Win Rate | Duration |
|-----|-----------|----------|----------|
| A   | ...       | ...      | ...      |
| B   | ...       | ...      | ...      |

## Findings

1. <Finding 1>
2. <Finding 2>

## Hypothesis Evaluation

- **Confirmed**: <what was confirmed>
- **Refuted**: <what was refuted>
- **Unexpected**: <surprises>

## Next Steps

1. <Recommended follow-up>
2. <Additional experiments>
```

### Update Experiment Tracker

Add results to `experiments/experiment-tracker.md`.

## Tips and Best Practices

### Sweep Design

- **Start small**: 2-4 runs, 50-100 steps each
- **One variable**: Test one thing at a time
- **Include control**: Always have a baseline (run A)
- **Clear naming**: Use descriptive run names

### Configuration

- **Use defaults**: Put shared config in `defaults`
- **Tag everything**: Always set `experiment_tag` for WandB filtering
- **Document overrides**: Comment non-obvious config choices

### Execution

- **Dry run first**: Always validate with `--dry-run`
- **Check early**: Monitor first run for issues before walking away
- **Keep notes**: Document any manual interventions

### Analysis

- **Wait for completion**: Don't draw conclusions from partial runs
- **Compare fairly**: Use same metrics across runs
- **Consider variance**: Single runs have noise - major differences only
- **Document everything**: Future you will thank present you
