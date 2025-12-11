# Analyze Experiment

Analyze a WandB experiment run to understand training dynamics, performance, and identify potential issues or improvements.

## Usage

```bash
analyze-experiment <run_name>
```

Where `<run_name>` is the WandB run name or run ID.

## Step-by-Step Analysis Instructions

### 0. Activate the virtual environment
```bash
source .venv/bin/activate
```

### 1. Get Run Information

First, retrieve the run details to understand the experiment configuration:

```bash
python scripts/wandb_cli.py get-run --run-id <run_name> --project diplomacy-grpo --include-history --include-artifacts --output-format json
```

**Key things to check:**
- **Config values**: Model architecture (`base_model_id`, `lora_rank`), training hyperparameters (`learning_rate`, `batch_size`, `rollout_horizon_years`), experiment settings (`num_groups_per_step`, `samples_per_group`)
- **Run state**: Is the run `finished`, `running`, or `crashed`?
- **Tags**: What tags are associated (e.g., `power-laws`, `baseline`)?
- **Summary metrics**: Final values for key metrics

### 2. Analyze Training Metrics Over Time

Get time-series data for core training metrics:

```bash
python scripts/wandb_cli.py get-metrics --run-id <run_name> --project diplomacy-grpo \
    --metrics benchmark/loss benchmark/kl benchmark/reward_mean benchmark/reward_std \
    benchmark/grad_norm --output-format json
```

**Analysis focus:**
- **Loss trends**: Is loss decreasing smoothly or showing instability/divergence?
- **KL divergence**: Monitor for policy collapse (KL should stay within reasonable bounds)
- **Reward progression**: Is `benchmark/reward_mean` improving over time? Check for plateaus or regressions
- **Gradient norms**: Large spikes may indicate training instability
- **Reward variance**: High `benchmark/reward_std` suggests high variance in game outcomes

### 3. Analyze Performance Metrics

Check throughput and efficiency:

```bash
python scripts/wandb_cli.py get-metrics --run-id <run_name> --project diplomacy-grpo \
    --metrics benchmark/rollout_time_s benchmark/training_time_s benchmark/pipeline_overlap_s \
    benchmark/trajectories --output-format json
```

**Analysis focus:**
- **Throughput**: Trajectories per second (compute from `benchmark/trajectories` / `benchmark/rollout_time_s`)
- **Pipeline efficiency**: `benchmark/pipeline_overlap_s` shows time saved from parallel execution
- **Training vs rollout time**: Ratio indicates if training is bottleneck
- **Trajectory count**: Ensure sufficient data collection per step

### 4. Analyze Order Extraction Quality

Monitor parsing/extraction reliability:

```bash
python scripts/wandb_cli.py get-metrics --run-id <run_name> --project diplomacy-grpo \
    --metrics extraction/rate extraction/orders_expected extraction/orders_extracted \
    extraction/empty_responses extraction/partial_responses --output-format json
```

**Analysis focus:**
- **Extraction rate**: Should be >95%. Low rates indicate prompt structure issues or model degradation
- **Empty responses**: Sudden spikes suggest model output quality problems
- **Partial responses**: May indicate context length issues or incomplete generation
- **Expected vs extracted**: Large gaps indicate parsing failures

### 5. Analyze Rollout Reliability

Check for infrastructure issues:

```bash
python scripts/wandb_cli.py get-metrics --run-id <run_name> --project diplomacy-grpo \
    --metrics rollout/max_volume_reload_s rollout/max_total_s rollout/failed_count --output-format json
```

**Analysis focus:**
- **Failed rollouts**: `rollout/failed_count` should be 0 or minimal. Spikes indicate infrastructure issues
- **Volume reload time**: `rollout/max_volume_reload_s` spikes suggest volume I/O bottlenecks
- **Total rollout time**: Compare `rollout/max_total_s` to expected values. Large spikes may indicate hanging games

### 6. Analyze Power Law Scaling

For power law experiments, check scaling behavior:

```bash
python scripts/wandb_cli.py get-metrics --run-id <run_name> --project diplomacy-grpo \
    --metrics power_law/cumulative_simulated_years power_law/simulated_years_per_step \
    power_law/reward_at_compute --output-format json
```

**Analysis focus:**
- **Cumulative compute**: Track `power_law/cumulative_simulated_years` progression
- **Compute per step**: `power_law/simulated_years_per_step` should be consistent
- **Reward scaling**: Plot `power_law/reward_at_compute` vs `power_law/cumulative_simulated_years` to check for power law scaling

### 7. Analyze League Training (if enabled)

If league training is enabled, check Elo progression:

```bash
python scripts/wandb_cli.py get-metrics --run-id <run_name> --project diplomacy-grpo \
    --metrics elo/challenger elo/win_rate elo/games_played league/num_checkpoints \
    league/best_elo --output-format json
```

**Analysis focus:**
- **Elo progression**: Is `elo/challenger` improving over time?
- **Win rate**: `elo/win_rate` against gatekeepers should improve with training
- **League growth**: `league/num_checkpoints` should increase as new checkpoints are added
- **Best Elo**: Track `league/best_elo` to see peak performance
- **Evaluation frequency**: Check `elo/games_played` to ensure sufficient evaluation games

### 8. Analyze Evaluation Results (if available)

Check evaluation artifacts and metrics:

```bash
python scripts/wandb_cli.py get-artifacts --run-id <run_name> --project diplomacy-grpo --output-format json
```

Look for:
- **Evaluation artifacts**: `eval-replays-*` artifacts contain game visualizations
- **Evaluation metrics**: Check for `eval/vs_{opponent}/win_rate`, `eval/vs_{opponent}/survival_rate`, `eval/vs_{opponent}/avg_centers`
- **Summary tables**: `eval/summary_table` provides opponent-by-opponent breakdown

### 9. Compare with Other Runs

Compare this run against similar experiments:

```bash
# Find similar runs (same config, different hyperparameters)
python scripts/wandb_cli.py search --project diplomacy-grpo \
    --config "base_model_id=Qwen/Qwen2.5-7B-Instruct" --config "lora_rank=16" \
    --output-format json

# Compare key metrics
python scripts/wandb_cli.py compare --run-ids <run1> <run2> <run3> --project diplomacy-grpo \
    --metrics benchmark/loss benchmark/reward_mean elo/challenger --output-format json
```

**Comparison focus:**
- **Final performance**: Compare summary metrics (final loss, final reward, best Elo)
- **Training efficiency**: Compare trajectories per second, total training time
- **Convergence speed**: Compare steps to reach target metrics

### 10. Export Complete Data for Deep Analysis

Export full run data for offline analysis:

```bash
python scripts/wandb_cli.py export --run-id <run_name> --project diplomacy-grpo \
    --output analysis_<run_name>.json
```

This creates a complete JSON file with:
- Full config
- All summary metrics
- Complete metric history (if included)
- Artifact information

## Common Analysis Patterns

### Pattern 1: Training Instability Detection

**Symptoms:**
- Spikes in `benchmark/loss` or `benchmark/grad_norm`
- High variance in `benchmark/reward_mean`
- Sudden drops in `extraction/rate`

**Investigation:**
```bash
# Get loss and grad norm over time
python scripts/wandb_cli.py get-metrics --run-id <run_name> --project diplomacy-grpo \
    --metrics benchmark/loss benchmark/grad_norm benchmark/reward_mean --output-format json
```

**Hypothesis generation:**
- Learning rate too high → reduce learning rate
- Batch size too small → increase batch size
- Gradient clipping needed → add gradient clipping

### Pattern 2: Performance Plateau

**Symptoms:**
- `benchmark/reward_mean` stops improving
- `elo/challenger` plateaus
- Loss stops decreasing

**Investigation:**
```bash
# Check reward and Elo progression
python scripts/wandb_cli.py get-metrics --run-id <run_name> --project diplomacy-grpo \
    --metrics benchmark/reward_mean elo/challenger benchmark/loss --x-axis _timestamp --output-format json
```

**Hypothesis generation:**
- Need more training steps → increase `total_steps`
- Need more diverse opponents → add more league opponents
- Learning rate decay → implement learning rate scheduling
- Exploration vs exploitation → adjust temperature/sampling

### Pattern 3: Infrastructure Bottleneck

**Symptoms:**
- High `rollout/max_total_s` or `rollout/max_volume_reload_s`
- Non-zero `rollout/failed_count`
- Low `benchmark/pipeline_overlap_s`

**Investigation:**
```bash
# Check rollout metrics
python scripts/wandb_cli.py get-metrics --run-id <run_name> --project diplomacy-grpo \
    --metrics rollout/max_total_s rollout/max_volume_reload_s rollout/failed_count \
    benchmark/pipeline_overlap_s --output-format json
```

**Hypothesis generation:**
- Volume I/O bottleneck → optimize checkpoint saving frequency
- Insufficient parallelism → increase `num_groups_per_step`
- Modal scaling issues → check Modal dashboard for container scaling

### Pattern 4: Model Quality Degradation

**Symptoms:**
- Decreasing `extraction/rate`
- Increasing `extraction/empty_responses`
- Degrading `benchmark/reward_mean` despite low loss

**Investigation:**
```bash
# Check extraction and reward metrics
python scripts/wandb_cli.py get-metrics --run-id <run_name> --project diplomacy-grpo \
    --metrics extraction/rate extraction/empty_responses benchmark/reward_mean \
    benchmark/loss --output-format json
```

**Hypothesis generation:**
- Overfitting → increase regularization or reduce model capacity
- Distribution shift → check if game distribution changed
- Prompt structure issues → review prompt engineering

### Pattern 5: Power Law Scaling Analysis

**Symptoms:**
- Need to verify scaling laws
- Compare compute efficiency across runs

**Investigation:**
```bash
# Get power law metrics
python scripts/wandb_cli.py get-metrics --run-id <run_name> --project diplomacy-grpo \
    --metrics power_law/cumulative_simulated_years power_law/reward_at_compute \
    --x-axis _timestamp --output-format json

# Compare with other power law runs
python scripts/wandb_cli.py search --project diplomacy-grpo --tag power-laws --output-format json
```

**Hypothesis generation:**
- Plot log(reward) vs log(compute) to check for power law scaling
- Compare scaling exponents across different model sizes
- Identify optimal compute budget for target performance

## Automated Analysis Workflow

For programmatic analysis, use this workflow:

1. **Get run details** → Extract config and summary
2. **Get training metrics** → Analyze loss, reward, KL trends
3. **Get performance metrics** → Check throughput and efficiency
4. **Get extraction metrics** → Verify parsing reliability
5. **Get rollout metrics** → Check infrastructure health
6. **Compare with baselines** → Identify improvements/regressions
7. **Generate hypotheses** → Based on patterns detected

## Key Metrics Reference

### Training Metrics
- `benchmark/loss`: Policy loss (should decrease)
- `benchmark/kl`: KL divergence (should stay bounded)
- `benchmark/reward_mean`: Average reward (should increase)
- `benchmark/reward_std`: Reward variance (monitor for stability)
- `benchmark/grad_norm`: Gradient magnitude (watch for spikes)

### Performance Metrics
- `benchmark/rollout_time_s`: Time per rollout batch
- `benchmark/training_time_s`: Time per training step
- `benchmark/pipeline_overlap_s`: Time saved from parallelism
- `benchmark/trajectories`: Trajectories per step

### Quality Metrics
- `extraction/rate`: Success rate of order extraction
- `extraction/orders_expected`: Expected orders per step
- `extraction/orders_extracted`: Successfully extracted orders
- `extraction/empty_responses`: Failed extractions

### League Metrics
- `elo/challenger`: Current challenger Elo rating
- `elo/win_rate`: Win rate against gatekeepers
- `league/best_elo`: Best Elo in league
- `league/num_checkpoints`: Number of checkpoints in league

### Power Law Metrics
- `power_law/cumulative_simulated_years`: Total compute used
- `power_law/simulated_years_per_step`: Compute per step
- `power_law/reward_at_compute`: Reward at given compute level

## Output Format

All CLI commands support `--output-format json` for programmatic parsing. Use this format when:
- Building automated analysis pipelines
- Comparing multiple runs
- Generating reports
- Creating visualizations

For human-readable output, use the default table format or omit `--output-format`.
