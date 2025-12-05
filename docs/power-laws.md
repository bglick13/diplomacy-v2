# Power Laws Experiment: Compute vs. Insight

This document describes the "Mini Power Laws" experiment to determine whether **scaling** (more compute) or **engineering** (reward shaping) is the right approach to teach defensive play in Diplomacy.

## The Hypothesis

There are two competing approaches to fix "suicidal rushing" behavior:

| Approach | Method | Example |
|----------|--------|---------|
| **Engineering** | Add hand-crafted penalties | "Penalize 0.5 points for leaving Paris empty" (model learns to camp) |
| **Scaling** | Increase lookahead horizon | "Look 4 years ahead. If I leave Paris empty, I die (-1.0). If I defend, I survive (+4.0)." (model learns naturally) |

**Current Bottleneck**: The scaling approach may be limited by:
- **Horizon** (2 years) - Too short to see consequences
- **Sample Variance** (N=8) - Too noisy to find the signal

## The Experiment

We test 3 configurations at constant training steps (100) to isolate the effect of compute allocation:

### Run A: Baseline (Fast & Loose)
```
rollout_horizon_years: 2
samples_per_group: 8
Total Compute: 1×
```
**Expectation**: High reward initially, but plateaus as "suicidal rushing" hits a ceiling against random opponents.

### Run B: Deep Search (Time Scaling)
```
rollout_horizon_years: 4
samples_per_group: 8
Total Compute: 2×
```
**Theory**: Doubling the horizon gives the model the "causal link" between bad tactics and elimination.

**Success Metric**: Does it learn slower at first (harder credit assignment) but reach a higher peak?

### Run C: Broad Search (Variance Scaling)
```
rollout_horizon_years: 2
samples_per_group: 16
Total Compute: 2×
```
**Theory**: Doubling the group size stabilizes the gradient. It filters out "I rushed and got lucky" vs "I rushed and got punished."

**Success Metric**: Does the reward curve become smoother and steeper?

## Running the Experiment

### Prerequisites

1. Deploy the Modal app:
   ```bash
   modal deploy app.py
   ```

2. Ensure WandB is configured:
   ```bash
   wandb login
   ```

### Quick Test (Smoke)
Run a shortened version to validate setup:
```bash
python scripts/launch_sweep.py --steps 10 --run A
```

### Full Experiment

#### Option 1: Sequential (GPU-Constrained)
```bash
# Run all three configurations one after another
python scripts/launch_sweep.py --steps 100
```

#### Option 2: Parallel (Multiple GPUs Available)
```bash
# Launch all configurations simultaneously
python scripts/launch_sweep.py --steps 100 --parallel
```

#### Option 3: Individual Runs
```bash
# Run specific configurations
python scripts/launch_sweep.py --steps 100 --run A  # Baseline
python scripts/launch_sweep.py --steps 100 --run B  # Deep Search
python scripts/launch_sweep.py --steps 100 --run C  # Broad Search
```

### Command Reference
```
usage: launch_sweep.py [-h] [--run {A,B,C,all}] [--steps STEPS] [--groups GROUPS]
                       [--lr LR] [--dry-run] [--no-warmup] [--parallel]

Options:
  --run {A,B,C,all}  Which configuration to run (default: all)
  --steps STEPS      Training steps per run (default: 100)
  --groups GROUPS    Rollout groups per step (default: 8)
  --lr LR            Learning rate (default: 1e-5)
  --dry-run          Show config without launching
  --no-warmup        Skip InferenceEngine warmup
  --parallel         Run all configs in parallel
```

## Visualizing Results in WandB

### Creating the Power Law Plot

After the runs complete:

1. Go to your WandB project: `diplomacy-grpo`
2. Filter runs by names starting with `power-laws-`
3. Create a custom chart with:
   - **X-Axis**: `power_law/cumulative_simulated_years`
   - **Y-Axis**: `power_law/reward_at_compute` (or `benchmark/reward_mean`)
4. Group by run name to compare configurations

### Expected Compute Costs

| Run | Steps | Sim Years/Step | Total Sim Years |
|-----|-------|----------------|-----------------|
| A (Baseline) | 100 | 128 | 12,800 |
| B (Deep) | 100 | 256 | 25,600 |
| C (Broad) | 100 | 256 | 25,600 |

*Note: Sim Years = groups × samples × horizon = 8 × 8 × 2 = 128 for baseline*

## Decision Rules

After comparing the runs at Step 100, interpret results as follows:

### If Run B (Horizon) Wins:
✅ **Simple reward works, you just need LONGER HORIZONS.**
- The model can learn defensive play naturally with more lookahead
- Don't hand-code defense; just pay for the simulation time
- **Action**: Increase `rollout_horizon_years` in production

### If Run C (Samples) Wins:
✅ **Simple reward works, you just need BETTER BASELINES.**
- The noise of Diplomacy requires massive groups to find the signal
- More samples stabilize gradients and reduce variance
- **Action**: Increase `samples_per_group` in production

### If Neither Wins Significantly Over A:
⚠️ **Scaling alone cannot escape the local optimum.**
- The simple reward is insufficient
- "Reward Hacking" (suicidal rushing) is a stable local minimum
- **Action**: Go back to reward engineering
  - Add defensive bonuses
  - Add position-based penalties
  - Consider curriculum learning

## Theory: Why This Works

The Power Law hypothesis suggests that model performance scales predictably with compute. By controlling the "compute" variable (simulated years) while varying *how* that compute is allocated (horizon vs. samples), we can identify the binding constraint:

1. **If horizon matters more**: Credit assignment over time is the bottleneck
2. **If samples matter more**: Signal-to-noise ratio is the bottleneck
3. **If neither matters**: The reward function itself needs work

This is a principled way to debug RL systems before committing to expensive reward engineering.

## Related Files

- `scripts/launch_sweep.py` - Experiment launcher
- `src/utils/config.py` - Configuration with `simulated_years` properties
- `app.py` - `train_grpo` function with power law metrics
