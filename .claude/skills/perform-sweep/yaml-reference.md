# Sweep YAML Reference

Complete schema documentation for `sweep.yaml` configuration files.

## File Location

```
experiments/sweeps/<sweep-name>/sweep.yaml
```

## Full Schema

```yaml
# ============================================================================
# METADATA (required)
# ============================================================================
metadata:
  # Unique identifier for this sweep (required)
  # Used for state tracking and WandB grouping
  name: "my-sweep-name"

  # Human-readable description (optional)
  description: "What this sweep tests"

  # The hypothesis being tested (optional but recommended)
  # Document your prediction before running
  hypothesis: |
    Multi-line hypothesis text.
    What do you expect to happen and why?

  # Prefix for WandB experiment tags (required)
  # Each run gets tag: {prefix}-{run_id}
  experiment_tag_prefix: "my-sweep"

  # Creation date (optional, defaults to today)
  created: "2024-12-24"

  # Author (optional)
  author: "your-name"

# ============================================================================
# DEFAULTS (optional)
# ============================================================================
# These values are applied to ALL runs unless overridden.
# Can be any ExperimentConfig field.
defaults:
  total_steps: 100
  learning_rate: 5e-6
  # ... any other ExperimentConfig field

# ============================================================================
# RUNS (required)
# ============================================================================
# Map of run IDs to configurations.
# Run IDs are typically letters (A, B, C, D) but can be any string.
runs:
  A:
    # Short name for this run (required)
    name: "baseline"

    # Description of what this run tests (optional)
    description: "Control group with default settings"

    # Config overrides for this run (optional)
    # These override defaults
    config:
      # Any ExperimentConfig field
      rollout_horizon_years: 4
      experiment_tag: "${metadata.experiment_tag_prefix}-A"

  B:
    name: "treatment"
    description: "With longer horizon"
    config:
      rollout_horizon_years: 8
      experiment_tag: "${metadata.experiment_tag_prefix}-B"

# ============================================================================
# ANALYSIS (optional)
# ============================================================================
analysis:
  # Primary metric to compare (default: trueskill/display_rating)
  primary_metric: "trueskill/display_rating"

  # Additional metrics to track
  secondary_metrics:
    - "game/win_bonus_rate"
    - "benchmark/reward_mean"
    - "game/avg_sc_count"

  # Expected ordering of runs by performance
  expected_ranking: ["B", "A"]

  # Success criteria for the sweep
  success_criteria:
    - "B achieves higher TrueSkill than A"
    - "No training instability"
```

## Variable Interpolation

Use `${path.to.field}` to reference other fields:

```yaml
metadata:
  name: "my-sweep"
  experiment_tag_prefix: "my-sweep"

runs:
  A:
    config:
      # Resolves to "my-sweep-A"
      experiment_tag: "${metadata.experiment_tag_prefix}-A"
```

## Common ExperimentConfig Fields

### Training Loop
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `total_steps` | int | 250 | Number of training steps |
| `num_groups_per_step` | int | 16 | Rollout groups per step |
| `samples_per_group` | int | 8 | Trajectory samples per group |
| `learning_rate` | float | 5e-6 | AdamW learning rate |

### Rollout Settings
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `rollout_horizon_years` | int | 4 | Short rollout horizon (years) |
| `rollout_long_horizon_years` | int | 6 | Long rollout horizon |
| `rollout_long_horizon_chance` | float | 0.2 | Probability of long horizon |

### Reward/Scoring
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `step_reward_weight` | float | 0.8 | Weight for per-step rewards |
| `final_reward_weight` | float | 0.2 | Weight for final outcome |
| `win_bonus` | float | 50.0 | Bonus for winning |
| `winner_threshold_sc` | int | 5 | Min SCs for win bonus |
| `use_strategic_step_scoring` | bool | false | Position-based step scoring |
| `use_position_based_scoring` | bool | true | Position-based final scoring |

### Reward Shaping
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `leader_gap_penalty_weight` | float | 0.3 | Penalty for falling behind leader |
| `balance_bonus_weight` | float | 0.2 | Bonus for balanced games |
| `leader_gap_threshold` | int | 3 | SC gap before penalty applies |

### KL Settings
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `kl_beta` | float | 0.0 | KL penalty coefficient |
| `kl_beta_warmup_steps` | int | 20 | Steps to warmup KL |
| `kl_target` | float | None | Target KL for adaptive control |

### League Training
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `league_training` | bool | true | Enable league training |
| `checkpoint_every_n_steps` | int | 10 | Checkpoint frequency |

## Examples

### A/B Test (2 runs)

```yaml
metadata:
  name: "learning-rate-ab"
  experiment_tag_prefix: "lr-test"

defaults:
  total_steps: 100

runs:
  A:
    name: "low-lr"
    config:
      learning_rate: 1e-6
      experiment_tag: "${metadata.experiment_tag_prefix}-A"
  B:
    name: "high-lr"
    config:
      learning_rate: 1e-5
      experiment_tag: "${metadata.experiment_tag_prefix}-B"
```

### 2x2 Factorial Design (4 runs)

```yaml
metadata:
  name: "horizon-scoring-ablation"
  experiment_tag_prefix: "ablation"

runs:
  A:
    name: "short-sc"
    description: "Short horizon, SC-based scoring"
    config:
      rollout_horizon_years: 4
      use_strategic_step_scoring: false
  B:
    name: "long-sc"
    description: "Long horizon, SC-based scoring"
    config:
      rollout_horizon_years: 8
      use_strategic_step_scoring: false
  C:
    name: "short-strategic"
    description: "Short horizon, strategic scoring"
    config:
      rollout_horizon_years: 4
      use_strategic_step_scoring: true
  D:
    name: "long-strategic"
    description: "Long horizon, strategic scoring"
    config:
      rollout_horizon_years: 8
      use_strategic_step_scoring: true
```

### KL Sweep (3 runs)

```yaml
metadata:
  name: "kl-beta-sweep"
  experiment_tag_prefix: "kl-sweep"

defaults:
  total_steps: 100
  kl_beta_warmup_steps: 10

runs:
  A:
    name: "no-kl"
    config:
      kl_beta: 0.0
  B:
    name: "low-kl"
    config:
      kl_beta: 0.02
  C:
    name: "high-kl"
    config:
      kl_beta: 0.08
```
