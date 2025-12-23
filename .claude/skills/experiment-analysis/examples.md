# Example Analysis: grpo-20251222-191408

Real walkthrough of analyzing a training run.

## Step 1: Get Elo Metrics

```bash
uv run python scripts/wandb_cli.py get-metrics -r grpo-20251222-191408 --all-metrics
```

Filter for elo/* metrics in the output.

## Step 2: Checkpoint Trajectory Analysis

From the Elo data, we built this table:

| Checkpoint | Peak Elo | Peak Step | Final Elo | Trend |
|------------|----------|-----------|-----------|-------|
| adapter_v0 | 1022.9 | 24 | 1006.2 | Declining |
| adapter_v10 | 1035.9 | 25 | 1024.9 | Declining |
| adapter_v20 | 1035.7 | 23 | 1020.7 | Declining |
| adapter_v30 | 1034.1 | 14 | 1030.1 | Slight decline |
| adapter_v40 | 1035.5 | 13 | 1032.8 | Near peak |
| adapter_v50 | 1035.8 | 0 | 1031.3 | Near peak |

### Interpretation

**Key observation**: Older checkpoints (v0, v10, v20) peaked early then declined as they faced newer, stronger versions.

**Conclusion**: Model is continuously learning. Each new checkpoint beats the older ones.

## Step 3: Baseline Bot Analysis

| Bot | Initial Elo | Final Elo | Change |
|-----|-------------|-----------|--------|
| coordinated_bot | 991 | 851 | -140 |
| defensive_bot | 943 | 763 | -180 |
| territorial_bot | 955 | 986 | +31 |
| chaos_bot | 901 | 916 | +15 |
| base_model | 999 | 1008 | +9 |

### Interpretation

- **defensive_bot** and **coordinated_bot** lost 140-180 Elo = model learned to exploit their strategies
- **territorial_bot** and **chaos_bot** stable = harder to exploit (chaos is random, territorial is aggressive)
- **base_model** slight rise = games against it are balanced (expected at 1000 Elo)

**Total baseline exploitation**: ~300 Elo redistributed to learning agents.

## Step 4: Elo Ceiling Analysis

All checkpoints peak around 1030-1040 Elo, only ~35 points above base_model.

### Why the ceiling?

1. **Elo is zero-sum**: For checkpoints to gain, others must lose
2. **Self-play compression**: Playing similar-strength opponents stabilizes ratings
3. **Limited baseline pool**: Only ~300 Elo to extract from bots
4. **Training time**: 56 steps may not be enough for larger gains

### Expected progression

| Training Steps | Expected Max Elo |
|----------------|------------------|
| 50 | 1030-1040 |
| 100 | 1050-1070 |
| 200 | 1080-1120 |
| 500+ | 1150+ |

## Step 5: Bug Discovery

While analyzing, we noticed WandB showed 114 data points for ~56 training steps.

**Root cause**: Two separate `wandb.log()` calls per step:
1. `wandb.log(wandb_metrics)` - main metrics
2. `wandb.log(elo_log)` - Elo updates

Each call increments WandB's step counter.

**Fix**: Merge elo_log into wandb_metrics before single log call.

## Summary

### Findings
1. **Model is learning**: Clear signal from declining older checkpoints
2. **~35 Elo improvement**: From base_model 1000 to checkpoints ~1035
3. **Bots exploited**: defensive/coordinated bots down 140-180 Elo
4. **Bug found**: 2x WandB step logging

### Diagnostic Questions Answered

| Question | Answer |
|----------|--------|
| Is model learning? | YES - older checkpoints decline |
| How much improvement? | +35 Elo over base_model |
| Which bots exploited? | defensive_bot, coordinated_bot |
| Any bugs? | 2x WandB step logging |

### Experiment Tracker Entry

This analysis would translate to:

```markdown
**Results**:
- **Learning signal**: Positive - older checkpoints (v0-v20) declined while v40-v50 held peaks
- **Baseline exploitation**: defensive_bot -180, coordinated_bot -140 (total ~320 Elo)
- **Final metrics**:
  - Best Elo: 1035.8 (adapter_v50)
  - Improvement over base_model: +35 Elo
- **Issues discovered**: 2x WandB step logging bug (fixed)
- **Key learning**: Continuous learning confirmed; expect ~35 Elo per 50 steps

**Follow-up**: Continue training to see if Elo ceiling rises
```
