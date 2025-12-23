# Learning Dynamics Analysis

Diagnose whether the model is learning and identify training issues.

## Elo Analysis

### Pull Elo Metrics

```bash
# Get all Elo metrics for a run
uv run python scripts/wandb_cli.py get-metrics -r <run-name> --all-metrics

# Use the helper script for full analysis
uv run python .claude/skills/experiment-analysis/analyze_elo.py <run-name>
```

### Interpreting Checkpoint Trajectories

| Pattern | Interpretation |
|---------|----------------|
| Older checkpoints decline, newer hold peak | **Model learning** - newer versions beat older |
| All checkpoints similar Elo | **Plateau** - no improvement between versions |
| Latest checkpoint declining | **Regression** - recent training made it worse |
| Erratic swings | **Instability** - training too aggressive |

**Key insight**: In a closed Elo pool, older checkpoints *should* decline as they get beaten by newer, better versions. This is the primary signal that learning is occurring.

### Baseline Exploitation

Baseline bots (chaos_bot, defensive_bot, etc.) start around 900-1000 Elo. If the model is learning:

| Bot Elo Trend | Meaning |
|---------------|---------|
| Declining | Model exploiting bot weaknesses |
| Stable | Model not learning to beat this bot |
| Rising | Bot's strategy is effective (unlikely) |

**Total baseline exploitation** = sum of Elo lost by all bots. Higher = more learning.

### Elo Ceiling Analysis

Elo is zero-sum within a closed pool. The theoretical max Elo depends on:

1. **Baseline floor**: How low can bots be pushed? ~700-800 is typical floor.
2. **Number of checkpoints**: More checkpoints = Elo spread thinner.
3. **Self-play proportion**: High self-play stabilizes Elo near peers.

Expected improvement over base_model:
- Early training (50 steps): 20-50 Elo
- Mid training (200 steps): 50-100 Elo
- Extended training (500+ steps): 100-200 Elo

## Reward Signals

### Key Metrics

```bash
uv run python scripts/wandb_cli.py get-metrics -r <run> -m benchmark/reward_mean benchmark/reward_std
```

| Metric | What to Watch |
|--------|---------------|
| `reward_mean` | Should be stable or rising |
| `reward_std` | High variance early OK, should stabilize |
| `win_bonus_rate` | Fraction achieving decisive wins |
| `avg_sc_count` | Territory control (7+ = strong) |

### Early Training Patterns

| Pattern | Cause | Action |
|---------|-------|--------|
| Initial decline (14→8) | KL penalty too aggressive | Add KL warmup |
| Flat from start | Learning rate too low | Increase LR |
| Wild oscillation | LR too high or gradient issues | Reduce LR, check grad_norm |

## KL Divergence

### Metrics

```bash
uv run python scripts/wandb_cli.py get-metrics -r <run> -m kl/mean kl/beta kl/warmup_progress
```

| Metric | Healthy Range | Warning Sign |
|--------|---------------|--------------|
| `kl/mean` | 0.01-0.05 | >0.1 (policy drifting too fast) |
| `kl/beta` | Configured value | N/A |
| `kl/warmup_progress` | 0→1 over warmup | N/A |

### KL Issues

| Issue | Symptom | Fix |
|-------|---------|-----|
| Policy collapse | KL→0, reward flat | Reduce beta, increase exploration |
| Runaway divergence | KL>0.2, erratic rewards | Increase beta, add clipping |
| Early instability | High KL in first 10 steps | Add KL warmup |

## Diagnostic Decision Tree

```
Is model learning?
│
├─ Check: Are older checkpoints declining in Elo?
│  ├─ YES → Model improving (newer > older)
│  └─ NO → Plateau or regression
│
├─ Check: Are baselines losing Elo?
│  ├─ YES → Model exploiting weaknesses
│  └─ NO → Not generalizing beyond self-play
│
├─ Check: Is reward_mean stable or rising?
│  ├─ YES → Training signal healthy
│  └─ NO → Check KL, check opponent distribution
│
└─ Check: Is KL divergence stable <0.1?
   ├─ YES → Policy updates appropriate
   └─ NO → Adjust beta or add warmup
```

## Common Issues

### "Reward declined in first 10 steps"
**Cause**: KL penalty pushing model back to base before it learns.
**Fix**: Add `--kl-beta-warmup-steps 20` to config.

### "All checkpoints have same Elo"
**Cause**: Not enough training steps between checkpoints OR model plateaued.
**Fix**: Train longer OR check if baselines are being beaten (if not, model isn't improving).

### "Baselines rising in Elo"
**Cause**: Model regressing OR evaluation noise.
**Fix**: Rare - usually indicates training issue. Check for gradient problems.
