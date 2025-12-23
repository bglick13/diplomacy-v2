# Experiment Tracker Guide

How to update `docs/experiment-tracker.md` with experiment results.

## When to Update

1. **After run completes** - Record final metrics and conclusions
2. **After run cancelled** - Document why and what was learned
3. **Significant mid-run finding** - If you discover something important
4. **Designing follow-up** - When creating next experiment based on learnings

## File Location

```
docs/experiment-tracker.md
```

## Experiment Entry Format

Each experiment should have:

```markdown
## Exp N: [Descriptive Title]

**Hypothesis**: What we expect to learn

**Key Config Changes**:
- `--flag-name value`: Description of change

**Command**:
```bash
python scripts/launch_training.py \
  --flag1 value1 \
  --flag2 value2
```

**Status**: `planned` | `running` | `completed` | `failed` | `concluded`
- wandb: [link if available]
- run name: grpo-YYYYMMDD-HHMMSS

**Results**: _pending_ (or fill in findings)

**Follow-up**: → Exp N+1 (if applicable)
```

## Results Template

When filling in results, use this structure:

```markdown
**Results**:
- **Learning signal**: [Did model learn?] - e.g., "Checkpoints v40-v50 reached 1035 Elo, older declined"
- **Baseline exploitation**: [Bot Elo changes] - e.g., "defensive_bot -180, coordinated_bot -140"
- **Final metrics**:
  - Best Elo: [value] (improvement over base_model: +[delta])
  - reward_mean: [start] → [peak] → [final]
  - Extraction rate: [percentage]
- **Issues discovered**: [Any bugs or problems]
- **Root cause** (if applicable): [What went wrong/right]
- **Key learning**: [What we learned for future experiments]

**Follow-up**: → Exp N (brief description of next experiment)
```

## Metrics to Include

### Required
- **Elo spread**: Best checkpoint vs base_model
- **Baseline exploitation**: Total Elo lost by bots
- **Training stability**: Any reward decline, KL spikes

### Optional (if relevant)
- Cache hit rate
- Extraction rate
- Rollout throughput
- Specific errors encountered

## Example: Completed Experiment

```markdown
## Exp 1: League Training from Scratch (Baseline)

**Hypothesis**: League training with PFSP matchmaking will develop stronger play than pure self-play.

**Key Config Changes**:
- `--rollout-horizon-years 5`: Medium horizon
- `--winner-threshold-sc 7`: Moderate win threshold

**Command**:
```bash
python scripts/launch_training.py \
  --rollout-horizon-years 5 \
  --winner-threshold-sc 7 \
  --total-steps 250
```

**Status**: `concluded` (cancelled at step ~10)
- wandb: https://wandb.ai/bglick13/diplomacy-grpo/runs/01dp7f4x
- run name: grpo-20251218-114516

**Results**:
- **Learning signal**: Negative - reward_mean declined from 14→8 in first 8 steps
- **Baseline exploitation**: Not measured (cancelled early)
- **Final metrics**:
  - win_bonus_rate: 20%→3%
  - avg_sc_count: 5.0→3.5
- **Root cause**: KL penalty (β=0.04) too aggressive early, pushing model toward base behavior
- **Key learning**: Need KL warmup to allow early exploration

**Follow-up**: → Exp 6 (KL Warmup)
```

## Follow-up Experiment Design

When results suggest a follow-up:

1. **State the hypothesis** clearly based on learnings
2. **Change one variable** from the previous experiment
3. **Predict expected outcome**

Example:
```markdown
## Exp 6: KL Warmup for Training Stability

**Hypothesis**: Early training reward decline is caused by aggressive KL penalty.
Warming up KL from 0 will allow early learning without penalty.

**Key Config Changes**:
- `--kl-beta-warmup-steps 20`: Linear warmup from 0 to full beta
- (All other settings same as Exp 1)
```

## Status Values

| Status | Meaning |
|--------|---------|
| `planned` | Not yet started |
| `running` | Currently executing |
| `completed` | Finished successfully |
| `failed` | Crashed or errored |
| `concluded` | Intentionally stopped (with learnings) |

## Quick Update Workflow

1. Run analysis:
   ```bash
   uv run python .claude/skills/experiment-analysis/analyze_elo.py <run-name>
   ```

2. Get key metrics from WandB:
   ```bash
   uv run python scripts/wandb_cli.py get-run -r <run-name>
   ```

3. Check for issues:
   ```bash
   uv run python scripts/axiom_cli.py errors --last 24h
   ```

4. Update `docs/experiment-tracker.md` with findings

5. Create follow-up experiment entry if needed
