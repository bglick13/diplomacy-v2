---
name: experiment-analysis
description: Analyze GRPO training runs for learning dynamics and pipeline performance. Use when diagnosing training issues, reviewing Elo progression, checking throughput, or updating experiment results.
---

# Experiment Analysis

Diagnose GRPO training runs using WandB metrics and Axiom logs.

## Quick Reference

| Question | Command |
|----------|---------|
| **Full Elo analysis** | `uv run python .claude/skills/experiment-analysis/analyze_elo.py <run>` |
| Is model learning? | `uv run python scripts/wandb_cli.py get-metrics -r <run> --all-metrics` |
| Rollout throughput? | `uv run python scripts/axiom_cli.py rollout-timing --last 6h` |
| Any errors? | `uv run python scripts/axiom_cli.py errors --last 1h` |
| Extraction rate? | `uv run python scripts/axiom_cli.py extraction-stats --last 24h` |
| System health? | `uv run python scripts/axiom_cli.py health --last 1h` |

## Tools Overview

### WandB CLI (`scripts/wandb_cli.py`)
Training metrics and Elo ratings. Use for:
- Elo trajectory analysis (learning signal)
- Reward/loss curves
- KL divergence and grad norm

### Axiom CLI (`scripts/axiom_cli.py`)
Real-time logs and events. Use for:
- Rollout timing and throughput
- Inference engine performance
- Error monitoring
- Order extraction stats

## Detailed Guides

- [Learning Dynamics](learning-dynamics.md) - Elo, rewards, KL analysis
- [Pipeline Performance](pipeline-performance.md) - Throughput, timing, errors
- [Experiment Tracker Guide](experiment-tracker-guide.md) - Updating docs/experiment-tracker.md
- [Examples](examples.md) - Real analysis walkthrough

## Key Metrics

### Learning Signal
| Metric | Good Sign | Bad Sign |
|--------|-----------|----------|
| Older checkpoint Elo | Declining | Stable |
| Baseline bot Elo | Declining (exploited) | Rising |
| reward_mean | Stable or rising | Declining |
| KL divergence | Stable <0.1 | Spikes >0.2 |

### Performance
| Metric | Target | Action if Miss |
|--------|--------|----------------|
| Rollout p95 duration | <120s | Check inference engine |
| Extraction rate | >95% | Check logits processor |
| Error rate | <1% | Check Axiom errors |
| Grad norm | <50 | Policy may be unstable |
