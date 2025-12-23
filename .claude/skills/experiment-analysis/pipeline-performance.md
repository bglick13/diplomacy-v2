# Pipeline Performance Analysis

Monitor throughput, timing, and system health.

## Rollout Performance

### Check Timing

```bash
uv run python scripts/axiom_cli.py rollout-timing --last 6h
```

Output includes:
- `avg_duration_s`: Mean rollout time
- `p50_duration`, `p95_duration`: Latency percentiles
- `total_trajectories`: Throughput indicator

### Targets

| Metric | Target | Action if Miss |
|--------|--------|----------------|
| p50 duration | <60s | Check inference engine |
| p95 duration | <120s | Check for hanging games |
| Trajectories/rollout | 7+ | Check horizon config |

### Throughput Calculation

```
trajectories_per_hour = (rollouts_per_hour) × (trajectories_per_rollout)
                      = (3600 / avg_duration_s) × 7
```

Target: >400 trajectories/hour for efficient training.

## Inference Engine

### Check Stats

```bash
uv run python scripts/axiom_cli.py inference-stats --last 1h

# By adapter (to find slow adapters)
uv run python scripts/axiom_cli.py inference-stats --last 1h --by-adapter
```

### Key Metrics

| Metric | Target | Issue if Low |
|--------|--------|--------------|
| `avg_output_tps` | >50 tok/s | GPU bottleneck |
| `avg_input_tps` | >500 tok/s | Tokenization or batching |
| `avg_batch_size` | >4 | Not enough parallelism |

### Troubleshooting

| Issue | Check | Fix |
|-------|-------|-----|
| Low TPS | GPU utilization | Increase batch size |
| High adapter_load_time | LoRA loading | Pre-warm adapters |
| Low batch size | Rollout concurrency | Increase buffer_depth |

## Error Monitoring

### Recent Errors

```bash
uv run python scripts/axiom_cli.py errors --last 1h
uv run python scripts/axiom_cli.py errors --last 1h --severity error
```

### System Health

```bash
uv run python scripts/axiom_cli.py health --last 1h
```

Returns:
- `status`: healthy / degraded / unhealthy
- `error_rate`: % of events that are errors
- `rollouts.completion_rate`: % of started rollouts that complete
- `inference`: requests, avg duration, TPS

### Common Error Patterns

| Error | Cause | Fix |
|-------|-------|-----|
| `rollout_error` | Game simulation crashed | Check game engine logs |
| `orders_empty` | LLM generated no valid orders | Check logits processor |
| `inference_timeout` | vLLM hung | Restart inference engine |
| `adapter_load_failed` | Missing LoRA | Check volume mount |

## Order Extraction

### Stats

```bash
uv run python scripts/axiom_cli.py extraction-stats --last 24h

# By power (find problematic powers)
uv run python scripts/axiom_cli.py extraction-stats --last 24h --by-power
```

### Targets

| Metric | Target | Action if Miss |
|--------|--------|----------------|
| `extraction_rate` | >95% | Check logits processor |
| `empty` count | <5% | Check prompt format |
| `partial` count | <10% | Check unit parsing |

### By-Power Analysis

If one power has low extraction:
- Check if that power has more units (more complex orders)
- Check if prompts for that power are formatted correctly

## Game Outcomes

```bash
uv run python scripts/axiom_cli.py game-outcomes --last 24h

# By power
uv run python scripts/axiom_cli.py game-outcomes --last 24h --by-power
```

### Healthy Distribution

All 7 powers should have roughly equal:
- Game participation rate
- Average score
- Win rate

Imbalance suggests:
- Biased power assignment
- One power harder to play
- Training overfitting to certain positions

## Performance Dashboard

Quick health check sequence:

```bash
# 1. System health
uv run python scripts/axiom_cli.py health --last 1h

# 2. Any errors?
uv run python scripts/axiom_cli.py errors --last 1h --limit 10

# 3. Rollout throughput
uv run python scripts/axiom_cli.py rollout-timing --last 1h

# 4. Inference performance
uv run python scripts/axiom_cli.py inference-stats --last 1h

# 5. Extraction quality
uv run python scripts/axiom_cli.py extraction-stats --last 1h
```
