# Axiom Dashboard Queries for Diplomacy GRPO

This document contains APL queries for building a Rollout Health Dashboard in Axiom.

## Dashboard Overview

After deploying with the new observability layer, you'll be able to track:
- Rollout success/failure rates
- Order extraction quality (empty responses, partial extractions)
- Inference latency
- Per-power performance

---

## Key Health Metrics

### 1. Empty Orders Alert (Critical Bug Detection)

This query catches the exact bug we encountered - LLM producing no parseable orders:

```apl
['diplomacy']
| where event == "orders_empty"
| summarize 
    empty_count = count(),
    affected_powers = dcount(power_name),
    affected_phases = dcount(phase)
  by bin(_time, 1h)
| order by _time desc
```

### 2. Order Extraction Success Rate

Track the overall health of order parsing:

```apl
['diplomacy']
| where event in ("orders_extracted", "orders_empty", "orders_partial")
| extend status = case(
    event == "orders_empty", "empty",
    event == "orders_partial", "partial",
    "full"
  )
| summarize count() by status, bin(_time, 1h)
```

### 3. Extraction Rate by Power

Some powers may have more complex positions leading to worse extraction:

```apl
['diplomacy']
| where event in ("orders_extracted", "orders_empty", "orders_partial")
| summarize 
    total = count(),
    empty = countif(event == "orders_empty"),
    partial = countif(event == "orders_partial")
  by power_name
| extend extraction_rate = 1.0 - (todouble(empty + partial) / todouble(total))
| order by extraction_rate asc
```

---

## Rollout Performance

### 4. Rollout Completion Rate

```apl
['diplomacy']
| where event in ("rollout_start", "rollout_complete", "rollout_error")
| summarize count() by event, bin(_time, 1h)
```

### 5. Rollout Duration Distribution

```apl
['diplomacy']
| where event == "rollout_complete"
| summarize 
    p50 = percentile(total_duration_ms, 50),
    p90 = percentile(total_duration_ms, 90),
    p99 = percentile(total_duration_ms, 99),
    avg_duration = avg(total_duration_ms)
  by bin(_time, 1h)
```

### 6. Trajectories per Rollout

```apl
['diplomacy']
| where event == "rollout_complete"
| summarize 
    avg_trajectories = avg(trajectories_count),
    total_trajectories = sum(trajectories_count)
  by bin(_time, 1h)
```

---

## Inference Performance

### 7. Inference Latency

```apl
['diplomacy']
| where event == "span_duration" and span_name contains "Inference"
| summarize 
    p50 = percentile(duration_ms, 50),
    p90 = percentile(duration_ms, 90),
    avg_latency = avg(duration_ms)
  by bin(_time, 5m)
```

### 8. Inference Errors

```apl
['diplomacy']
| where event == "span_duration" and status == "error"
| summarize count() by span_name, bin(_time, 1h)
```

---

## Error Analysis

### 9. Recent Errors (Last 24h)

```apl
['diplomacy']
| where event == "rollout_error"
| project _time, rollout_id, error, phase
| order by _time desc
| take 50
```

### 10. Empty Orders Details (Debugging)

```apl
['diplomacy']
| where event == "orders_empty"
| project _time, rollout_id, power_name, phase, raw_response_length, raw_response_preview, expected_count
| order by _time desc
| take 100
```

### 10b. Empty Orders by Phase Type

```apl
['diplomacy']
| where event == "orders_empty"
| extend phase_type = case(
    phase contains "MOVEMENT", "MOVEMENT",
    phase contains "ADJUSTMENTS", "ADJUSTMENTS",
    phase contains "RETREAT", "RETREAT",
    "OTHER"
  )
| summarize count() by phase_type
```

---

## Summary Dashboard Metrics

### 11. Overall Health Score (Last Hour)

```apl
['diplomacy']
| where _time > ago(1h)
| summarize 
    rollouts_started = countif(event == "rollout_start"),
    rollouts_completed = countif(event == "rollout_complete"),
    rollouts_errored = countif(event == "rollout_error"),
    empty_orders = countif(event == "orders_empty"),
    partial_orders = countif(event == "orders_partial"),
    total_extractions = countif(event in ("orders_extracted", "orders_empty", "orders_partial"))
| extend 
    completion_rate = todouble(rollouts_completed) / todouble(rollouts_started),
    extraction_quality = 1.0 - (todouble(empty_orders + partial_orders) / todouble(total_extractions))
```

---

## Creating the Dashboard

1. Go to Axiom dashboard: https://app.axiom.co
2. Create a new dashboard named "Diplomacy GRPO - Rollout Health"
3. Add charts using the queries above:

**Recommended Layout:**
- Row 1: Health Score (stat), Completion Rate (stat), Empty Orders Count (stat)
- Row 2: Order Extraction by Status (stacked bar), Rollout Duration (line)
- Row 3: Extraction Rate by Power (bar), Recent Errors (table)
- Row 4: Inference Latency (line), Empty Orders Timeline (line)

---

## Power Laws Experiment Queries

These queries help monitor and analyze the Power Laws scaling experiment.

### 12. Training Progress by Run

Track reward progression across all power-laws runs:

```apl
['diplomacy']
| where event == "training_step"
| where run_name startswith "power-laws-"
| extend config_type = case(
    run_name contains "baseline", "A: Baseline",
    run_name contains "deep", "B: Deep Search",
    run_name contains "broad", "C: Broad Search",
    "unknown"
  )
| summarize avg_reward = avg(reward_mean) by config_type, step
| order by config_type, step
```

### 13. Compute Efficiency Comparison

Compare simulated years per wall-clock hour:

```apl
['diplomacy']
| where event == "rollout_complete"
| where rollout_id startswith "power-laws-"
| summarize 
    total_rollouts = count(),
    total_trajectories = sum(trajectories_count),
    avg_duration_ms = avg(total_duration_ms)
  by bin(_time, 1h)
| extend sim_years_per_hour = (total_trajectories * 2) / (avg_duration_ms / 3600000)
```

### 14. Reward Distribution by Config

```apl
['diplomacy']
| where event == "training_step"
| where run_name startswith "power-laws-"
| extend config = case(
    run_name contains "baseline", "baseline",
    run_name contains "deep", "deep-search",
    run_name contains "broad", "broad-search",
    "unknown"
  )
| summarize 
    p25 = percentile(reward_mean, 25),
    p50 = percentile(reward_mean, 50),
    p75 = percentile(reward_mean, 75),
    final_reward = argmax(step, reward_mean)
  by config
```

### 15. Rollout Health by Experiment

```apl
['diplomacy']
| where event in ("rollout_complete", "rollout_error")
| where rollout_id startswith "power-laws-"
| summarize 
    completed = countif(event == "rollout_complete"),
    errors = countif(event == "rollout_error")
  by bin(_time, 30m)
| extend success_rate = todouble(completed) / todouble(completed + errors)
```

### 16. Power Laws Summary (Last Run)

```apl
['diplomacy']
| where event == "training_step"
| where run_name startswith "power-laws-"
| where _time > ago(6h)
| extend config = case(
    run_name contains "baseline", "A: Baseline (1x)",
    run_name contains "deep", "B: Deep (2x)",
    run_name contains "broad", "C: Broad (2x)",
    "unknown"
  )
| summarize 
    max_step = max(step),
    final_reward = argmax(step, reward_mean),
    avg_loss = avg(loss),
    avg_kl = avg(kl)
  by config, run_name
| order by config
```

---

## Alert Recommendations

Set up Axiom monitors for:

1. **Critical: Empty Orders Spike**
   - Condition: `orders_empty` count > 10 in 15 minutes
   - This indicates the logits processor or prompt may be broken

2. **Warning: High Partial Orders**
   - Condition: `orders_partial` > 30% of extractions
   - This indicates prompt quality degradation

3. **Critical: Rollout Errors**
   - Condition: `rollout_error` count > 5 in 1 hour
   - This indicates infrastructure issues

4. **Power Laws: Run Divergence**
   - Condition: Reward variance across configs > threshold after step 50
   - This indicates one config is significantly outperforming others
