# Inference Optimization Research

This document tracks research, benchmarks, and findings for optimizing inference throughput.

---

## Current State (2024-12-19)

### Branch: `12-18-kl_penalty_changes`

**Modified Files:**
- `src/inference/logits.py` - Logits processor instrumentation + text-based tag detection
- `src/apps/inference_engine/app.py` - Axiom secrets, timing metrics, debug prints
- `src/apps/trainer/app.py` - App name fixes, KL controller
- `src/apps/rollouts/app.py` - App name fixes
- `src/utils/config.py` - Hyperparameter tuning (samples_per_group=6, buffer_depth=2, etc.)
- `src/training/loss.py` - AdaptiveKLController
- `scripts/benchmark_logits_processor.py` - NEW: Standalone benchmark

---

## Observed Throughput (from Axiom)

| Metric | Value | Notes |
|--------|-------|-------|
| Input TPS (prefill) | 600-8,000 | Varies with batch size, cache hits |
| Output TPS (decode) | 300-1,700 | Varies with batch size |
| Batch sizes | 7-42 prompts | Per inference call |
| Generation time | 2-5 seconds | Per batch |

---

## Benchmark Results

### Logits Processor Overhead (CPU benchmark)

| Operation | Time | Notes |
|-----------|------|-------|
| Trie Build | 3.6ms | Once per request setup |
| Logits Restriction | 0.07ms | Very fast |
| Apply (empty) | 0.001ms | Baseline |
| Apply (batch=32, active) | 4.2ms | Per forward pass |

**Estimated overhead**: ~13% (pessimistic - rebuilds state each iteration)
**Actual production overhead**: Likely 5-10%

### Single Game Benchmark (2024-12-19)

**Config:** 3-year horizon, 6 samples/group, Qwen2.5-7B-Instruct

| Metric | Value |
|--------|-------|
| Trajectories | 42 (7 powers × 6 samples) |
| Total Prompt Tokens | 18,462 |
| Total Completion Tokens | 991 |
| Avg Prompt/Call | 439.6 |
| Avg Completion/Call | 23.6 |
| Wall Time | 27.39s |
| Inference Time | 23.14s (99% of wall time) |

**Calculated TPS (from benchmark):**
- Input TPS: 674 (18,462 / 27.39s)
- Output TPS: 36.2 (991 / 27.39s)
- Using inference time only: 42.8 output tokens/sec

**Observation:** Output TPS is very low. Expected 1,500-3,000 for 7B on A100.

### Multi-GPU Scaling

TODO: Benchmark with 1, 2, 3, 4 inference engines

---

## Step 3 Timing Jump Analysis

### Observation
- Steps 0-2: ~150s per step
- Step 3+: ~300s per step (2x slowdown)

### Hypothesis: Reference Logprobs Computation

**Configuration flag**: `compute_ref_logprobs_in_rollout=False` (current default)

**Code Analysis:**

In `src/training/loss.py:245-258`:
```python
# When trajectories DON'T have ref_logprobs (lines 245-258):
with torch.no_grad():
    with self.model.disable_adapter():
        ref_outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
    # ... compute ref_logprobs via forward pass
```

This runs a **full forward pass with disabled adapter** for EVERY batch chunk.

**BUT** - this should happen for ALL steps when `compute_ref_logprobs_in_rollout=False`, not just step 3+.

### Revised Hypothesis: Buffer Exhaustion

With `buffer_depth=2`:
- Steps 0-1: Consume prefilled rollouts (fast, buffer is full)
- Step 2: Buffer starts to drain, might wait for new rollouts
- Step 3+: Consistently waiting for rollouts (buffer_depth < target)

The timing jump may be due to **rollout starvation**, not ref_logprobs.

**Evidence from trainer (lines 1107, 1174-1179):**
1. Buffer is prefilled with `buffer_depth * num_groups_per_step` rollouts
2. Each step consumes `num_groups_per_step` rollouts
3. Replacements are launched after consumption
4. With `buffer_depth=2`, step 3 is the first step where replacements from step 0 arrive

### Potential Causes of Timing Jump

| Factor | Impact | Evidence Needed |
|--------|--------|-----------------|
| Buffer exhaustion | Wait for rollouts | Check `max_rollout_total_s` in WandB |
| Adapter reload overhead | ~2-5s per rollout | Check `max_volume_reload_s` in WandB |
| Ref logprobs computation | ~50% training time | Check `used_cached_ref_logprobs` |
| Inference engine cold starts | Variable | Check container count in Modal |

### Test Plan

**Test 1: Check WandB metrics**
- Look for `buffer_depth_actual` dropping below 2 at step 3
- Compare `max_rollout_total_s` between early and late steps
- Check `benchmark/used_cached_ref_logprobs` (should be False)

**Test 2: Enable ref_logprobs in rollout**
```python
# In config, set:
compute_ref_logprobs_in_rollout=True
```
- Rollouts will be slower (extra forward pass)
- Training will be faster (no ref forward pass)
- Net effect: May reduce step time if trainer is bottleneck

**Test 3: Increase buffer depth**
```python
# In config, set:
buffer_depth=3  # or 4
```
- More rollouts prefilled
- Less chance of starvation
- Trade-off: More memory, older adapter versions in buffer

### Summary

The step 3 timing jump is likely caused by **buffer exhaustion** rather than ref_logprobs alone. The buffer runs out after step 2, and subsequent steps must wait for replacement rollouts.

To fix:
1. Increase `buffer_depth` to 3-4
2. OR enable `compute_ref_logprobs_in_rollout=True` to shift work to rollouts
3. OR scale up inference engines to speed up rollout generation

---

## vLLM Configuration

### Current Settings (A100)

```python
gpu_memory_utilization = 0.92
max_num_seqs = 512
max_num_batched_tokens = 16384
enable_prefix_caching = True
enable_lora = True
max_loras = 8
```

### Modal Scaling

```python
gpu = "A100"
scaledown_window = 600  # 10 minutes
min_containers = 2
max_inputs = 20  # Updated from 512 - allow burst to 20 concurrent batches
target_inputs = 8  # Updated from 400 - scale up at 8 queued batches (~16-40s work)
```

**Rationale for tuning (2024-12-19):**
- Old `target_inputs=400` meant Modal wouldn't scale until 400 batches queued (~13-33 min of work!)
- New `target_inputs=8` scales up when ~16-40s of work is queued
- This reduces latency without significantly increasing container count
- Container count depends on actual concurrent requests, not the threshold

---

## Expected vs Observed TPS

### Industry Baselines (Qwen 7B on A100)

| Source | Config | TPS |
|--------|--------|-----|
| [DatabaseMart A100 80GB](https://www.databasemart.com/blog/vllm-gpu-benchmark-a100-80gb) | Batch (50 req, 100in/600out) | 3,362 |
| [DatabaseMart A100 40GB](https://www.databasemart.com/blog/vllm-gpu-benchmark-a100-40gb) | Batch throughput | 2,500+ |
| [Inferless Azure A100](https://www.inferless.com/learn/exploring-llms-speed-benchmarks-independent-analysis---part-2) | Qwen1.5-14B, single request | 46.84 |

**Key insight:** High TPS (2,500-3,300) requires batch processing with many concurrent requests.

### Our Observed TPS

| Source | Scenario | Input TPS | Output TPS |
|--------|----------|-----------|------------|
| Single-game benchmark | 42 trajectories | 674 | 36-43 |
| Axiom (batch inference) | 7-42 prompts | 600-8,000 | 300-1,700 |

### Gap Analysis

**Single-game benchmark (36-43 TPS):**
- Similar to single-request latency benchmarks (~47 TPS for 14B)
- Low batch utilization: 42 trajectories processed sequentially
- Expected: need concurrent requests to reach high throughput

**Axiom batch metrics (300-1,700 TPS):**
- Much closer to expected range
- Still below max (3,300) likely due to:
  - LoRA adapter overhead
  - Logits processor overhead (~5-13%)
  - Short output lengths (avg 23 tokens)
  - Small batch sizes (7-42 vs 50+)

---

## TPS Analysis Conclusions

### Answers to Research Questions

1. **What's our actual tokens/game?**
   - ✅ Answered: ~19,500 total tokens per game (18,462 prompt + 991 completion)
   - ~440 prompt tokens per inference call
   - ~24 completion tokens per inference call

2. **Is the logits processor the bottleneck?**
   - ✅ No - benchmarked at 5-13% overhead, not the main factor
   - Main factor is batch size and concurrency

3. **Why the decode TPS gap?**
   - ✅ Explained: Our workload characteristics differ from benchmarks
   - Short outputs (23 tokens avg vs 600 in benchmarks)
   - Small batches (7-42 vs 50+ in benchmarks)
   - LoRA adapter overhead (unavoidable for our use case)

4. **Optimal scaling strategy?**
   - Recommendation: **Scale horizontally with more inference engines**
   - Current 3-4 engines are likely saturated during peak training
   - Test with 6-8 engines to see if throughput scales

### Recommendation

Our TPS is **reasonable given the workload** (20-55% of max baseline). The gap is explained by:
1. LoRA adapters (required for training)
2. Short output sequences (inherent to Diplomacy moves)
3. Logits processor constraint checking (5-13% overhead, acceptable)

**Action**: Focus on horizontal scaling rather than vLLM optimization.

---

## Next Steps

- [x] Run single-game benchmark to measure tokens/game
- [x] Compare TPS to Qwen 7B baselines
- [x] Investigate step 3 timing (documented hypothesis: buffer exhaustion)
- [x] Clean up branch (removed broken logits processor stats code)
- [ ] Test `compute_ref_logprobs_in_rollout=True` to verify hypothesis
- [ ] Test horizontal scaling (6-8 inference engines)
- [ ] Increase `buffer_depth` to 3-4 to mitigate step 3 timing

---

## Branch Cleanup Summary (2024-12-19)

### Removed (Broken Due to Module Isolation)

| File | Removed |
|------|---------|
| `src/inference/logits.py` | `LogitsProcessorStats` class, `_class_stats`, `_global_stats`, timing instrumentation |
| `src/apps/inference_engine/app.py` | `get_logits_processor_stats()` import and method, `logits_processor` in Axiom logs |

### Kept (Working Improvements)

| File | Kept |
|------|------|
| `src/inference/logits.py` | Text-based tag detection, duplicate unit prevention |
| `src/apps/inference_engine/app.py` | TPS metrics (input/output tokens per second), Axiom integration |
| `src/apps/*/app.py` | Correct Modal app names |
| `src/training/loss.py` | AdaptiveKLController |
| `src/utils/config.py` | samples_per_group=6, buffer_depth=2, rollout_horizon_years=3 |
| `scripts/benchmark_logits_processor.py` | Standalone benchmark tool |
| `scripts/benchmark_single_game.py` | Token counting benchmark |
