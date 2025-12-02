# Problem
Training is too slow to run experiments and learn anything meaningful at a reasonable pace. We need a plan that attacks every stage of the pipeline (rollouts → data prep → trainer) while tightening observability so we can prove each win.

## Last training run
- `python scripts/benchmark_training.py --full --steps 20 --groups 32 --samples 2 --lr 1e-5 --horizon 4`
- ~1 hour to run 20 steps (≈180 s/step). Rollouts alone take 70‑130 s and the trainer GPU never exceeds ~40‑50 % utilization.
- `train_grpo_benchmark` already implements double-buffered rollouts, so the remaining latency is real work (game sim + inference + dual forward in the loss).

## Observed bottlenecks
1. **Rollout throughput** (`app.py::run_rollout`)
   - One rollout ~=2 min even though `run_rollout` batches all 7 powers per phase. Warmup phases, repeated prompt construction, and cloudpickle cloning dominate CPU time.
   - Volume reload + LoRA discovery happens for every worker per step, adding 5‑10 s of serialized IO.
   - No reuse of simulated states: every rollout replays 0‑8 warmup turns before the actual fork.
2. **Trainer compute** (`app.py::train_grpo`, `src/training/loss.py`)
   - GRPO loss does two full forward passes (policy + reference) over padded sequences using the same model, cutting effective throughput in half.
   - Tokenization happens on the trainer even though prompts/completions were already tokenized for inference.
   - Chunk size of 4 keeps memory headroom but wastes the H100; gradient checkpointing further slows each pass.
3. **Inference engine**
   - vLLM is configured with `max_num_seqs=256`, but we rarely reach large concurrency because rollout workers call `InferenceEngine().generate.remote` sequentially per phase.
   - LoRA adapter reloads require `volume.reload()` even when the adapter was already loaded on the GPU side.
4. **Scheduling & buffers**
   - Trainer waits for all rollouts from the previous step before kicking off the next set (single-step lag buffer). Any slow rollout stalls the entire pipeline.
   - No policy-lag budget: we always train on the newest data, but that means the GPU idles whenever fresh rollouts are still generating.
5. **Observability gaps**
   - Axiom logs have timing spans, but we do not log per-phase inference token throughput, GPU metrics, or diplomacy engine stats. Hard to attribute wins.

## Optimization program

### 0. Measurement & profiling
**Opportunities**
- Build deterministic micro-benchmarks for (a) rollout-only, (b) trainer-only, (c) end-to-end pipeline to break down wall time.
- Capture GPU utilization, SM occupancy, and memory BW every step via NVML (Modal lets us call `nvidia-smi --query-gpu=utilization.gpu,...`).
- Emit per-phase inference stats: batch size, tokens generated, latency.

**Implementation next steps**
1. Extend `scripts/benchmark_training.py` to optionally run `--profile {rollout|trainer|e2e}` which toggles targeted timers and writes JSON to `/data/benchmarks`.
2. In `InferenceEngine.generate`, wrap each `AsyncLLM.generate` call with `log_inference_request/response`, including `tokens/sec` (vLLM exposes it via `output.metrics`).
3. Add a lightweight NVML sampler coroutine inside `train_grpo[_benchmark]` that logs utilization every 5 s to Axiom.
4. Build a one-off `scripts/profile_rollout.py` that runs `run_rollout` locally (vs Modal) for flamegraphing with `py-spy` before/after optimizations.

### 1. Rollout & environment throughput
**Opportunities**
- Warm start reuse: instead of random warmup each rollout, maintain a library of frozen game states sampled offline and store them in `diplomacy-data`. Each worker loads a seed state via `cloudpickle`—skip 30‑60 s of inference before the fork.
- Persistent LoRA cache: keep the latest adapter mounted inside the worker between rollouts. Only call `volume.reload()` when the adapter version changes.
- Async fork pipeline: convert `run_rollout` to `asyncio` tasks per cloned game. Today we iterate `active_indices` serially; we can schedule inference + parsing concurrently to hide CPU work.
- Batch prompts more aggressively: combine prompts across *multiple rollout workers* (Modal map) by introducing a rollout broker that aggregates `prompts/valid_moves` before calling vLLM, keeping GPU micro-batches near 128 sequences.
- Game engine efficiency: avoid repeated `self.game.get_all_possible_orders()` inside `DiplomacyWrapper.get_valid_moves` for each power by memoizing per phase.
- Visualization toggle currently default 10 %; force `rollout_visualize_chance=0` in training configs.

**Implementation next steps**
1. Create `rules/state_cache/` with ~1k serialized `DiplomacyWrapper` objects representing mid-game phases. Update `run_rollout` to pick from cache when `cfg.use_state_cache` flag is set.
2. Add a per-worker `current_lora_name` global; only call `volume.reload()` + `os.listdir` when a new adapter arrives.
3. Replace the while-loop over `active_indices` with an async producer/consumer: gather prompts, submit to inference concurrently, parse responses as soon as they stream back (vLLM streaming), and immediately step games.
4. Memoize `possible_orders` inside `DiplomacyWrapper.get_all_inputs` so each phase only asks the diplomacy engine once.
5. Prototype a rollout broker Modal function that receives `(prompt, moves, rollout_id)` from many workers, calls `InferenceEngine.generate` once, and routes responses back via async queues.
6. Lower horizon or add adaptive early stopping rules (if no supply center changes for N years) to shrink trajectories without losing variance.

### 2. Inference engine tuning
**Opportunities**
- Enable vLLM prefix caching: our prompts share a long static instruction; caching reduces per-request prefill.
- Use vLLM continuous batching properly: set `allow_concurrent_inputs` even higher and ensure rollout workers never await sequentially.
- LoRA hot-swap: pre-register LoRA adapters with `AsyncEngineArgs(lora_paths=[...])` or keep them on NVMe; avoid per-request `LoRARequest`.
- Trim prompts: `LLMAgent` currently dumps pretty-printed JSON with indentation; flatten + compress reduces tokens by 30‑40 %.
- Explore `vLLM v2` or `TensorRT-LLM` for better throughput once we stabilize correctness.

**Implementation next steps**
1. Update `InferenceEngine.setup` to enable prefix caching (`enable_prefix_caching=True`) and set `max_prefill_tokens` to cover Diplomacy prompts.
2. Benchmark `max_num_seqs` (256 → 512) and `gpu_memory_utilization` (0.85 → 0.92) on A100/H100 to find the sweet spot without OOM.
3. Teach rollout workers to send `lora_version` metadata; GPU engine keeps a dict of mounted adapters and only reloads when missing.
4. Implement a `compact_prompt` mode in `LLMAgent` (minify JSON, remove redundant instructions, swap `<orders>` block to top) and measure tokens/response.
5. Profile helion custom kernels for the logits processor if masking becomes a bottleneck (currently CPU-bound but worth verifying).

### 3. Trainer computation
**Opportunities**
- Avoid double forward pass: capture reference logprobs during rollouts. vLLM can return greedy token logprobs (`return_logprobs=True`). Persist them with trajectories so the trainer only needs the policy forward.
- If we still need reference on the trainer, run it on a frozen copy of the base model WITHOUT gradient checkpointing and with `torch.compile(..., mode="reduce-overhead")`. Keep LoRA-enabled model separate for the policy forward.
- Increase `chunk_size` dynamically based on sequence length histogram; we can likely push to 12‑16 with gradient checkpointing disabled on the reference-only model.
- Consider FSDP or ZeRO-1 if we move to larger base models; for 7B we can simply raise `torch.set_grad_enabled(False)` for 50 % of the layers during reference pass and reuse activations (Helmholtz-style).
- Helion kernels: custom fused cross-entropy + masked sum for GRPO would shave a few ms/token.

**Implementation next steps**
1. Extend rollout trajectories to include `token_ids`, `attention_mask`, `completion_logprobs`, and `prompt_len`. Modify `process_trajectories` to trust provided tensors, skipping tokenizer entirely.
2. Split `GRPOLoss` into two modules: `PolicyForward` (LoRA enabled, grad) and `ReferenceForward` (base model only, compiled). Share embeddings/output head to avoid duplication.
3. Experiment with `torch.compile(policy_model, mode="max-autotune")` after we remove adapter toggling from the compiled graph (possible if we keep separate policy/ref models).
4. Increase `chunk_size` adaptively by measuring max sequence length; log actual tokens per batch and adjust at runtime.
5. Evaluate gradient accumulation vs larger physical batch to keep GPU busy but within VRAM budget; add CLI flag to `train_grpo_benchmark` for quick sweeps.

### 4. Trajectory processing & data plumbing
**Opportunities**
- Tokenize once: rollouts already know the prompt and completion strings; tokenization there lets us reuse `input_ids` without recomputing on the trainer.
- Compress trajectories: store them in `pyarrow` or `npz` batches on Modal volume so trainer can mmap rather than hold Python dicts.
- Streaming buffer: implement the “continuous rollout buffer” idea with a priority sampler that biases to recent data but never stalls the trainer waiting for perfect freshness.

**Implementation next steps**
1. Modify `run_rollout` to return `{"input_ids": list[int], "prompt_len": int, "logprob": float, ...}`; adjust serialization to use `torch.int32` tensors (less bandwidth).
2. Update `process_trajectories` to detect pre-tokenized inputs and simply normalize advantages; keep legacy path for older data.
3. Add a circular buffer service (Modal volume + sqlite index) where rollouts push batches and trainer pulls whichever batches satisfy the policy-lag budget (e.g., allow up to 2 steps of staleness).

### 5. Scheduling & resource usage
**Opportunities**
- Multi-step rollout buffer: always keep 2‑3 steps of rollouts queued ahead of the trainer. Already partially implemented in `train_grpo_benchmark`; port it to `train_grpo`.
- Autoscale rollout workers based on trainer demand; use Modal `Function.starmap` with `concurrency_limit` per region.
- Separate warm pool for inference vs rollouts (cold start cost differs).

**Implementation next steps**
1. Refactor `train_grpo` to share the double-buffer logic from `train_grpo_benchmark`, but generalize to `buffer_depth` flag and integrate the continuous rollout queue.
2. Introduce a small controller process that monitors outstanding rollouts and spins up/down workers (Modal `Function.allow_concurrent_inputs`).
3. Add CLI knobs in `scripts/benchmark_training.py` to explore `buffer_depth`, `num_groups`, `samples`, and `horizon` quickly.

### 6. Observability upgrades
**Opportunities**
- Richer dashboards: Axiom panel for rollout latency percentiles, token/sec, GPU util, policy lag.
- Alerting on stalled rollouts or NaN loss immediately.

**Implementation next steps**
1. Extend `axiom_dashboard_queries.md` with new queries (rollout latency histogram, inference throughput, trainer idle time).
2. Emit `policy_lag_steps` metric every training iteration to ensure buffer logic works.
3. Wire Modal log streams into WandB tables for easier diffing between runs.

## Proposed execution order
1. Land profiling + telemetry (Section 0) so future optimizations have clear baselines.
2. Ship rollout warm-start cache + LoRA caching (Section 1) to cut per-rollout latency; expect >2×.
3. Reuse tokens/logprobs from rollouts to eliminate trainer re-tokenization + reference forward (Sections 3 & 4). This should roughly halve trainer step time.
4. Port double-buffer + continuous queue into `train_grpo` (Section 5) to keep GPU saturated.
5. Iterate on inference prompt compaction + vLLM tuning (Section 2).
6. Explore helion/custom kernels once the macro architecture is stable.

With this plan we can move from ~20 steps/hour to a target of ≥60 steps/hour (3× throughput) before scaling out hardware. The key is to first remove wasted work (duplicate tokenization, redundant inference warmups) and only then consider larger models or GPU counts.

