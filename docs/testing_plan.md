## Testing & Benchmark Plan

This document lists the required validation steps for every performance phase. All commands assume you are at the repo root and have run `uv sync` once locally.

### Baseline
- `uv run pytest -q` – must stay green before/after every change.
- `uv run python scripts/profile_rollout.py --target trainer --steps 1 --groups 1 --samples 1 --horizon 1 --no-warmup --profile-name local-trainer-profile` – sanity check that the Modal profiling harness works. Inspect the printed trace directory under `/traces`.

### Phase 1 – Instrumentation & Telemetry
- `uv run python scripts/benchmark_training.py --smoke --profile rollout --profile-name rollout-profile --no-warmup` – persists a JSON snapshot via `persist_profile_snapshot`.
- For GPU stats, start any trainer function (`modal run app.py::train_grpo --detach`) and confirm `gpu_stats` events arrive in Axiom (Dashboard panel 9).

### Phase 2 – Rollout Throughput
- Warm-start cache: `uv run python scripts/benchmark_training.py --profile rollout --groups 2 --samples 2 --horizon 2 --use-state-cache --compact-prompts`.
- Verify LoRA cache hits by tailing `modal logs run_rollout` and confirming `Adapter already cached locally` appears.
- Measure batching efficiency via `uv run python scripts/profile_rollout.py --target rollout --steps 2 --groups 4 --samples 2 --compact-prompts`.

### Phase 3 – Trainer Compute
- `uv run python scripts/profile_rollout.py --target trainer --steps 1 --groups 2 --samples 2 --compact-prompts` – ensures pre-tokenized rollouts and reference logprobs are consumed without a second forward pass.
- Inspect the saved payload in `/data/benchmarks` and confirm `loss_forward_ms` ≈ `backward_ms`.

### Phase 4 – Scheduling & Buffering
- `uv run python scripts/benchmark_training.py --steps 4 --groups 2 --samples 2 --buffer-depth 3 --policy-lag 2 --profile e2e`.
- Confirm Axiom dashboard (#11) shows `policy_lag_steps <= max_policy_lag_steps` and `buffer_depth` oscillating around the configured value.

### Phase 5 – Inference Prompt & Engine Tuning
- `uv run python scripts/profile_rollout.py --target rollout --steps 1 --groups 2 --samples 2 --compact-prompts`.
- Compare `timing/tokens_per_second` WandB metric before/after enabling compact prompts (expect ≳30% improvement for movement phases).

### Phase 6 – Observability/Dashboards
- `uv run python scripts/profile_rollout.py --target trainer --steps 1 --groups 1 --samples 1 --profile-name dashboard-check`.
- Open the Axiom dashboards defined in `docs/axiom_dashboard_queries.md` (panels 9–11) to ensure GPU stats, inference throughput, and policy lag metrics update in near real time.

### CI Coverage
GitHub Actions (`.github/workflows/ci.yml`) runs:
1. `tests` job – unit tests via `uv run pytest -q`.
2. `benchmarks` job – (requires `MODAL_TOKEN_ID/SECRET` secrets) executes `uv run python scripts/profile_rollout.py --target trainer --steps 1 --groups 1 --samples 1 --horizon 1 --no-warmup --profile-name ci-trainer-profile --compact-prompts` to catch regressions in logging/profiling.

Configure Modal credentials and (optionally) Axiom secrets in the repository settings before enabling the benchmark job.
