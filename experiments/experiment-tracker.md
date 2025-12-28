# Experiment Tracker

## Experiment Schema

Each experiment should include:
- **Hypothesis**: What we expect to learn
- **Key Config Changes**: Differences from baseline
- **Command**: Full command to run
- **Status**: `planned` | `running` | `completed` | `failed`
- **Results**: Key findings (after completion)

---

## Exp 1: League Training from Scratch (Baseline)

**Hypothesis**: League training with PFSP matchmaking and Elo pressure will develop stronger play than pure self-play by exposing the model to diverse opponents.

**Key Config Changes**:
- `--league-training`: Enable league mode with PFSP matchmaking
- `--rollout-horizon-years 5`: Medium horizon for balanced exploration/exploitation
- `--winner-threshold-sc 7`: Moderate win threshold to encourage aggressive play
- `--buffer-depth 3`: Prefetch rollouts to keep trainer fed

**Command**:
```bash
python scripts/launch_training.py \
  --rollout-horizon-years 5 \
  --winner-threshold-sc 7 \
  --buffer-depth 3 \
  --num-groups-per-step 16 \
  --total-steps 250
```

**Status**: `concluded` (cancelled at step ~10)
- wandb: https://wandb.ai/bglick13/diplomacy-grpo/runs/01dp7f4x?nw=nwuserbglick13
- run name: grpo-20251218-114516

**Results**:
- **Early training instability observed**: reward_mean declined from 14→8 in first 8 steps
- `win_bonus_rate` dropped from 20%→3%
- `avg_sc_count` dropped from 5.0→3.5
- **Root cause identified**: KL penalty (β=0.04) too aggressive in early training, pushing model back toward base model behavior before it could learn
- **Key learning**: Need KL warmup to allow early exploration without penalty

**Follow-up**: → Exp 6 (KL Warmup)

---

## Exp 2: Resume from Self-Play Checkpoint

**Hypothesis**: Starting league training from a self-play checkpoint (with basic competence) will converge faster than training from scratch, as the model already understands game mechanics.

**Key Config Changes**:
- `--resume-from-adapter`: Path to self-play checkpoint
- `--initial-elo 1200`: Start with elevated Elo reflecting prior training
- Lower learning rate to fine-tune rather than re-learn

**Command**:
```bash
python scripts/launch_training.py \
  --league-training \
  --resume-from-adapter "runs/selfplay_v1/adapter_v100" \
  --rollout-horizon-years 5 \
  --winner-threshold-sc 7 \
  --buffer-depth 3 \
  --num-groups-per-step 16 \
  --learning-rate 1e-5 \
  --total-steps 150
```

**Status**: `planned`

**Results**: _pending_

---

## Exp 3: Extended Horizon Training

**Hypothesis**: Longer rollout horizons (more game years) will develop better long-term strategic planning, though at higher compute cost per rollout.

**Key Config Changes**:
- `--rollout-horizon-years 8`: Extended horizon (vs baseline 5)
- `--winner-threshold-sc 10`: Higher threshold for longer games
- May need `--buffer-depth 4` to compensate for slower rollouts

**Command**:
```bash
python scripts/launch_training.py \
  --league-training \
  --rollout-horizon-years 8 \
  --winner-threshold-sc 10 \
  --buffer-depth 4 \
  --num-groups-per-step 12 \
  --total-steps 200
```

**Status**: `planned`

**Results**: _pending_

---

## Exp 4: Baseline Phase-Out Schedule

**Hypothesis**: Gradually reducing baseline bot opponents over training will provide curriculum learning - easy opponents early, harder (self-play/league) opponents later.

**Key Config Changes**:
- Custom PFSP weight schedule that reduces `baseline_weight` over time
- Start: 40% baseline, 30% self, 20% peer, 10% exploitable
- End: 10% baseline, 40% self, 35% peer, 15% exploitable

**Implementation Note**: Requires adding dynamic PFSP weight scheduling to config. For now, can approximate with multiple runs:

**Phase 1 (steps 0-100)**: High baseline exposure
```bash
python scripts/launch_training.py \
  --league-training \
  --pfsp-baseline-weight 0.4 \
  --pfsp-self-play-weight 0.3 \
  --pfsp-peer-weight 0.2 \
  --pfsp-exploitable-weight 0.1 \
  --total-steps 100
```

**Phase 2 (steps 100-250)**: Reduced baseline, more league play
```bash
python scripts/launch_training.py \
  --league-training \
  --resume-from-adapter "runs/phase1/adapter_v100" \
  --pfsp-baseline-weight 0.1 \
  --pfsp-self-play-weight 0.4 \
  --pfsp-peer-weight 0.35 \
  --pfsp-exploitable-weight 0.15 \
  --total-steps 150
```

**Status**: `planned`

**Results**: _pending_

---

## Exp 5: High Win Pressure

**Hypothesis**: Increasing win bonus and lowering threshold will create more aggressive, decisive play styles that aim for outright victory rather than survival.

**Key Config Changes**:
- `--win-bonus 20.0`: Double the win incentive (vs baseline 10.0)
- `--winner-threshold-sc 5`: Lower threshold makes wins more achievable
- May increase variance in training - monitor KL divergence

**Command**:
```bash
python scripts/launch_training.py \
  --league-training \
  --win-bonus 20.0 \
  --winner-threshold-sc 5 \
  --rollout-horizon-years 5 \
  --buffer-depth 3 \
  --num-groups-per-step 16 \
  --total-steps 250
```

**Status**: `planned`

**Results**: _pending_

---

## Exp 6: KL Warmup for Training Stability

**Hypothesis**: Early training reward decline (14→8 in first 8 steps) is caused by aggressive KL penalty pushing model back toward base model behavior. Warming up KL beta from 0 will allow early learning without penalty, then gradually introduce regularization.

**Key Config Changes**:
- `--kl-beta-warmup-steps 20`: Linear warmup from 0 to full beta over 20 steps
- `--advantage-clip 5.0`: Clip extreme advantages to prevent gradient explosions
- All other settings same as Exp 1

**Command**:
```bash
python scripts/launch_training.py \
  --rollout-horizon-years 5 \
  --winner-threshold-sc 7 \
  --buffer-depth 3 \
  --num-groups-per-step 16 \
  --total-steps 250 \
  --kl-beta-warmup-steps 20 \
  --advantage-clip 5.0
```

**Status**: `planned`

**Results**: _pending_

**New Metrics to Watch**:
- `kl/beta`: Should ramp from 0 → 0.04 over first 20 steps
- `kl/warmup_progress`: 0 → 1 over warmup period
- `advantage/clipped_count`: How many advantages were clipped
- `processing/skip_rate`: Fraction of trajectories skipped

---

## Exp 7: Zero KL Penalty (Logits Processor Only)

**Hypothesis**: Since the logits processor constrains outputs to valid moves, KL penalty may be unnecessary. Removing it entirely could allow faster policy improvement without fighting regularization.

**Key Config Changes**:
- `--kl-beta 0.0`: Disable KL penalty entirely
- `--advantage-clip 10.0`: Advantage clipping for stability
- `--rollout-horizon-years 4`: Slightly shorter horizon
- `--winner-threshold-sc 5`: Lower win threshold

**Command**:
```bash
python scripts/launch_training.py \
  --rollout-horizon-years 4 \
  --winner-threshold-sc 5 \
  --buffer-depth 3 \
  --num-groups-per-step 16 \
  --total-steps 250 \
  --kl-beta 0.0 \
  --advantage-clip 10.0
```

**Status**: `completed`
- wandb: https://wandb.ai/bglick13/diplomacy-grpo/runs/fobg1kbv
- run name: grpo-20251222-191408

**Results**:
- **Learning signal**: POSITIVE - older checkpoints (v0-v25) declined to ~960 Elo while mid-training checkpoints (v80-v150) peaked at 1050-1068
- **Best checkpoint**: adapter_v150 at 1067.9 Elo (+69 over base_model)
- **Baseline exploitation**: 484 total Elo lost by bots
  - defensive_bot: 943 → 700 (-243)
  - coordinated_bot: 991 → 797 (-195)
  - territorial_bot: 955 → 940 (-15)
  - chaos_bot: 901 → 892 (-9)
- **Training stability**: KL spikes up to 392K but recovered within 5 steps; grad_norm increased 10x (3.7 → 34.5 mean)
- **Reward trajectory**: 3.60 → 3.46 (stable, not climbing)
- **Risk confirmed**: Policy did drift significantly (transient KL spikes) but training didn't collapse

**Key learnings**:
1. Zero KL is survivable but produces chaotic training dynamics
2. +69 Elo improvement shows real learning occurred
3. Diminishing returns after step ~150 (best checkpoint)
4. Grad norm escalation (10x) suggests instability accumulating

---

## Exp 8: Adaptive KL Control

**Hypothesis**: PPO-style adaptive KL control will automatically tune the penalty - increasing when policy drifts too fast, decreasing when learning is slow.

**Key Config Changes**:
- `--kl-beta-warmup-steps 10`: Short warmup
- `--kl-target 0.02`: Target KL divergence
- Beta auto-adjusts: ×1.5 if KL > 1.5×target, ÷1.5 if KL < 0.5×target

**Command**:
```bash
python scripts/launch_training.py \
  --league-training \
  --rollout-horizon-years 5 \
  --winner-threshold-sc 7 \
  --buffer-depth 3 \
  --num-groups-per-step 16 \
  --total-steps 250 \
  --kl-beta 0.04 \
  --kl-beta-warmup-steps 10 \
  --kl-target 0.02 \
  --advantage-clip 5.0
```

**Status**: `planned`

**Results**: _pending_

**New Metrics to Watch**:
- `kl/beta_adjusted`: +1 when beta increased, -1 when decreased
- `kl/ema`: Smoothed KL used for adaptation decisions

---

## Exp 9: Stability Ablation Sweep

**Hypothesis**: Training instability (KL explosion, clip_frac > 60%) was introduced after commit 6ea97cbb0. By testing each change in isolation, we can identify the culprit(s).

**Changes tested** (each in isolation from stable baseline):
- A: Baseline (attention-only LoRA r16, all new features OFF)
- B: +PPO clipping
- C: +Token-level loss weighting
- D: +LoRA rank 32
- E: +Entropy bonus
- F: +KL penalty
- G: +MLP target modules
- H: +Full LoRA (rank 32 + MLP modules)

**Key Config** (baseline):
- `learning_rate: 5e-6`
- `lora_rank: 16`
- `lora_target_modules: [q_proj, k_proj, v_proj, o_proj]` (attention only)
- All new features OFF

**Status**: `completed`
- Sweep config: `experiments/sweeps/stability-ablation/sweep.yaml`
- Run names: `stability-ablation-{A-H}-20251226-124256`

**Results** (Fixed Reference Analysis):

| Run | Config | Elo Gap | Base Model Elo | KL Max | Verdict |
|-----|--------|---------|----------------|--------|---------|
| H | Full LoRA (r32+MLP) | **+250** | **1001** | 137.1 | **Most learning, unstable** |
| C | +Token-level loss | +98 | 1050 | 3.6 | Good |
| A | Baseline | +89 | 1072 | 0.8 | Stable baseline |
| F | +KL penalty | +84 | 1024 | 1.8 | Stable, good |
| G | +MLP modules | +62 | 1036 | 7.2 | Learning, some instability |
| E | +Entropy bonus | +39 | 1097 | 1.0 | Underperformed |
| D | +LoRA rank 32 | +33 | 1062 | 1.8 | **Worst attention-only** |
| B | +PPO clipping | +22 | 1109 | 2.6 | Underperformed |

**Key Learnings**:
1. **MLP modules enable more learning**: H achieved +250 Elo gap (3x better than attention-only)
2. **Win rate is misleading**: F had 27% win rate but only +84 Elo gap; H had 17% win rate but +250 gap
3. **Fixed references are essential**: Use base_model Elo and Elo gap, not win rate
4. **Attention rank increase alone hurts**: D was worst performer (more routing without knowledge)
5. **MLP instability is manageable**: H learned despite KL_max=137

**Analysis command**:
```bash
uv run python .claude/skills/experiment-analysis/analyze_sweep.py --sweep stability-ablation
```

---

## Exp 10: MLP Ablation (Does MLP Help When KL-Constrained?)

**Hypothesis**: MLP layers store "knowledge" while attention stores "routing". Adding MLP with KL penalty should enable learning Diplomacy-specific strategies while maintaining stability.

**Runs**:
- A: Attention-only r16 + KL penalty (replicates winning F config)
- B: Attention + MLP r16 + KL penalty (test if MLP adds value)

**Key Config**:
- `learning_rate: 5e-6`
- `kl_beta: 0.01` (immediate, no warmup)
- `total_steps: 75`
- All other features OFF

**Status**: `completed`
- Sweep config: `experiments/sweeps/mlp-ablation/sweep.yaml`
- Run names: `mlp-ablation-{A,B}-20251226-182415`

**Results** (Fixed Reference Analysis):

| Run | Steps | Base Model Elo | Elo Gap | KL Mean | Total Ref Drop |
|-----|-------|----------------|---------|---------|----------------|
| **B (MLP + KL)** | 57 (crashed) | **1041** | **+99** | 0.267 | **524** |
| A (attn + KL) | 75 | 1084 | +96 | 0.049 | 382 |

**Key Findings**:
1. **MLP learns faster**: B achieved lower base_model Elo (1041 vs 1084) in fewer steps
2. **MLP crashed at step 57**: KL mean was 0.267 vs 0.049 for attention-only
3. **Need stronger KL constraint for MLP**: kl_beta=0.01 wasn't enough

**Conclusion**: MLP + KL penalty is the right approach, but needs:
- Higher kl_beta (0.02 instead of 0.01)
- Shorter warmup (5 steps instead of 10)

**Follow-up**: → Exp 11 (MLP Stability Sweep)

---

## Exp 11: MLP Stability Sweep (PPO vs KL Regularization)

**Hypothesis**: MLP modules need both PPO clipping AND KL penalty for stable training. Testing four configurations:
- A: KL-only (no PPO clipping)
- B: PPO-only (no KL penalty)
- C: PPO + KL standard (β=0.01)
- D: PPO + KL strong (β=0.02)

**Key Config** (shared):
- `lora_target_modules`: All (q,k,v,o + gate,up,down MLP)
- `lora_rank: 16`
- `learning_rate: 5e-6`
- `total_steps: 30`

**Config Variants**:
| Run | PPO Clipping | KL Beta | Description |
|-----|--------------|---------|-------------|
| A | ❌ | 0.01 | KL regularization only |
| B | ✅ | 0.00 | PPO clipping only |
| C | ✅ | 0.01 | Both (standard KL) |
| D | ✅ | 0.02 | Both (strong KL) |

**Status**: `completed`
- Sweep config: `experiments/sweeps/mlp-stability/sweep.yaml`
- Run names: `mlp-stability-{A,B,C,D}-20251227-124027`
- WandB: https://wandb.ai/bglick13/diplomacy-grpo

**Results** (Fixed Reference Analysis):

| Run | Elo Gap | Base Elo | KL Mean | KL Max | Win% | Top3% |
|-----|---------|----------|---------|--------|------|-------|
| **D** | **+171** | **1024** | 0.58 | **39** | 12% | 56% |
| B | +108 | 1045 | 0.95 | 176 | 14% | 53% |
| C | +38 | 1043 | 1.05 | **496** | 16% | 60% |
| A | +31 | 1058 | 0.44 | 51 | 7% | 58% |

**Clear Winner**: Config D (+171 Elo gap, 63 points ahead of runner-up B)

**Key Learnings**:
1. **You need BOTH regularizers**: Neither KL-only (A) nor PPO-only (B) matched the combined approach
2. **PPO bounds local updates, KL bounds cumulative drift**: They serve different purposes
   - PPO clipping prevents high-variance individual gradient updates
   - KL penalty prevents the policy from drifting too far cumulatively
3. **Standard KL (β=0.01) is too weak for MLP**: Config C had worst KL spikes (496!) despite both regularizers
4. **Strong KL (β=0.02) enables stable exploration**: D explored more (KL mean 0.58 vs A's 0.44) but without instability
5. **Unconstrained optimization isn't better**: B had more freedom but worse results than D

**Core Insight**: The 2x KL penalty isn't "being conservative" — it's enabling *stable exploration* which compounds over training steps. Without it, the policy either stays too close (A) or random-walks through policy space (B, C).

**Action**: Updated `reward-discount-gamma` sweep to use Config D settings:
- `use_ppo_clipping: true`
- `kl_beta: 0.02`
- `kl_beta_warmup_steps: 0`

---

## Exp 12: Reward Structure (Dense vs Sparse)

**Hypothesis**: Temporal credit assignment via `gamma > 0` creates correlated gradients that destabilize training. There are two theoretically sound approaches:
- **Dense + Immediate** (gamma=0): Every step gets its immediate reward (no temporal credit)
- **Sparse + Trajectory**: One sample per trajectory with final outcome as reward

Sparse trajectory-level rewards compare "which overall strategy was better" rather than "which single move was better." This may better capture strategic positioning moves (like Spring maneuvers that enable Fall captures).

**Key Config** (shared, from Exp 11 winner):
- `kl_beta: 0.02`
- `use_ppo_clipping: true`
- `lora_target_modules: [attention + MLP]`
- `total_steps: 100`

**Config Variants** (matched effective batch size):
| Run | Reward Type | Rollouts/Step | Samples/Step |
|-----|-------------|---------------|--------------|
| A | Dense (step deltas) | 16 | ~960 |
| B | Sparse (final score) | 240 | ~960 |

Note: Sparse needs 15x more rollouts to match sample count, isolating the reward structure effect.

**Status**: `completed` (invalidated by EMA bug - see below)
- Sweep config: `experiments/sweeps/reward-structure/sweep.yaml`
- Run names: `reward-structure-v3-{A,B}-20251228-005158`

**Results** (before EMA bug discovery):

| Run | Steps | Base Model Δ | DumbBot Δ | KL Max | Verdict |
|-----|-------|--------------|-----------|--------|---------|
| A (Dense) | 40 (crashed) | **-78** ✓ | +7 | 1,598 | **Better learning** |
| B (Sparse) | 100 (unstable) | -21 | +68 ✗ | 129,633 | Slower, worse |

**Observations**:
- Dense dropped base_model Elo by 78 points in 40 steps (~2 Elo/step)
- Sparse only dropped 21 points in 100 steps (~0.2 Elo/step) - 10x slower
- Sparse LOST to dumbbot more over time (+68 Elo) - learning wrong things
- Both crashed due to KL explosion

**Key Finding**: Dense per-step rewards work better for Diplomacy. Sparse trajectory-level rewards provide too weak/delayed a signal.

**INVALIDATED**: Both runs used `use_ema_reference=True` which was buggy (see Bug Discovery below). The EMA reference was never actually used - KL measured vLLM-HF mismatch instead of policy drift. Need to re-run after fix.

---

## Bug Discovery: EMA Reference Bypass (2024-12-28)

**Problem Found**: When `use_ema_reference=True`, the EMA weights were initialized and updated correctly, but **never used for computing reference logprobs**.

**Root Cause**: In `loss.py`, cached `ref_logprobs` from rollouts were used whenever available, bypassing the EMA forward pass entirely:

```python
if all_have_ref_logprobs:  # Always True when rollouts provide logprobs
    ref_completion_log_probs = torch.tensor([b["ref_logprobs"] for b in batch], ...)
    # ^^^ This skips EMA entirely!
```

**What This Means**:
- KL was measuring `HuggingFace_logprobs - vLLM_rollout_logprobs` (numerical mismatch)
- NOT measuring policy drift from EMA reference
- As policy became more peaked, small numerical differences exploded
- `used_cached_ref_logprobs = True` in all affected runs confirms the bug

**Fix Applied**: `src/training/loss.py` now checks `should_use_cached = all_have_ref_logprobs and not self.use_ema_reference`

**Affected Experiments**:
- `reward-structure-v3-{A,B}` - invalidated
- `reward-structure-v2-A` - invalidated
- Any run with `use_ema_reference=True` and rollout logprobs

**Verification**: After fix, runs should show:
- `used_cached_ref_logprobs = False` (EMA actually being used)
- `kl/max < 10` (no explosion)
- Stable training for 100+ steps

---

## Exp 13: GRPO Grouping Strategies

**Hypothesis**: Two orthogonal improvements to GRPO training:
1. **State-stratified groups**: Only compare actions from similar game situations (same SC bucket × year bucket)
2. **Elo-conditioned rewards**: Calibrate rewards by opponent difficulty (beating strong = higher reward)

State stratification should reduce noise from comparing early-game defense with late-game expansion. Elo conditioning should help when playing mixed-strength opponents.

**Key Config** (EMA reference fixed + dense rewards + MLP LoRA):
- `use_ema_reference: true` (now correctly used after bug fix)
- `use_trajectory_level_rewards: false` (dense, validated in Exp 12)
- `lora_target_modules: [attention + MLP]`
- `total_steps: 30` (quick validation)

**Config Variants** (2×2 factorial):
| Run | State Stratified | Elo Conditioned | Description |
|-----|------------------|-----------------|-------------|
| A | ❌ | ❌ | Baseline control |
| B | ✅ | ❌ | Stratification only |
| C | ❌ | ✅ | Elo conditioning only |
| D | ✅ | ✅ | Both combined |

**Status**: `planned`
- Sweep config: `experiments/sweeps/grpo-grouping/sweep.yaml`

**Command**:
```bash
uv run modal deploy -m src.apps.deploy  # Deploy EMA fix first
uv run python scripts/launch_sweep.py experiments/sweeps/grpo-grouping/sweep.yaml
```

**Success Criteria** (validates EMA fix):
1. All runs complete 30 steps without KL explosion (`kl/max < 10`)
2. `used_cached_ref_logprobs = False` (EMA actually being used)
3. At least one run shows `base_model` Elo decline

**Key Questions**:
1. Does state stratification improve learning signal quality?
2. Does Elo conditioning help with mixed-strength opponents?
3. Do they interact positively or negatively when combined?

**Results**: _pending_

---

## Notes

### Fixed Reference Analysis (NEW)

**Key insight**: Win rate against a dynamic league is meaningless. As training progresses, the league gets stronger, so win rate can stay flat even if the model improves.

**Use fixed references instead**:
| Metric | What It Measures |
|--------|------------------|
| `base_model` Elo | Are we better than untrained? (lower = better) |
| Elo Gap (best checkpoint - base_model) | How much better is trained model? (higher = better) |
| Baseline bot Elo | Are we exploiting fixed strategies? (lower = better) |

**Analysis command**:
```bash
uv run python .claude/skills/experiment-analysis/analyze_sweep.py --sweep <prefix>
```

### Compute Estimates
- Each rollout: ~60-90s depending on horizon
- 16 groups/step × 4 samples/group = 64 trajectories/step
- Buffer depth 3 = ~3 rollouts prefetched

### Key Metrics to Track

**Learning Signal (Fixed References)**:
- `elo/base_model`: Lower = league beats untrained model more (PRIMARY)
- Elo Gap (best checkpoint - base_model): Higher = more improvement (PRIMARY)
- `elo/chaos_bot`, `elo/defensive_bot`, etc.: Lower = exploiting fixed strategies

**Performance (Relative to League)**:
- `game/win_bonus_rate`: Misleading if league is dynamic - use with caution
- `game/avg_sc_count`: Territory control indicator
- `benchmark/reward_mean`: Training signal (but affected by opponent strength)

**Stability**:
- `kl/mean`: Policy drift from reference (target < 0.1)
- `kl/max`: Peak divergence (warning if > 10)
- `benchmark/grad_norm`: Gradient magnitude (warning if > 50)

### KL & Stability Metrics (New)
- `kl/beta`: Effective KL penalty coefficient (shows warmup progress)
- `kl/warmup_progress`: 0→1 during warmup period
- `kl/ema`: Smoothed KL for adaptive control decisions
- `kl/beta_adjusted`: +1/-1 when adaptive control adjusts beta
- `advantage/mean`, `advantage/std`: Normalized advantage distribution
- `advantage/clipped_count`: Advantages clipped to prevent extreme gradients
- `processing/skip_rate`: Fraction of trajectories skipped (low variance groups)

### Config Reference
See `src/utils/config.py` for all available options and their descriptions.
