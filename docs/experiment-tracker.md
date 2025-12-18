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
- `--advantage-clip 5.0`: Keep advantage clipping for stability

**Command**:
```bash
python scripts/launch_training.py \
  --league-training \
  --rollout-horizon-years 5 \
  --winner-threshold-sc 7 \
  --buffer-depth 3 \
  --num-groups-per-step 16 \
  --total-steps 250 \
  --kl-beta 0.0 \
  --advantage-clip 5.0
```

**Status**: `planned`

**Results**: _pending_

**Risk**: Without KL, policy could drift too far and become unstable. Monitor `benchmark/loss` for divergence.

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

## Notes

### Compute Estimates
- Each rollout: ~60-90s depending on horizon
- 16 groups/step × 4 samples/group = 64 trajectories/step
- Buffer depth 3 = ~3 rollouts prefetched

### Key Metrics to Track
- `benchmark/reward_mean`: Primary training signal
- `game/win_bonus_rate`: How often model achieves decisive wins
- `game/avg_sc_count`: Territory control indicator
- `benchmark/kl`: Policy drift from reference (stability)
- League Elo progression over training steps

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
