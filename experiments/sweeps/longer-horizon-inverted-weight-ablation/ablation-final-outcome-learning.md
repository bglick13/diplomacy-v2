# Ablation Study: Strategic Learning vs Greedy SC Capture

## Problem

From experiment `grpo-20251222-191408` (Zero KL):
- +69 Elo improvement shows learning occurred
- BUT reward trajectory stayed FLAT at 3.46 (not climbing)
- Baselines heavily exploited (defensive_bot -243 Elo, coordinated_bot -195)
- Suggests model learned to exploit weak play, not strategic play

**Previous Scoring Issues:**
```
reward = step_delta * 0.8 + final_score * 0.2 + strategic_component

Per-SC capture: 2.0 * 0.8 = 1.6 points (immediate)
Win bonus (18 SCs): 50 * 0.2 = 10 points (at game end)
Sparse: Only winner gets bonus, 2nd-7th place get nothing extra
```

**Hypothesis**: Over-rewarding greedy SC capture, under-rewarding strategic play that leads to wins.

## New: Position-Based Final Scoring

Inspired by [webDiplomacy scoring systems](https://webdiplomacy.net/points.php) (Draw-Size Scoring, Sum-of-Squares), we now use position-based bonuses for all finishing ranks:

```python
# Default position bonuses (top-heavy but continuous)
position_bonus_1st: 50.0   # Leader or tied for lead
position_bonus_2nd: 25.0
position_bonus_3rd: 15.0
position_bonus_4th: 10.0
position_bonus_5th: 5.0
position_bonus_6th: 2.0
position_bonus_7th: 0.0    # Surviving but last

# Special cases
solo_victory_sc: 18        # True solo gets full win_bonus on top of 1st bonus
elimination_penalty: -30.0 # Much worse than 7th place
```

**Benefits:**
- Continuous reward signal for all positions, not just winner
- Beating one more opponent always matters
- Large gap between elimination (-30) and surviving 7th (+0)
- Solo victory (50 bonus + 200 win_bonus) >> 1st in draw (50 bonus only)

## Goal

Train agents that learn to WIN, not just capture SCs greedily.

## Key Insight: GRPO Within-Group Normalization

All trajectories in a group (same game_id, power, step) have different `final_component` values because forks diverge and end differently. The signal IS there - but it's currently weighted at only 0.2.

---

## Ablation Design

### Two Dimensions (2x2 = 4 runs):

**1. Horizon:**
- **Short**: 4yr (80%) / 6yr (20%) - ~4.4 years average
- **Long**: 8yr (50%) / 12yr (50%) - ~10 years average

**2. Scoring Philosophy:**
- **SC-Based (Current)**: step_weight=0.8, final_weight=0.2, win_bonus=50, threshold=5
- **Strategic (Position-Based)**: step_weight=0.2, final_weight=0.8, win_bonus=200, threshold=10, `use_strategic_step_scoring=True`

### Full Matrix:

| Run | Horizon | Scoring | Description |
|-----|---------|---------|-------------|
| A | Short | SC-Based | Baseline control |
| B | Long | SC-Based | Test horizon effect alone |
| C | Short | Strategic | Test scoring effect alone |
| D | Long | Strategic | Full strategic (recommended) |

---

## Strategic Step Scoring

The strategic scoring variant (`use_strategic_step_scoring=True`) fundamentally changes how step rewards work:

**SC-Based (Current):**
```python
score = 2.0 * SC + 0.3 * units + tactical_signals
```

**Strategic (New):**
```python
# NO SC count in step score!
base_score = units * 0.5 + 0.5  # survival bonus

# Leader bonus / gap penalty
if gap_to_leader == 0:
    position_score = 2.0  # Bonus for leading
else:
    position_score = -gap * 0.3  # Penalty grows with gap

# Balance bonus for non-leaders
if not_leader:
    balance_score = (1 - cv) * 0.5  # Encourage stopping runaway leaders

step_score = base_score + position_score + balance_score + tactical_signals
```

**Why this works:**
- Step rewards focus on POSITION not accumulation
- Leaders get bonus for staying ahead
- Non-leaders get bonus when game is balanced
- Final score still uses SC count (outcome matters)
- Decouples "immediate tactics" from "SC hoarding"

---

## Ablation Commands

### Run A: Baseline (Short Horizon + SC-Based Scoring)
```bash
uv run python scripts/launch_training.py \
  --total-steps 100 \
  --experiment-tag "ablation-scoring-A-baseline" \
  --leader-gap-penalty-weight 0.0 \
  --balance-bonus-weight 0.0
```

### Run B: Long Horizon + SC-Based Scoring
```bash
uv run python scripts/launch_training.py \
  --total-steps 100 \
  --experiment-tag "ablation-scoring-B-horizon" \
  --rollout-horizon-years 8 \
  --rollout-long-horizon-years 12 \
  --rollout-long-horizon-chance 0.5 \
  --leader-gap-penalty-weight 0.0 \
  --balance-bonus-weight 0.0
```

### Run C: Short Horizon + Strategic Step Scoring
```bash
uv run python scripts/launch_training.py \
  --total-steps 100 \
  --experiment-tag "ablation-scoring-C-strategic" \
  --use-strategic-step-scoring \
  --step-reward-weight 0.2 \
  --final-reward-weight 0.8 \
  --win-bonus 200.0 \
  --winner-threshold-sc 10 \
  --leader-gap-penalty-weight 0.0 \
  --balance-bonus-weight 0.0
```

### Run D: Long Horizon + Strategic Step Scoring (RECOMMENDED)
```bash
uv run python scripts/launch_training.py \
  --total-steps 100 \
  --experiment-tag "ablation-scoring-D-combined" \
  --rollout-horizon-years 8 \
  --rollout-long-horizon-years 12 \
  --rollout-long-horizon-chance 0.5 \
  --use-strategic-step-scoring \
  --step-reward-weight 0.2 \
  --final-reward-weight 0.8 \
  --win-bonus 200.0 \
  --winner-threshold-sc 10 \
  --leader-gap-penalty-weight 0.0 \
  --balance-bonus-weight 0.0
```

---

## Analysis

After runs complete, compare in WandB:
1. Filter by `experiment_tag` prefix "ablation-scoring-"
2. Compare TrueSkill display_rating progression curves
3. Check win_bonus_awarded rate
4. Evaluate compute efficiency (TrueSkill per compute-hour)

## Success Metrics

**Primary:** TrueSkill display_rating progression per compute-hour

**Secondary:**
- Win bonus rate (are games ending decisively?)
- Late-game Elo (does model play well in year 8+?)
- Baseline exploitation pattern
- SC count trajectory (does strategic scoring reduce SC-hoarding?)

**Expected Outcomes:**
- D > C > B > A for TrueSkill
- C, D should have higher win_bonus_rate than A, B
- B, D should have more decisive outcomes than A, C

**Failure indicators:**
- Elo regression
- Training instability (loss spikes, gradient explosions)
- High skip rate due to low-variance groups

---

## Note on Reward Shaping

For clean comparison, all runs disable the separate `leader_gap_penalty` and `balance_bonus` shaping:
```python
leader_gap_penalty_weight: 0.0
balance_bonus_weight: 0.0
```

For strategic scoring (C, D), these concepts are integrated into the step scoring function itself rather than applied as separate add-on shaping.
