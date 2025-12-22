# Trajectory Grouping in GRPO Training

## Original Design (Step 1 Only)

```
                    WARMUP PHASE                      FORK PHASE
                    ───────────────                   ──────────────────────────────────────

Game State S₀ ──► Warmup Steps ──► State S_fork ─┬─► Fork 1: Step1 → Step2 → ... → Final Score = 7.2
                                                 │
                                                 ├─► Fork 2: Step1 → Step2 → ... → Final Score = 5.1
                                                 │
                                                 ├─► Fork 3: Step1 → Step2 → ... → Final Score = 8.4
                                                 │
                                                 └─► Fork 4: Step1 → Step2 → ... → Final Score = 6.0

COLLECTED DATA (Old):
┌─────────────────────────────────────────────────────────────┐
│ Group: "game123_FRANCE_1901"                                │
│                                                             │
│   Fork 1, Step 1: action="A PAR-BUR", reward=7.2           │
│   Fork 2, Step 1: action="A PAR-MAR", reward=5.1           │
│   Fork 3, Step 1: action="A PAR-BUR", reward=8.4           │
│   Fork 4, Step 1: action="A PAR-GAS", reward=6.0           │
│                                                             │
│   All 4 trajectories share IDENTICAL starting state S_fork  │
│   Advantage = (reward - 6.675) / 1.38                       │
└─────────────────────────────────────────────────────────────┘

Problem: Steps 2-10 are discarded. Model can't learn which later decisions mattered.
```

## New Design (All Steps)

```
Game State S₀ ──► Warmup ──► S_fork ─┬─► Fork 1: S_fork → S₁ᵃ → S₂ᵃ → S₃ᵃ → Final = 7.2
                                     │
                                     ├─► Fork 2: S_fork → S₁ᵇ → S₂ᵇ → S₃ᵇ → Final = 5.1
                                     │
                                     ├─► Fork 3: S_fork → S₁ᶜ → S₂ᶜ → S₃ᶜ → Final = 8.4
                                     │
                                     └─► Fork 4: S_fork → S₁ᵈ → S₂ᵈ → S₃ᵈ → Final = 6.0

COLLECTED DATA (New):
┌─────────────────────────────────────────────────────────────┐
│ Group: "game123_FRANCE_step1"                               │
│                                                             │
│   Fork 1: state=S_fork, action=A₁ᵃ, reward=7.2             │
│   Fork 2: state=S_fork, action=A₁ᵇ, reward=5.1             │  ← IDENTICAL starting state ✓
│   Fork 3: state=S_fork, action=A₁ᶜ, reward=8.4             │
│   Fork 4: state=S_fork, action=A₁ᵈ, reward=6.0             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Group: "game123_FRANCE_step2"                               │
│                                                             │
│   Fork 1: state=S₁ᵃ, action=A₂ᵃ, reward=7.2                │
│   Fork 2: state=S₁ᵇ, action=A₂ᵇ, reward=5.1                │  ← DIFFERENT states!
│   Fork 3: state=S₁ᶜ, action=A₂ᶜ, reward=8.4                │
│   Fork 4: state=S₁ᵈ, action=A₂ᵈ, reward=6.0                │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Group: "game123_FRANCE_step3"                               │
│                                                             │
│   Fork 1: state=S₂ᵃ, action=A₃ᵃ, reward=7.2                │
│   Fork 2: state=S₂ᵇ, action=A₃ᵇ, reward=5.1                │  ← Even MORE different states!
│   Fork 3: state=S₂ᶜ, action=A₃ᶜ, reward=8.4                │
│   Fork 4: state=S₂ᵈ, action=A₃ᵈ, reward=6.0                │
└─────────────────────────────────────────────────────────────┘
```

## The Theoretical Question

**You're right to notice this:** For step 2+, the forks have DIVERGED. We're comparing:
- Fork 1's step 2 decision from state S₁ᵃ
- Fork 2's step 2 decision from state S₁ᵇ (different state!)

Is this still valid in GRPO?

## Why It Still Works (With Caveats)

### 1. Same Opportunity Principle

All forks had the **same opportunity** to reach any state. The divergence came from their own decisions:

```
S_fork ──► Fork 1 chose A₁ᵃ ──► reached S₁ᵃ (maybe good position)
       └─► Fork 2 chose A₁ᵇ ──► reached S₁ᵇ (maybe bad position)
```

The step 2 comparison asks: "Given the path you took to get here, how did you do?"

### 2. Reward Still Correlates with Decision Quality

```
Fork 3: Made good decisions at steps 1, 2, 3 → Final reward 8.4
Fork 2: Made bad decisions at steps 1, 2, 3  → Final reward 5.1

Step 2 advantage for Fork 3 = (8.4 - 6.675) / 1.38 = +1.25  ← positive
Step 2 advantage for Fork 2 = (5.1 - 6.675) / 1.38 = -1.14  ← negative
```

Even though step 2 states are different, trajectories that led to better outcomes get positive advantage.

### 3. The Noise Averages Out

Over many training steps:
- Sometimes a fork is in a great position at step 2 but gets unlucky later
- Sometimes a fork is in a bad position at step 2 but recovers
- These cancel out in expectation

### 4. This is How AlphaGo Zero Worked

AlphaGo Zero assigned the final game outcome (+1/-1) to EVERY move in the game:

```
Move 1 ─► Move 2 ─► ... ─► Move 100 ─► Win (+1)
  ↑         ↑                  ↑
  +1        +1                +1  (all get same reward)
```

The intuition: "All moves on the winning trajectory were probably good moves."

## Trade-off Analysis

| Aspect | Step 1 Only (Old) | All Steps (New) |
|--------|-------------------|-----------------|
| Starting state | Identical ✓ | Diverges over time |
| Signal clarity | Very clean | Noisier for later steps |
| Data volume | ~900 trajectories/step | ~9,000 trajectories/step |
| Credit assignment | Step 1 gets all credit | All steps get credit |
| Learning speed | Slow (sparse signal) | Faster (dense signal) |

## The Key Insight

**GRPO doesn't require identical starting states.** It requires that trajectories in a group be "comparable" in some sense.

For game-playing agents, comparing trajectories from the same initial game state (even after they've diverged) is a reasonable choice because:

1. They all had equal opportunity at the start
2. The final reward reflects cumulative decision quality
3. The noise from state divergence is bounded and averages out

## Visual: Advantage Computation Flow

```
Raw Trajectories                    Grouped by Step                 Normalized Advantages
─────────────────                   ───────────────                 ─────────────────────

Fork 1, Step 1, r=7.2 ─┐            Group "step1":                  Fork 1 Step 1: +0.38
Fork 2, Step 1, r=5.1 ─┼──►         mean=6.675, std=1.38    ──►     Fork 2 Step 1: -1.14
Fork 3, Step 1, r=8.4 ─┤                                            Fork 3 Step 1: +1.25
Fork 4, Step 1, r=6.0 ─┘                                            Fork 4 Step 1: -0.49

Fork 1, Step 2, r=7.2 ─┐            Group "step2":                  Fork 1 Step 2: +0.38
Fork 2, Step 2, r=5.1 ─┼──►         mean=6.675, std=1.38    ──►     Fork 2 Step 2: -1.14
Fork 3, Step 2, r=8.4 ─┤            (same rewards!)                 Fork 3 Step 2: +1.25
Fork 4, Step 2, r=6.0 ─┘                                            Fork 4 Step 2: -0.49

                                    ↓

                        Global re-normalization (REINFORCE++)
                        across ALL groups in the batch
```

## Potential Improvement: Step-Weighted Rewards

If you find later steps are too noisy, you could discount by step:

```python
# Earlier steps had more influence on outcome
step_weight = 1.0 / step_count  # or exponential decay
adjusted_reward = final_reward * step_weight
```

But I'd recommend trying the current approach first - it may work well enough.
