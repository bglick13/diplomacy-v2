# Stratification vs Value Function Baselines

*Analysis from 2024-12-28 training session*

## Context

During Exp 13 (GRPO grouping ablation), we found state stratification provides ~34% variance reduction compared to no stratification. This raised the question: is stratification just a discrete approximation of what a learned value function does?

## What Stratification Does

- Partitions states into buckets: `(power, SC ∈ {low,mid,high}, year ∈ {early,mid,late})`
- GRPO computes advantages *within* each bucket: `A(s,a) = R(s,a) - mean(R(s,·))` for s in same bucket
- Implicitly says "the baseline for a 3-SC early-game state is the average reward of other 3-SC early-game states"

## What a Value Function Does

- V(s) estimates expected future return from state s
- PPO/A2C compute advantages: `A(s,a) = R(s,a) - V(s)`
- Explicitly learns what each *individual* state is worth

## The Key Difference: Granularity

| Approach | Baseline | Granularity |
|----------|----------|-------------|
| No stratification | mean(all rewards in group) | 1 bucket (whole game) |
| State stratification | mean(rewards in state bucket) | 9-27 buckets |
| Learned V(s) | V(s) for each state | Continuous (~∞) |

The 34% variance reduction shows stratification captures *some* of the "what state am I in" signal. But it's coarse - a 3-SC position with BUR+PAR+MAR is very different from 3-SC with TYR+VEN+ROM, yet they're in the same bucket.

## Theoretical Case for Value Functions

1. **Credit assignment**: V(s) directly tells you "this state is worth X", separating "got here by luck" from "made a good move"
2. **Continuous granularity**: No arbitrary bucket boundaries
3. **Temporal structure**: Value functions naturally propagate information backwards through trajectories

## Practical Challenges

1. **Sample efficiency**: Need to train value network alongside policy. With LoRA + vLLM, we're optimizing for fast inference, not value estimation.
2. **Architecture complexity**: Where does V(s) live? Separate head on the LLM? Separate smaller network? Either adds engineering.
3. **Bootstrap instability**: V(s) targets depend on V(s'), creating circular dependencies that can be unstable.

## Recommendation

Stratification gives ~80% of the benefit with ~10% of the complexity. For initial training runs, this is a reasonable tradeoff.

**Trigger conditions for adding a value function:**
- Learning plateau despite good stratification metrics
- High variance even within buckets
- Clear evidence the model is "confusing state quality with action quality"

**Before adding V(s), try:**
- More granular buckets (unit count? specific territory control?)
- Position-specific features in bucket definition
- Finer year/SC thresholds

## The Deeper Question

Can GRPO with clever grouping ever match PPO with a good value function?

In theory, no - GRPO's within-group normalization is fundamentally limited to discrete partitions.

In practice, for getting a model from "can't play Diplomacy" to "plays competently", stratification may be sufficient. The value function complexity becomes justified when pushing for expert-level play where fine-grained credit assignment matters.

## Related Experiments

- **Exp 13**: GRPO grouping ablation, showed state stratification outperforms no stratification
- **Golden v1 config**: Uses state stratification, deferred value function to future work
