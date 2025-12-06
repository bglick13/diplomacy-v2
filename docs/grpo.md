Here is the comprehensive technical specification based on our discussion. You can provide this directly to a coding agent or engineering team to build the training pipeline.

-----

# Technical Specification: GRPO Training for Multiplayer Diplomacy

## 1\. Project Overview

**Goal:** Train an LLM to play *Diplomacy* (7-player, negotiation-based strategy game) using **Group Relative Policy Optimization (GRPO)**.
**Key Innovation:** Replacing the computational expense of a value network (Critic) with group-based sampling and replacing full-game rollouts with truncated, heuristic-evaluated rollouts.

## 2\. Core Algorithm: GRPO

Instead of PPO with a Critic, we optimize the policy $\pi_\theta$ by maximizing:
$$\mathbb{E}[ \frac{\pi_\theta(a|s)}{\pi_{old}(a|s)} \cdot \hat{A}(s,a) ]$$
Where the Advantage $\hat{A}$ for sample $i$ in a group of $G$ samples is:
$$\hat{A}_i = \frac{R_i - \text{mean}(R_{group})}{\text{std}(R_{group}) + \epsilon}$$

## 3\. The Training Loop (The "Multiverse")

### A. Group Sampling

For a given game state $S_t$:

1.  **Viewpoint:** Select a training power (e.g., FRANCE).
2.  **Sampling:** Generate $G$ independent outputs (e.g., $G=8$) from the current policy $\pi_\theta$.
3.  Each output consists of **Press** (Messages) and **Orders** (Moves).

### B. Rollout Mechanism (Truncated Horizon)

To evaluate each sample $i$ in $G$:

1.  **Step 0:** Execute the sampled orders/press for the Viewpoint Player.
2.  **Environment Agents:** The other 6 powers are controlled by a **Frozen Policy** (previous checkpoint or baseline bot) to isolate the Viewpoint's impact.
3.  **Horizon:** Simulate **2 Game Years** (Spring/Fall/Winter $\times$ 2).
      * *Rationale:* 1 year is too short to capture Supply Center (SC) changes; full games are too slow.

### C. Reward Function

The reward $R_i$ is a hybrid of Hard and Soft metrics at the end of the 2-year rollout.

$$R_{total} = w_1 R_{hard} + w_2 R_{soft}$$

**1. Hard Metrics (Objective State):**

  * **Solo Win:** +10.0
  * **Elimination:** -1.0
  * **Progress Score:**
    $$(N_{SCs} \times 1.0) + (N_{Units} \times 0.2) + (N_{Centers\_Owned} \times 0.1)$$

**2. Soft Metrics (LLM-as-a-Judge):**

  * Use a high-reasoning model (e.g., GPT-4o, Claude 3.5 Sonnet) to evaluate the final state.
  * **Prompt:** "Rate France's position on 0-1 scale based on: 1. Tactical Safety, 2. Alliance Reliability, 3. Expansion Potential."

-----

## 4\. Input/Output Architecture

### A. Input Schema (The "Egocentric View")

The LLM context must be strictly limited to what the specific power knows.
*Format: JSON*

```json
{
  "meta": { "role": "FRANCE", "season": "SPRING", "year": 1901 },
  "board_state": {
    "my_units": ["A PAR", "F BRE"],
    "opponents": { "ENGLAND": {"units": [...], "threat": "HIGH"} }
  },
  "valid_moves": {
    // CRITICAL: Used for Logit Masking
    "A PAR": ["HOLD", "MV BUR", "MV PIC"],
    "F BRE": ["HOLD", "MV MAO", "MV ENG"]
  },
  "message_history": [
    // Sliding window of last 2 years + Summary of previous eras
    { "sender": "ENGLAND", "content": "DMZ in Channel?" }
  ]
}
```

### B. Output Schema (Structured CoT & Tooling)

The model must output distinct sections for reasoning, intent, and action.

```xml
<analysis>
  England is asking for a DMZ, but their fleet build suggests aggression.
  I need to feign agreement while moving to a defensive spot.
</analysis>

<communication>
  <target>ENGLAND</target>
  <truth_status>LIE</truth_status> <message_content>Agreed. I will move my fleet south.</message_content>
</communication>

<orders>
  F BRE - ENG
  A PAR - PIC
</orders>
```

-----

## 5\. Inference Constraints (Implementation Critical)

### A. The "Lie Tool" (`truth_status`)

  * **Mechanism:** Explicitly training the model to tag its intent (`TRUTH` vs `LIE`) allows GRPO to reinforce *successful deception* specifically, rather than just vague communication.

### B. Order Tokenization (Logit Masking)

To ensure 100% validity of the $G$ samples (preventing wasted compute on syntax errors):

1.  **Do not** use standard sampling for the `<orders>` block.
2.  **Implementation:**
      * Parse the `valid_moves` JSON provided in input.
      * During autoregressive generation of orders, set logits of invalid tokens to $-\infty$.
      * *Example:* If generating ` F BRE -  `, mask all tokens except `[HOLD, MAO, ENG, PIC]`.

-----

## 6\. Implementation Stack Recommendations

  * **Game Engine:** `diplomacy` (Python package) for state management and adjacency rules.
  * **Training Framework:** HuggingFace `trl` (Transformer Reinforcement Learning) library supports GRPO; requires custom trainer class modification for the multiplayer rollout step.
  * **Judge Model:** Hosted API (OpenAI/Anthropic) or local quantization (e.g., Llama-3-70b) for the Soft Reward.