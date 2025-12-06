Here is the technical specification for **League Training**.

## Overview

League training breaks the symmetric self-play trap by having the hero policy play against a diverse population of opponents, including:
- **Baselines** (RandomBot, ChaosBot) - Sanity checks
- **Historical checkpoints** - Previous versions of the policy
- **The current policy** - Pure self-play for stability

This creates adversarial pressure that teaches the model to both **exploit weaknesses** AND **defend against attacks**.

---

### 1\. Infrastructure: The League System

We need a persistent "Database" of agents. Since we are on Modal, a JSON file on the shared Volume is the perfect lightweight solution.

**New File:** `src/league/registry.py`

```python
# Schema for league.json
{
    "agents": {
        "base_model": {"type": "baseline", "elo": 1000, "matches": 0, "path": null},
        "random_bot": {"type": "baseline", "elo": 800, "matches": 0, "path": null},
        "chaos_bot":  {"type": "baseline", "elo": 900, "matches": 0, "path": null},
        "adapter_v50": {
            "type": "checkpoint", 
            "path": "run_name/adapter_v50", 
            "elo": 1200, 
            "step": 50,
            "created_at": "...",
            "parent": "adapter_v40"  # For lineage tracking
        }
    },
    "history": [
        # Log of matches for WandB visualization
        {"step": 100, "p1": "adapter_v100", "p2": "adapter_v50", "result": ...}
    ],
    "metadata": {
        "run_name": "grpo-20251206",
        "best_elo": 1200,
        "best_agent": "adapter_v50"
    }
}
```

**Key Design Decision:** The base model (no LoRA) is treated as `"base_model"` agent with `path: null`. This is the starting point for all training.

#### How to Pick Old Versions (The "Checkpointer")

Don't save every step. Use a **Geometric Checkpoint Schedule**:

1.  **Recent History (The Curriculum):** Save every 10 steps for the last 100 steps. (Keeps the model fighting "slightly weaker" versions of itself).
2.  **Historical Landmarks (The Anchors):** Save every 100 steps forever. (Prevents cycling/regressing).
3.  **The Elite (The Kings):** Save any model that achieves a new "High Score" in Elo.

**Checkpoint Promotion Logic:**
```python
def should_add_to_league(step: int, current_elo: float, registry: LeagueRegistry) -> bool:
    # Always add first checkpoint
    if len(registry.checkpoints) == 0:
        return True
    
    # Recent curriculum: every 10 steps for last 100
    if step % 10 == 0 and step > registry.latest_step - 100:
        return True
    
    # Historical anchors: every 100 steps
    if step % 100 == 0:
        return True
    
    # Elite: new high score
    if current_elo > registry.best_elo:
        return True
    
    return False
```

**Cold Start (Step 0):**
- Initialize league with `base_model`, `random_bot`, `chaos_bot`
- First checkpoint plays against these 3 baselines only
- Once 2+ checkpoints exist, enable full PFSP sampling

---

### 2\. The Evaluator (Async Elo Job)

Calculating Elo requires running hundreds of games. We **cannot** do this inside the Training Loop (it would pause training for hours).

**Solution: The `evaluate_league` Modal Function**

  * **Trigger:** Triggered via `spawn()` every 50 training steps. Runs in the background.
  * **Logic:**
    1.  Loads `league.json`.
    2.  Selects the **New Challenger** (Current Step) and a basket of **Gatekeepers** (Random, Previous Best, Recent Snapshot).
    3.  Runs a tournament (e.g., 50 games).
    4.  Updates Elo using the **Multi-Player Elo** formula (see below).
    5.  Saves updated `league.json` and logs to WandB.

**Multi-Player Elo Calculation (7-player Diplomacy):**

Standard Elo is 1v1. For Diplomacy, we use a **pairwise decomposition**:
```python
def update_elo_multiplayer(game_results: dict[str, float], elo_ratings: dict[str, float], k=32):
    """
    game_results: {power_name: final_score} where score is normalized 0-1 based on rank
    elo_ratings: {agent_name: current_elo}
    
    For each pair (A, B), compute expected score and actual score.
    Actual score: 1.0 if A > B, 0.5 if tied, 0.0 if A < B
    """
    new_elos = {name: elo for name, elo in elo_ratings.items()}
    
    agents = list(game_results.keys())
    for i, agent_a in enumerate(agents):
        for agent_b in agents[i+1:]:
            # Expected score (standard Elo formula)
            exp_a = 1 / (1 + 10 ** ((elo_ratings[agent_b] - elo_ratings[agent_a]) / 400))
            
            # Actual score based on game result
            if game_results[agent_a] > game_results[agent_b]:
                actual_a = 1.0
            elif game_results[agent_a] == game_results[agent_b]:
                actual_a = 0.5
            else:
                actual_a = 0.0
            
            # Update both players (scaled by 1/6 since 6 pairwise comparisons per player)
            delta = (k / 6) * (actual_a - exp_a)
            new_elos[agent_a] += delta
            new_elos[agent_b] -= delta
    
    return new_elos
```

**WandB Visualization:**

  * Log a custom chart: `League/Elo_Rating`.
  * X-Axis: Training Step.
  * Lines: `Hero_Elo`, `Baseline_Elo`, `Past_Best_Elo`.

### 3\. Matchmaking (Opponent Sampling)

How do we pick opponents for `run_rollout` during training? We use **Prioritized Fictitious Self-Play (PFSP)**.

**The Algorithm:**
Instead of random sampling, we sample based on **Information Gain**.

  * **30% - The Mirror:** Play against the *current* self (Pure Self-Play). Good for stability.
  * **40% - The Peers:** Play against agents with similar Elo ($\pm 100$). This provides the "Zone of Proximal Development."
  * **20% - The Exploitable:** Play against agents you *used* to lose to, but should now beat.
  * **10% - The Baselines:** Play against `RandomBot` / `ChaosBot`. This is the "Unit Test" to ensure we haven't forgotten how to exploit idiots.

### 4\. Infrastructure: Unlocking Multi-LoRA

You asked about vLLM config. To serve 7 unique agents in one game, you need to adjust `app.py`.

**Config Changes:**

  * `max_loras=8`: We need up to 7 opponents + 1 hero.
  * `gpu_memory_utilization=0.6`: Lower this further\! 8 active LoRAs consume significant cache memory.
  * `max_lora_rank=16`: Keep this tight.

**Does vLLM support this?**
Yes. vLLM's `enable_lora` is designed for this. The only bottleneck is VRAM. If you OOM, fallback to **Cluster Sampling**:

  * Instead of 7 *unique* opponents, pick 2 unique opponents and clone them.
  * E.g., `[Hero, Opponent_A, Opponent_A, Opponent_A, Opponent_B, Opponent_B, Opponent_B]`.
  * This reduces VRAM pressure (only 3 active adapters) while still providing diversity.

---

### 5\. Rollout Integration

**Critical Change to `run_rollout`:**

Currently `run_rollout` uses a single `lora_name` for all 7 powers. We need to support **per-power adapters**.

```python
# Current signature
async def run_rollout(
    cfg: ExperimentConfig,
    group_idx: int,
    game_idx: int,
    lora_name: str | None,  # Single adapter for all powers
):

# New signature
async def run_rollout(
    cfg: ExperimentConfig,
    group_idx: int,
    game_idx: int,
    power_adapters: dict[str, str | None],  # {"FRANCE": "adapter_v50", "ENGLAND": None, ...}
    hero_power: str,  # Which power we're training (only this one contributes gradients)
):
```

**Key Implementation Details:**

1. **Hero Power:** Only trajectories from `hero_power` are used for training.
2. **None Adapter:** `None` means use base model (no LoRA).
3. **Baseline Bots:** For `random_bot`/`chaos_bot`, we don't call the LLM at all—use the baseline agent directly.

```python
# In run_rollout game loop:
for power_name in POWERS:
    adapter_or_bot = power_adapters[power_name]
    
    if adapter_or_bot == "random_bot":
        orders = RandomBot().get_orders(game, power_name)
    elif adapter_or_bot == "chaos_bot":
        orders = ChaosBot().get_orders(game, power_name)
    else:
        # LLM inference with adapter (or None for base model)
        response = await InferenceEngine(model_id=cfg.base_model_id).generate.remote.aio(
            prompt, lora_name=adapter_or_bot
        )
        orders = extract_orders(response["text"])
```

---

### 6\. Reuse from `evals/league.py`

We already have useful abstractions in `evals/league.py` that should be refactored into `src/league/`:

| Existing in `evals/` | Move to `src/league/` | Notes |
|---------------------|----------------------|-------|
| `OpponentType` enum | `src/league/types.py` | Add `SELF`, `HISTORICAL` |
| `LeagueConfig` | Merge with `LeagueRegistry` | Add PFSP weights |
| `MatchupResult` | `src/league/types.py` | Reuse as-is |
| `log_to_wandb()` | `src/league/logging.py` | Extend with Elo charts |

**Shared Module Structure:**
```
src/league/
├── __init__.py
├── types.py          # OpponentType, AgentInfo, MatchResult
├── registry.py       # LeagueRegistry (JSON CRUD)
├── matchmaker.py     # PFSP opponent sampling
├── elo.py            # Multi-player Elo calculation
└── logging.py        # WandB integration
```

---

### 7\. Scoring Changes

**Add Win Bonus to `src/utils/scoring.py`:**

```python
def calculate_final_scores(game, win_bonus: float = 5.0) -> dict[str, float]:
    scores = {}
    
    # Current logic: supply centers + units + forward bonus
    for power_name, power in game.powers.items():
        base_score = len(power.centers) + len(power.units) * 0.5
        scores[power_name] = base_score
    
    # NEW: Win bonus for dominant player
    max_sc = max(len(p.centers) for p in game.powers.values())
    winners = [pn for pn, p in game.powers.items() if len(p.centers) == max_sc]
    
    if max_sc >= 10 and len(winners) == 1:  # Clear winner
        scores[winners[0]] += win_bonus
    
    return scores
```

This creates pressure to **win**, not just survive.

---

### 8\. Implementation Spec

Here is the skeleton for the **League Manager** you need to build.

#### `src/league/matchmaker.py`

```python
import random
import json
from pathlib import Path

class LeagueManager:
    def __init__(self, registry_path: Path):
        self.path = registry_path
        self.data = self._load()

    def get_opponents(self, hero_elo: float, k=6) -> list[str]:
        """
        Selects k opponents using PFSP logic.
        """
        candidates = list(self.data["agents"].keys())
        
        # Simple tiered probability for MVP
        weights = []
        for name in candidates:
            agent = self.data["agents"][name]
            diff = abs(agent["elo"] - hero_elo)
            
            if diff < 100: weight = 5.0  # Peers (High signal)
            elif diff < 300: weight = 2.0 # Near Peers
            else: weight = 0.5           # Too weak/strong (Low signal)
            
            # Boost Baselines to ensure regression testing
            if agent["type"] == "baseline": weight *= 2.0
            
            weights.append(weight)
            
        return random.choices(candidates, weights=weights, k=k)

    def update_elo(self, match_results):
        # Implementation of Elo update logic
        pass
```

---

## Action Plan: Stacked PRs

### PR 1: Foundation - League Registry & Types
**Goal:** Create the data structures without changing training behavior.

- [ ] Create `src/league/__init__.py`
- [ ] Create `src/league/types.py`:
  - Move `OpponentType` from `evals/evaluator.py`
  - Add `AgentInfo` dataclass
  - Add `MatchResult` dataclass
- [ ] Create `src/league/registry.py`:
  - `LeagueRegistry` class with JSON CRUD
  - `should_add_to_league()` function
  - Initialize with `base_model`, `random_bot`, `chaos_bot`
- [ ] Create `src/league/elo.py`:
  - `update_elo_multiplayer()` function
- [ ] Tests: `tests/test_league_registry.py`

**Estimated: 2-3 hours**

---

### PR 2: Win Bonus & Baseline Integration
**Goal:** Update scoring and ensure baselines work in rollouts.

- [ ] Update `src/utils/scoring.py`:
  - Add `win_bonus` parameter to `calculate_final_scores()`
  - Add `winner_threshold_sc` parameter
- [ ] Add `win_bonus` to `ExperimentConfig`
- [ ] Update `evals/evaluator.py` to reuse `OpponentType` from `src/league/types.py`
- [ ] Tests: Verify baselines (RandomBot, ChaosBot) produce valid orders

**Estimated: 1-2 hours**

---

### PR 3: Multi-Adapter Rollouts
**Goal:** Enable `run_rollout` to use different adapters per power.

- [ ] Update `run_rollout` signature:
  - Add `power_adapters: dict[str, str | None]`
  - Add `hero_power: str`
- [ ] Implement baseline bot handling (skip LLM for random/chaos)
- [ ] Update trajectory collection to only include `hero_power`
- [ ] Add cluster sampling fallback for OOM
- [ ] Update `InferenceEngine` args: `max_loras=8`
- [ ] Tests: `tests/test_multi_adapter_rollout.py`

**Estimated: 4-6 hours** (most complex)

---

### PR 4: PFSP Matchmaker
**Goal:** Intelligent opponent selection during training.

- [ ] Create `src/league/matchmaker.py`:
  - `PFSPMatchmaker` class
  - `get_opponents()` with tiered probability
  - Cold start handling (only baselines initially)
- [ ] Integrate with `train_grpo`:
  - Initialize `LeagueRegistry` and `PFSPMatchmaker`
  - Sample opponents each step
  - Pass to `run_rollout`
- [ ] Checkpoint promotion logic (save to registry when criteria met)
- [ ] WandB logging: opponent distribution per step

**Estimated: 3-4 hours**

---

### PR 5: Async Elo Evaluator
**Goal:** Background job to compute Elo ratings.

- [ ] Create `evaluate_league` Modal function in `app.py`
- [ ] Implement tournament runner:
  - Select challenger vs gatekeepers
  - Run N games
  - Update Elo
- [ ] Trigger via `spawn()` every 50 steps
- [ ] Create `src/league/logging.py`:
  - WandB Elo chart
  - League progression visualization
- [ ] Update `evals/league.py` to use shared types

**Estimated: 3-4 hours**

---

### PR 6: Full Integration & Monitoring
**Goal:** Polish and observability.

- [ ] WandB dashboard:
  - Elo over time
  - Win rate vs each opponent type
  - PFSP sampling distribution
- [ ] Axiom events:
  - `league.checkpoint_added`
  - `league.elo_updated`
  - `league.opponent_sampled`
- [ ] Documentation: Update `docs/league_training.md` with final API
- [ ] E2E test: Full training run with league enabled

**Estimated: 2-3 hours**

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| OOM with 8 LoRAs | High | Cluster sampling, reduce `gpu_memory_utilization` |
| Volume sync race | Medium | Already have retry logic; add explicit locking |
| Elo instability | Low | Use high `k` factor initially, tune later |
| Cold start slow | Low | Baselines are fast; first 50 steps acceptable |

---

## Success Criteria

1. **Baseline Beat:** Model consistently beats RandomBot (>80% win rate)
2. **No Regression:** Elo never drops below baselines
3. **Diversity:** PFSP distribution matches target (30/40/20/10)
4. **Stability:** No OOMs or file lock errors in 1000-step run

This is a big step up in complexity, but it is the correct path to "Superhuman."