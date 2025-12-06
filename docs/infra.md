Here is the **Infrastructure & Engineering Specification**. Combined with the previous **Algorithm Spec**, this provides the complete blueprint for the coding agent.

-----

# Infrastructure & Engineering Specification: GRPO Diplomacy

## 1\. High-Level Stack Strategy

**Design Philosophy:** "Burst Compute." We utilize serverless concurrency for the massive CPU demands of simulating games (rollouts) and dedicated GPU instances for centralized inference and training.

  * **Compute/Orchestration:** **Modal** (Serverless Python).
  * **Inference Engine:** **vLLM** (Async Engine with Prefix Caching).
  * **Weight Management:** **LoRA** (PEFT) adapters hot-swapped via shared storage.
  * **Observability:** **Weights & Biases (W\&B)** (Metrics, Artifacts, HTML Logs).
  * **Configuration:** **Pydantic** (Strict schemas).

-----

## 2\. System Architecture (The "Ray-on-Modal" Pattern)

The system is composed of 4 distinct Modal objects that interact via **RPC** and **Shared Volumes**.

### A. Shared Storage (`modal.Volume`)

  * **Name:** `diplomacy-data`
  * **Purpose:** Acts as the synchronization layer between the Trainer and Inference Engine.
  * **Contents:**
      * `/models`: Stores LoRA adapter checkpoints (e.g., `step_100/adapter_model.bin`).
      * `/data`: Stores replay buffers (if async) and eval logs.

### B. The Inference Engine (`modal.Cls`)

  * **Hardware:** A100 (40GB/80GB) or H100.
  * **Role:** Centralized "Brain" for all agents (Learner + Opponents).
  * **Key Features:**
      * **Prefix Caching:** Enabled in vLLM to deduplicate the 90% overlapping history between turns.
      * **Dynamic LoRA Loading:** Must implement an endpoint to reload adapters without restarting the container.
      * **Logit Processing:** Must accept `valid_move_masks` in the request and apply them to generation to ensure 100% valid orders.

### C. The Rollout Workers (`modal.Function`)

  * **Hardware:** CPU (Standard).
  * **Scaling:** Map-reduce pattern (scale to 100+ concurrent containers).
  * **Role:** Runs the `diplomacy` Python engine.
  * **Constraint:** Strict timeouts (e.g., 600s). If a game hangs, the worker dies and the trajectory is dropped.

### D. The Trainer (`modal.Function`)

  * **Hardware:** H100 or A100.
  * **Role:** Runs the `trl` GRPO loop.
  * **Flow:**
    1.  Push new LoRA to Volume.
    2.  Signal Inference Engine to reload.
    3.  Trigger Rollout Workers via `map()`.
    4.  Collect trajectories.
    5.  Compute Loss & Update.

-----

## 3\. Evaluation Strategy

Separated into **Macro** (Statistical) and **Micro** (Behavioral) pipelines.

### A. The "League" (Macro)

A periodic job (every N steps) that updates the model's Elo/TrueSkill.

  * **Opponent Pool:**
      * `RandomBot`: Baseline sanity check.
      * `PrevSelf`: The model from N-50 steps.
      * `RuleBot`: Simple heuristic bot (if available).
  * **Metrics:** Win Rate, Share of Supply Centers (SOSC), Survival Rate.

### B. The "Microscope" (Micro/Vibe)

  * **Artifacts:** Every training run must generate `game_viewer` HTML logs uploaded to W\&B.
  * **Tactical Puzzles:** A suite of 20 fixed board states (JSON scenarios) where the optimal move is known (e.g., "Support hold to prevent cut").
      * *Pass Condition:* Model chooses the correct move \>80% of the time.

-----

## 4\. Code Standards & Primitives

We enforce strict interfaces to prevent "script spaghetti."

### A. The Agent Protocol

All players (LLM, Random, Hardcoded) must adhere to this interface:

```python
class DiplomacyAgent(Protocol):
    def get_orders(self, game: Game, power_name: str) -> List[str]:
        """Returns list of valid orders (e.g. 'A PAR - BUR')"""
        ...
        
    def get_press(self, game: Game, power_name: str) -> List[Message]:
        """Returns structured messages with intent tags"""
        ...
```

### B. Data Structures

```python
@dataclass
class Trajectory:
    """Immutable record of a rollout for training"""
    game_id: str
    # List of (Observation, Action, Reward) tuples
    steps: List[StepData] 
    final_score: float
    metadata: Dict[str, Any]

class ExperimentConfig(BaseModel):
    """Single source of truth for Hyperparams"""
    model_name: str = "mistralai/Mistral-7B-v0.1"
    rollout_horizon_years: int = 2
    n_samples_per_group: int = 8
    kl_coeff: float = 0.01
```

-----

## 5\. Development Roadmap (Benchmarks)

Do not proceed to the next level until the current Benchmark passes.

| Level | Name | Task | Metric |
| :--- | :--- | :--- | :--- |
| **0** | **Syntax** | Generating Orders/Press via Inference Engine. | 99% JSON Parse Rate. |
| **1** | **Semantics** | Generate orders with Logit Masking. | 0% Invalid Move Rate (Engine Rejection). |
| **2** | **Tactics** | Run "The Tactical Gym" (Simple Puzzles). | \>70% Success on known tactical problems. |
| **3** | **Dummy** | Full Game vs. `RandomBot` (2 Year Horizon). | \>80% Win/Draw Rate. |
| **4** | **Stability** | Run 50 concurrent games on Modal. | 0 Infrastructure Crashes / Timeouts handled. |
| **5** | **Learning** | Run GRPO loop. | Reward curve shows positive slope over 100 steps. |

-----

## 6\. Directory Structure

```text
diplomacy-grpo/
├── src/
│   ├── agents/          # Agent implementations (Random, LLM)
│   ├── engine/          # Wrapper for 'diplomacy' package & puzzle loader
│   ├── inference/       # vLLM Modal Class & Logit Processors
│   ├── trainer/         # TRL Logic & Modal Entrypoints
│   └── utils/           # Configs, HTML Rendering, Scoring Logic
├── evals/
│   ├── scenarios/       # JSON Tactical Puzzles
│   └── league.py        # TrueSkill logic
├── infra/               # Dockerfile definitions (if custom images needed)
├── pyproject.toml
└── app.py               # Main Modal Application
```