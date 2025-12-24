# Diplomacy GRPO

A reinforcement learning system that trains LLMs to play the board game Diplomacy using Group Relative Policy Optimization (GRPO).

## Overview

Diplomacy GRPO uses a "Ray-on-Modal" architecture to train language models for strategic decision-making in Diplomacy, a 7-player negotiation and strategy game. The system combines:

- **vLLM inference** with LoRA hot-reloading for fast iteration
- **Distributed CPU rollouts** for parallel game simulation
- **GPU training** with GRPO and parameter-efficient fine-tuning
- **Trie-based constrained decoding** ensuring only valid moves are generated

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                        Modal Cloud                              │
├──────────────────┬──────────────────┬──────────────────────────┤
│  InferenceEngine │   RolloutWorker  │        Trainer           │
│     (GPU L4)     │    (CPU × N)     │      (GPU H100)          │
│   vLLM + LoRA    │  Game Simulation │    GRPO + LoRA           │
└────────┬─────────┴────────┬─────────┴───────────┬──────────────┘
         │                  │                     │
         └──────────────────┴─────────────────────┘
                            │
                   Shared Volume (diplomacy-data)
                   Models, checkpoints, replays
```

**Key Components:**
- `src/apps/inference_engine/` - GPU-backed vLLM with LoRA hot-reloading
- `src/apps/rollouts/` - CPU-backed game simulation via `modal.Function.map`
- `src/apps/trainer/` - GPU-backed GRPO training with HuggingFace `trl`
- `src/inference/logits.py` - Trie-based token masking for valid move generation
- `src/league/` - Elo ratings, matchmaking, checkpoint registry

## Features

- **GRPO Training** - Group Relative Policy Optimization for strategic reasoning
- **Constrained Decoding** - Token-level trie masking ensures only valid Diplomacy moves
- **LoRA Hot-Reloading** - Swap adapters without restarting inference servers
- **Elo Matchmaking** - League system with skill-based opponent selection
- **Web Interface** - Interactive play and trajectory collection
- **Sweep Framework** - Hyperparameter experiments with configurable ablations
- **Observability** - Axiom event logging + WandB experiment tracking

## LLM Input/Output Schema

**Input** (egocentric JSON):
```json
{
  "meta": {"role": "FRANCE", "season": "SPRING", "year": 1901},
  "board_state": {"my_units": ["A PAR"], "opponents": {...}},
  "valid_moves": {"A PAR": ["A PAR - BUR", "A PAR - MAR"]}
}
```

**Output** (structured XML):
```xml
<analysis>Reasoning about the strategic situation...</analysis>
<orders>
A PAR - BUR
</orders>
```

## Prerequisites

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** - Fast Python package manager
- **Node.js 18+** - For web frontend (optional)
- **Xcode Command Line Tools** - macOS/Apple Silicon only (for vLLM CPU build)

## Setup

### 1. Clone and Initialize

```bash
git clone https://github.com/bglick13/diplomacy-v2.git
cd diplomacy-v2
```

### 2. Git Subtree (external/vllm)

The `external/vllm` directory contains vLLM as a git subtree. It should already be present after cloning. If you need to update it:

```bash
git subtree pull --prefix=external/vllm https://github.com/vllm-project/vllm.git <commit> --squash
```

### 3. Install Python Dependencies

```bash
uv sync
```

### 4. vLLM on Apple Silicon (CPU-only)

<details>
<summary>Click to expand Apple Silicon setup</summary>

For local testing on Apple Silicon Macs:

**Requirements:**
- macOS Sonoma or later
- Xcode 15.4+ with Command Line Tools
- Apple Clang >= 15.0.0

**Build from source:**
```bash
cd external/vllm
uv pip install -r requirements/cpu.txt --index-strategy unsafe-best-match
uv pip install -e .
```

**Notes:**
- Only FP32 and FP16 are supported (no BF16 on Apple Silicon)
- Production inference uses Modal GPUs; local CPU is for testing only

</details>

## Modal Setup (Production/Cloud)

### 1. Install Modal CLI & Authenticate

```bash
pip install modal
modal token new
```

### 2. Create Required Secrets

Create these secrets in your [Modal workspace](https://modal.com/secrets):

**axiom-secrets** (for observability):
```bash
modal secret create axiom-secrets \
  AXIOM_TOKEN=<your-axiom-token> \
  AXIOM_ORG_ID=<your-axiom-org-id>
```

**wandb-secret** (for training metrics):
```bash
modal secret create wandb-secret \
  WANDB_API_KEY=<your-wandb-api-key>
```

### 3. Deploy to Modal

```bash
uv run modal deploy -m src.apps.deploy
```

This deploys all apps as a single Modal app named `diplomacy-grpo`.

## Web App Setup

**Backend:**
```bash
INFERENCE_MODE=mock PYTHONPATH=. uvicorn web.backend.server:app --reload --port 8000
```

**Frontend** (separate terminal):
```bash
cd web/frontend
npm install
npm run dev
```

**Inference modes:**
- `mock` - Random valid moves (no GPU required, for UI development)
- `modal` - Cloud inference via Modal (requires deployment)
- `local` - Local vLLM (requires CUDA GPU)

## Commands

```bash
# Run tests
uv run pytest -q

# Linting
uv run ruff check .
uv run ruff format --check .

# Local development
uv run python scripts/local_sim_check.py        # Test game engine
uv run python scripts/local_sim_check_llm.py    # Test LLM inference
uv run python scripts/prompt_tinkerer.py        # Interactive prompts

# Training on Modal
uv run python scripts/launch_training.py --smoke           # Smoke test
uv run python scripts/launch_training.py --total-steps 100 # Standard run

# Evaluation
uv run python scripts/run_eval.py --checkpoint "run-name/adapter_v50"
```

## Development

See [CLAUDE.md](./CLAUDE.md) for detailed development guidelines including:
- Architecture deep-dive
- Trajectory format specification
- Logits processing internals
- Coding standards and testing requirements
