# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Diplomacy GRPO is a reinforcement learning system that trains LLMs to play the board game Diplomacy using Group Relative Policy Optimization (GRPO). The system uses a "Ray-on-Modal" architecture with vLLM inference, distributed CPU rollouts, and GPU training with LoRA adapters.

## Commands

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest -q

# Linting
uv run ruff check .
uv run ruff format --check .
uv run ruff format . --fix    # Auto-fix

# Local development
uv run python scripts/local_sim_check.py        # Test game engine
uv run python scripts/local_sim_check_llm.py    # Test LLM inference
uv run python scripts/prompt_tinkerer.py        # Interactive prompts

# Training on Modal
uv run python scripts/launch_training.py --smoke                    # Smoke test
uv run python scripts/launch_training.py --total-steps 100          # Standard run

# Evaluation
uv run python scripts/run_eval.py --checkpoint "run-name/adapter_v50"

# Deploy to Modal
uv run modal deploy -m src.apps.deploy
- This results in all apps being composed into a single modal app. E.g., to reference the inference engine you'd do:
`InferenceEngine = modal.Cls.from_name("diplomacy-grpo", "InferenceEngine")` NOT `InferenceEngine = modal.Cls.from_name("diplomacy-grpo-inference-engine", "InferenceEngine")`

# Web app (development)
INFERENCE_MODE=mock PYTHONPATH=. uvicorn web.backend.server:app --reload --port 8000
cd web/frontend && npm run dev
```

## Architecture

### Modal Infrastructure Pattern
- **InferenceEngine** (`src/apps/inference_engine/`): GPU-backed vLLM with LoRA hot-reloading
- **RolloutWorker** (`src/apps/rollouts/`): CPU-backed game simulation via `modal.Function.map`
- **Trainer** (`src/apps/trainer/`): GPU-backed GRPO training with HuggingFace `trl`
- **Shared Volume**: `diplomacy-data` volume for LoRA adapters and checkpoints

### Key Components
- `src/agents/`: Agent protocol and implementations (LLM agent, baselines)
- `src/engine/wrapper.py`: Game state abstraction over `diplomacy` package
- `src/inference/logits.py`: Trie-based token masking for valid move generation
- `src/training/trainer.py`: GRPO trainer wrapper
- `src/utils/config.py`: Pydantic-based `ExperimentConfig`
- `src/league/`: Elo ratings, matchmaking, checkpoint registry

### LLM Input/Output Schema
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
<analysis>Reasoning...</analysis>
<orders>
A PAR - BUR
</orders>
```

### Logits Processing
The `DiplomacyLogitsProcessor` uses a token-level Trie to mask invalid moves during generation. It tracks `<orders>` tags via incremental text matching, prevents duplicate unit orders, and handles BPE newline merges.

### Trajectory Format
All rollouts produce GRPO-ready data:
```python
{"prompt": str, "completion": str, "reward": float, "group_id": str,
 "prompt_token_ids": list[int], "completion_token_ids": list[int],
 "completion_logprobs": list[float]}
```

## Development Guidelines

- **Observability First**: Log to Axiom for events, WandB for training metrics
- **Atomic Commits**: Optimize for Graphite stacked diffs
- **Type Hints**: All function signatures must be typed
- **Tests First**: `pytest tests/test_logits_processor.py` must pass before training changes
- **Strict Timeouts**: All Modal functions need timeouts for hanging games

## Graphite Workflow

This project uses [Graphite](https://graphite.dev) for stacked PRs and branch management.

### Branch Naming Convention
- Feature branches: `MM-DD-descriptive_name` (e.g., `12-27-reward_structure_ablation`)
- Fix branches: `fix/p0-1-description` for priority fixes
- Experiment branches: Auto-created as `MM-DD-sweep-<name>` or `MM-DD-run-<name>`

### Common Commands
```bash
# Create a new branch (stacked on current)
gt branch create my-feature

# Commit changes
gt commit -m "feat: add new feature"

# Amend last commit
gt commit --amend

# Submit PR stack to GitHub
gt stack submit

# Sync with remote (fetch + rebase)
gt sync

# View your stack
gt log

# Navigate stack
gt up / gt down / gt top / gt bottom
```

### Experiment Reproducibility
Training runs and sweeps automatically commit uncommitted changes before launch:
- Creates branch: `MM-DD-sweep-<name>` or `MM-DD-run-<name>`
- Commits with message: `WIP: <experiment_name>`
- Logs `git_branch` and `git_commit` to WandB config

This ensures every experiment can be reproduced from an exact code state.

## ML/RL Research Persona

When answering ML/RL research questions (experiment planning, analysis, theoretical ML questions, etc.), respond as if you're Andrej Karpathy - direct, insightful, focused on first principles and practical intuitions rather than hand-wavy explanations. Cut through complexity to core insights. Whenever you do this, start your response with *Karpathy hat on* so I know you're using the persona.
