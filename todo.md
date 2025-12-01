# Diplomacy GRPO Implementation Plan

## Level 0: Foundations & Syntax
- [ ] **Setup Project Structure**: Create folders `src/`, `tests/`, `infra/`.
- [ ] **Define Configs**: Implement `src/config.py` using Pydantic.
- [ ] **Implement Logits Processor**: Create `src/inference/logits.py` using the Trie-based code from `.cursor/rules`.
- [ ] **Add Unit Tests**: Create `tests/test_logits_processor.py` covering path following, dead ends, and newlines.
- [ ] **Verify Logic**: Run `pytest` and ensure 100% pass rate before moving on.

## Level 1: The Game Engine & Parsing
- [ ] **Implement Wrapper**: Create `src/engine/wrapper.py` to wrap the `diplomacy` package.
    - [ ] Ensure `get_valid_moves` returns strings matching the tokenizer's expected format.
- [ ] **Implement Parsing**: Create `src/utils/parsing.py` to extract `<orders>` and `<truth_status>`.
- [ ] **Local Simulation Check**: Write a small script to run a random game loop locally using these wrappers (no LLM yet, just random valid moves).

## Level 2: The Inference Engine (Modal + vLLM)
- [ ] **Create `app.py`**: Define CPU and GPU images.
- [ ] **Implement InferenceEngine**: Create the `modal.Cls` wrapping vLLM.
    - [ ] Add `setup()` to load the base model.
    - [ ] Add `generate()` method that accepts `valid_moves` and instantiates the `DiplomacyLogitsProcessor`.
- [ ] **RPC Test**: Create a simple Modal function `test_inference` that calls `InferenceEngine.generate` with a dummy prompt and asserts valid output.

## Level 3: The Rollout Loop
- [ ] **Implement Rollout Worker**: Create the `run_rollout` Modal function.
    - [ ] It should instantiate a game.
    - [ ] Loop for 2 years (Spring/Fall x 2).
    - [ ] Call `InferenceEngine.generate` via RPC for all 7 powers.
- [ ] **Implement Scoring**: Define the Reward Function (SC count + status).
- [ ] **Benchmark**: Run 50 concurrent rollouts on Modal and check for crashes/timeouts.

## Level 4: The GRPO Trainer
- [ ] **Implement Training Loop**: Create `train_grpo` in `src/training/trainer.py`.
    - [ ] Initialize `trl.GRPOTrainer`.
    - [ ] Implement the `save_lora` -> `volume.commit` -> `inference.reload` loop.
- [ ] **Connect Components**: Wire the Trainer to trigger the Rollout Workers via `map()`.
- [ ] **Dry Run**: Run a training loop for 5 steps with `batch_size=2` to verify the pipeline.

### Optimizations:
- [x] Pipeline Rollouts and Training
Currently: Rollout → Train → Rollout → Train
Better: Rollout[n+1] ↔ Train[n] (overlap them)
✅ Implemented using Modal's `spawn()` for async rollouts
- [ ] Async LoRA Loading
Preload the next adapter while current step trains.
- [ ] Gradient accumulation scaling issue → we divide by num_chunks but might need adjustment for larger batches

## Level 5: Evaluation & Polish
- [ ] **Tactical Gym**: Implement `evals/scenarios/` with 5 simple tactical puzzles.
- [ ] **League Eval**: Create a periodic job to play against `RandomBot`.
- [ ] **Visualization**: Add HTML export of game logs to Weights & Biases.