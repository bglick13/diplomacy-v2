from pathlib import Path

import modal

# ==============================================================================
# 1. IMAGE DEFINITIONS
# ==============================================================================

start_cmd = "pip install --upgrade pip"
requirements = [
    "diplomacy",
    "pydantic",
    "numpy",
    "tqdm",
    "cloudpickle",
]

# MODERN PATTERN: Use .add_local_python_source("src")
# This makes 'import src.engine.wrapper' work inside the container automatically.

cpu_image = (
    modal.Image.debian_slim()
    .run_commands(start_cmd)
    .pip_install(*requirements)
    .add_local_python_source("src")
)

cuda_version = "12.8.1"  # should be no greater than host CUDA version
flavor = "devel"  # includes full CUDA toolkit
operating_sys = "ubuntu24.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

gpu_image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .run_commands(start_cmd)
    .uv_pip_install(
        *requirements,
        "vllm",
        "torch",
        "transformers",
        "peft",
        "trl",
        "wandb",
        "accelerate",
    )
    .env({"VLLM_USE_V1": "1"})  # Enable vLLM v1 engine
    .add_local_python_source("src")
)

# ==============================================================================
# 2. STORAGE & APP SETUP
# ==============================================================================

app = modal.App("diplomacy-grpo")
volume = modal.Volume.from_name("diplomacy-data", create_if_missing=True)

VOLUME_PATH = Path("/data")
MODELS_PATH = VOLUME_PATH / "models"
REPLAYS_PATH = VOLUME_PATH / "replays"

# ==============================================================================
# 3. INFERENCE ENGINE (THE BRAIN)
# ==============================================================================


# TODO: Add hf cache volume
@app.cls(
    image=gpu_image,
    gpu="A100",
    volumes={str(VOLUME_PATH): volume},
    container_idle_timeout=300,
    concurrency_limit=5,
    allow_concurrent_inputs=100,
)
class InferenceEngine:
    @modal.enter()
    def setup(self):
        from transformers import AutoTokenizer
        from vllm import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        # Import Processor (Works because of add_local_python_source)
        from src.inference.logits import DiplomacyLogitsProcessor

        print("🥶 Initializing vLLM v1 Engine...")

        model_id = "Qwen/Qwen2.5-7B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # vLLM v1: Pass logits_processors at engine initialization
        # The processor is loaded once and handles all requests at batch level
        engine_args = AsyncEngineArgs(
            model=model_id,
            enable_lora=True,
            max_loras=4,
            gpu_memory_utilization=0.9,
            disable_log_stats=False,
            logits_processors=[DiplomacyLogitsProcessor],
        )
        self.engine = AsyncLLM.from_engine_args(engine_args)

        print("✅ Engine Ready.")

    @modal.method()
    async def generate(
        self, prompts: list[str], valid_moves: list[dict], lora_path: str | None = None
    ):
        import asyncio
        import uuid

        from vllm.lora.request import LoRARequest
        from vllm.sampling_params import SamplingParams

        lora_req = None
        if lora_path:
            adapter_id = str(hash(lora_path))
            lora_req = LoRARequest(adapter_id, 1, lora_path)

        async def _generate_single(prompt: str, moves: dict) -> str:
            """Generate for a single prompt. Allows concurrent execution."""
            request_id = str(uuid.uuid4())
            sampling_params = SamplingParams(
                temperature=0.7,
                max_tokens=200,
                extra_args={"valid_moves_dict": moves},
                stop=["</orders>", "</Orders>"],
            )
            generator = self.engine.generate(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=request_id,
                lora_request=lora_req,
            )
            final_output = None
            async for output in generator:
                final_output = output
            return final_output.outputs[0].text if final_output else ""

        # CRITICAL: Await all generators concurrently for proper batching
        # Old code awaited sequentially, killing vLLM's continuous batching
        final_texts = await asyncio.gather(
            *[_generate_single(p, m) for p, m in zip(prompts, valid_moves)]
        )

        return list(final_texts)


# ==============================================================================
# 4. ROLLOUT WORKER (THE SIMULATION)
# ==============================================================================


@app.function(
    image=cpu_image,
    cpu=1.0,
    memory=2048,  # Increased memory to hold G game copies
    timeout=3600,
    secrets=[modal.Secret.from_name("axiom-secrets")],
    volumes={str(VOLUME_PATH): volume},
)
async def run_rollout(config_dict: dict, lora_path: str | None = None):
    import random
    import time

    import cloudpickle

    from src.agents import LLMAgent
    from src.engine.wrapper import DiplomacyWrapper
    from src.utils.config import ExperimentConfig
    from src.utils.observability import (
        RolloutMetrics,
        axiom,
        log_orders_extracted,
        log_rollout_complete,
        log_rollout_error,
        log_rollout_start,
        logger,
        stopwatch,
    )
    from src.utils.parsing import extract_orders
    from src.utils.scoring import calculate_final_scores
    from src.utils.vis import GameVisualizer

    rollout_start_time = time.time()
    rollout_id = ""
    metrics: RolloutMetrics | None = None

    try:
        cfg = ExperimentConfig(**config_dict)
        should_visualize = random.random() < cfg.rollout_visualize_chance

        # Initialize the LLM agent for prompt generation
        agent = LLMAgent()

        # 1. THE WARMUP (Generate a random state)
        # ---------------------------------------
        warmup_phases = random.randint(0, 8)

        # Init main game
        main_game = DiplomacyWrapper(horizon=99)
        rollout_id = main_game.game.game_id
        metrics = RolloutMetrics(rollout_id=rollout_id)

        # Log rollout start
        log_rollout_start(
            rollout_id=rollout_id,
            warmup_phases=warmup_phases,
            samples_per_group=cfg.samples_per_group,
            horizon_years=cfg.rollout_horizon_years,
        )

        vis = None
        if should_visualize:
            logger.info("Visualizing game...")
            vis = GameVisualizer()
            vis.capture_turn(main_game.game, "Warmup Start")
        logger.info(f"🔥 Starting Warmup: {warmup_phases} phases")

        # Play through warmup
        for i in range(warmup_phases):
            if main_game.is_done():
                break

            with stopwatch(f"Warmup Inference {i + 1}/{warmup_phases}"):
                inputs = main_game.get_all_inputs(agent=agent)
                metrics.inference_calls += 1

                raw_responses = InferenceEngine().generate.remote(
                    prompts=inputs["prompts"],
                    valid_moves=inputs["valid_moves"],
                    lora_path=lora_path,
                )

            # Parse & Execute with observability
            all_orders = []
            phase = main_game.get_current_phase()

            for idx, (response, power) in enumerate(
                zip(raw_responses, inputs["power_names"])
            ):
                orders = extract_orders(response)
                expected_count = len(inputs["valid_moves"][idx])

                # Log extraction result
                log_orders_extracted(
                    rollout_id=rollout_id,
                    power_name=power,
                    orders_count=len(orders),
                    expected_count=expected_count,
                    raw_response_length=len(response),
                    phase=phase,
                    raw_response=response,
                )
                metrics.record_extraction(len(orders), expected_count)

                all_orders.extend(orders)

            main_game.step(all_orders)

            if should_visualize and vis:
                vis.capture_turn(
                    main_game.game,
                    f"Warmup step {i + 1}/{warmup_phases}\n{chr(10).join(all_orders)}",
                )

        if main_game.is_done():
            logger.info("Game ended during warmup, discarding.")
            return []

        # 2. THE FORK (Clone the state G times)
        # -------------------------------------
        frozen_state = cloudpickle.dumps(main_game)
        frozen_vis = cloudpickle.dumps(vis)
        if should_visualize and vis:
            # IMPORTANT: Create SEPARATE visualizer objects for each game
            # Using [obj] * n would create n references to the SAME object!
            visualizers = [
                cloudpickle.loads(frozen_vis) for _ in range(cfg.samples_per_group)
            ]
        else:
            visualizers = None

        games = [cloudpickle.loads(frozen_state) for _ in range(cfg.samples_per_group)]
        current_year = main_game.get_year()
        target_year = current_year + cfg.rollout_horizon_years

        fork_data = {i: {} for i in range(len(games))}
        active_indices = list(range(len(games)))
        is_fork_step = True
        step_count = 0

        # 3. THE GROUP ROLLOUT
        # --------------------------------------------------
        while active_indices:
            step_count += 1
            batch_prompts = []
            batch_valid_moves = []
            batch_meta = []  # (game_idx, power_name, expected_orders)

            for g_idx in active_indices:
                g = games[g_idx]
                if g.get_year() >= target_year or g.is_done():
                    continue

                inputs = g.get_all_inputs(agent=agent)

                for p_idx, power in enumerate(inputs["power_names"]):
                    batch_prompts.append(inputs["prompts"][p_idx])
                    batch_valid_moves.append(inputs["valid_moves"][p_idx])
                    expected = len(inputs["valid_moves"][p_idx])
                    batch_meta.append((g_idx, power, expected))

            if not batch_prompts:
                break

            # Inference call
            with stopwatch(
                f"Rollout Step {step_count} (Batch Size: {len(batch_prompts)})"
            ):
                metrics.inference_calls += 1
                responses = InferenceEngine().generate.remote(
                    prompts=batch_prompts,
                    valid_moves=batch_valid_moves,
                    lora_path=lora_path,
                )

            # Distribute responses with observability
            game_orders = {g_idx: [] for g_idx in active_indices}
            phase = (
                games[active_indices[0]].get_current_phase() if active_indices else ""
            )

            for i, response in enumerate(responses):
                g_idx, power, expected_count = batch_meta[i]
                orders = extract_orders(response)

                # Log extraction result
                log_orders_extracted(
                    rollout_id=rollout_id,
                    power_name=power,
                    orders_count=len(orders),
                    expected_count=expected_count,
                    raw_response_length=len(response),
                    phase=phase,
                    raw_response=response,
                )
                metrics.record_extraction(len(orders), expected_count)

                if is_fork_step:
                    fork_data[g_idx][power] = {
                        "prompt": batch_prompts[i],
                        "completion": response,
                    }

                game_orders[g_idx].extend(orders)

            # Step all games
            next_active = []
            for g_idx in active_indices:
                if g_idx in game_orders:
                    games[g_idx].step(game_orders[g_idx])
                    if should_visualize and visualizers is not None:
                        visualizers[g_idx].capture_turn(
                            games[g_idx].game,
                            f"Rollout step {step_count}\n{chr(10).join(game_orders[g_idx])}",
                        )
                    if not (
                        games[g_idx].get_year() >= target_year or games[g_idx].is_done()
                    ):
                        next_active.append(g_idx)

            active_indices = next_active
            is_fork_step = False

        # 4. SCORING & RETURN
        # -------------------
        trajectories = []

        for g_idx, game in enumerate(games):
            final_scores = calculate_final_scores(game)

            for power, data in fork_data[g_idx].items():
                if power in final_scores:
                    trajectories.append(
                        {
                            "prompt": data["prompt"],
                            "completion": data["completion"],
                            "reward": final_scores[power],
                            "group_id": f"{main_game.game.game_id}_{power}_{current_year}",
                        }
                    )

        # Save visualizations
        if should_visualize and vis and visualizers is not None:
            output_file = str(REPLAYS_PATH / f"rollout_{main_game.game.game_id}.html")
            vis.save_html(output_file)
            logger.info(f"✅ Saved replay to {output_file}")

            for g_idx, game in enumerate(games):
                output_file = str(
                    REPLAYS_PATH / f"rollout_{game.game.game_id}_{g_idx}.html"
                )
                visualizers[g_idx].save_html(output_file)
                logger.info(f"✅ Saved group replay {g_idx} to {output_file}")

        # Log successful completion
        total_duration_ms = int((time.time() - rollout_start_time) * 1000)
        log_rollout_complete(
            rollout_id=rollout_id,
            trajectories_count=len(trajectories),
            total_inference_calls=metrics.inference_calls,
            total_duration_ms=total_duration_ms,
        )
        metrics.log_summary()

        return trajectories

    except Exception as e:
        logger.error(f"❌ Error running rollout: {e}")
        if rollout_id:
            log_rollout_error(rollout_id=rollout_id, error=str(e))
        raise e

    finally:
        await axiom.flush()


# ==============================================================================
# 5. TRAINER (THE LEARNER)
# ==============================================================================


@app.function(
    image=gpu_image,
    gpu="H100",  # Need high VRAM for Training + Reference Model
    volumes={VOLUME_PATH: volume},
    timeout=86400,
    secrets=[modal.Secret.from_name("axiom-secrets")],
)
def train_grpo():
    import asyncio

    import numpy as np
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.training.loss import GRPOLoss
    from src.training.trainer import process_trajectories
    from src.utils.config import ExperimentConfig
    from src.utils.observability import axiom, stopwatch

    cfg = ExperimentConfig()
    print(f"🚀 Starting GRPO Loop: {cfg.run_name}")

    # 1. Load Models
    # We load the BASE model
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token  # Fix padding

    print("Loading Base Model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )

    # Create LoRA Adapter (The Policy)
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    policy_model = get_peft_model(base_model, peft_config)
    policy_model.print_trainable_parameters()

    # Reference Model: Just the base model (disable adapter)
    # Optimized: We reuse the same base model weights, just don't pass through LoRA layers
    # But for simplicity in custom loss, we can pass the same model and use a context manager
    # to disable adapters if supported, or simpler: keep 'ref_model' as base_model.
    # Actually, get_peft_model modifies in place.
    # To get reference logits, we use: with policy_model.disable_adapter(): ...

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=1e-5)
    loss_fn = GRPOLoss(
        policy_model, policy_model
    )  # We handle disable_adapter inside loss logic ideally

    # 2. Training Loop
    for step in range(cfg.total_steps):
        # A. Save Current Policy for Rollouts
        # We save to a versioned path so vLLM reloads it
        adapter_path = MODELS_PATH / cfg.run_name / f"adapter_v{step}"
        policy_model.save_pretrained(str(adapter_path))
        volume.commit()  # Sync to cloud storage

        print(f"\n=== Step {step}: Version {adapter_path} ===")

        # B. Launch Rollouts (The "E-Step")
        # We pass the NEW adapter path to the workers
        with stopwatch(f"Rollout_Step_{step}"):
            rollout_futures = run_rollout.map(
                [cfg.dict()] * cfg.num_groups_per_step,
                kwargs={"lora_path": str(adapter_path)},
            )

            raw_trajectories = []
            for res in rollout_futures:
                raw_trajectories.extend(res)

        print(f"Collected {len(raw_trajectories)} trajectories.")

        # C. Update Policy (The "M-Step")
        with stopwatch(f"Training_Step_{step}"):
            # Format data
            batch_data = process_trajectories(raw_trajectories, tokenizer)

            # Simple Mini-batching (Gradient Accumulation)
            # We just take one big step for simplicity, or chunk it
            # For H100, we might fit all 64 sequences? Let's try chunks of 4.

            chunk_size = 4
            accum_loss = 0
            optimizer.zero_grad()

            for i in range(0, len(batch_data), chunk_size):
                chunk = batch_data[i : i + chunk_size]
                if not chunk:
                    break

                # Move tensors to GPU
                for item in chunk:
                    item["input_ids"] = item["input_ids"].to(policy_model.device)
                    item["attention_mask"] = item["attention_mask"].to(
                        policy_model.device
                    )
                    item["labels"] = item["labels"].to(policy_model.device)

                # Compute Loss
                # Note: We need to adapt GRPOLoss to handle the 'disable_adapter' context
                # Update loss.py to use `with self.model.disable_adapter():` for ref pass
                loss, pg, kl = loss_fn.compute_loss(chunk)

                loss.backward()
                accum_loss += loss.item()

            optimizer.step()

            # Logs
            avg_loss = accum_loss / (len(batch_data) / chunk_size)
            print(f"Loss: {avg_loss:.4f} | KL: {kl:.4f}")

            # Send to Axiom/WandB
            axiom.log(
                {
                    "step": step,
                    "loss": avg_loss,
                    "kl": kl,
                    "reward_mean": np.mean([t["reward"] for t in raw_trajectories]),
                }
            )
            asyncio.run(axiom.flush())
