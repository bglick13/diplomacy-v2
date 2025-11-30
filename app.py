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

        model_id = "mistralai/Mistral-7B-v0.1"
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

    import cloudpickle

    from src.engine.wrapper import DiplomacyWrapper
    from src.utils.config import ExperimentConfig
    from src.utils.observability import (
        axiom,
        logger,
        stopwatch,
    )  # Make sure to mount/include this
    from src.utils.parsing import extract_orders
    from src.utils.scoring import calculate_final_scores
    from src.utils.vis import GameVisualizer

    try:
        cfg = ExperimentConfig(**config_dict)
        should_visualize = random.random() < cfg.rollout_visualize_chance

        # 1. THE WARMUP (Generate a random state)
        # ---------------------------------------
        # Randomly pick a start year between 0 and 4 (1901-1905)
        # We multiply by 2 because each year has roughly 2 move phases (Spring/Fall)
        warmup_phases = random.randint(0, 8)

        # Init main game
        main_game = DiplomacyWrapper(horizon=99)  # Infinite horizon for warmup
        vis = None
        if should_visualize:
            logger.info("Visualizing game...")
            vis = GameVisualizer()
            vis.capture_turn(main_game.game, "Warmup Start")
        logger.info(f"🔥 Starting Warmup: {warmup_phases} phases")

        # Play through warmup
        # Note: We use the same policy (the model) to generate the warmup
        # This ensures the mid-game states are "reachable" by the model.
        for i in range(warmup_phases):
            if main_game.is_done():
                break
            with stopwatch(f"Warmup Inference {i + 1}/{warmup_phases}"):
                inputs = main_game.get_all_inputs()
                logger.info(f"Prompts:\n{'\n'.join(inputs['prompts'])}")
                raw_responses = InferenceEngine().generate.remote(
                    prompts=inputs["prompts"],
                    valid_moves=inputs["valid_moves"],
                    lora_path=lora_path,
                )
            # Parse & Execute (Linear)
            all_orders = []
            for r in raw_responses:
                logger.info(f"Raw response: {r}")
                logger.info(f"Extracted orders: {extract_orders(r)}")
                all_orders.extend(extract_orders(r))
            main_game.step(all_orders)
            if should_visualize and vis:
                logger.info(f"Visualizing warmup step {i + 1}/{warmup_phases}")
                logger.info(f"Orders: {'\n'.join(all_orders)}")
                vis.capture_turn(
                    main_game.game,
                    f"Warmup step {i + 1}/{warmup_phases}\n{'\n'.join(all_orders)}",
                )

        if main_game.is_done():
            return []  # Game ended during warmup, discard.

        # 2. THE FORK (Clone the state G times)
        # -------------------------------------
        # We now have a specific state S. We want G samples from this S.

        # Cloudpickle handles local classes (like StringComparator in diplomacy package)
        # better than standard pickle
        frozen_state = cloudpickle.dumps(main_game)
        frozen_vis = cloudpickle.dumps(vis)
        if should_visualize and vis:
            visualizers = [cloudpickle.loads(frozen_vis)] * cfg.samples_per_group
        else:
            visualizers = None

        # Create G independent game objects
        games = [cloudpickle.loads(frozen_state) for _ in range(cfg.samples_per_group)]

        # Set the Horizon for these games (e.g. Current + 2 Years)
        current_year = main_game.get_year()
        target_year = current_year + cfg.rollout_horizon_years

        # Store the prompt/response for the fork point (The Training Data)
        # format: {game_index: {power_name: {prompt: ..., completion: ...}}}
        fork_data = {i: {} for i in range(len(games))}

        # 3. THE GROUP ROLLOUT (Simulate parallel universes)
        # --------------------------------------------------
        # We assume all G games move in lockstep phases for efficiency,
        # though eventually they might diverge in phase timing (rare but possible).

        active_indices = list(range(len(games)))  # Games still running

        # Mark the first step of the fork to capture the "Action"
        is_fork_step = True
        step_count = 0

        while active_indices:
            step_count += 1
            # Batching: We collect inputs from ALL active games
            batch_prompts = []
            batch_valid_moves = []
            # Metadata to map result back to (game_idx, power_name)
            batch_meta = []

            for g_idx in active_indices:
                g = games[g_idx]

                # Check horizon/done
                if g.get_year() >= target_year or g.is_done():
                    continue

                inputs = g.get_all_inputs()

                for p_idx, power in enumerate(inputs["power_names"]):
                    batch_prompts.append(inputs["prompts"][p_idx])
                    batch_valid_moves.append(inputs["valid_moves"][p_idx])
                    batch_meta.append((g_idx, power))

            # If no games are active, break
            if not batch_prompts:
                break

            # Massive Inference Call
            # We send G * 7 prompts at once.
            # vLLM handles this efficiently.
            with stopwatch(
                f"Rollout Step {step_count} (Batch Size: {len(batch_prompts)})"
            ):
                responses = InferenceEngine().generate.remote(
                    prompts=batch_prompts,
                    valid_moves=batch_valid_moves,
                    lora_path=lora_path,
                )

            # Distribute responses back to games
            game_orders = {g_idx: [] for g_idx in active_indices}

            for i, response in enumerate(responses):
                g_idx, power = batch_meta[i]

                # Capture data if this is the Fork Step (The one we train on)
                if is_fork_step:
                    fork_data[g_idx][power] = {
                        "prompt": batch_prompts[i],
                        "completion": response,
                    }

                orders = extract_orders(response)
                game_orders[g_idx].extend(orders)

            # Step all games
            next_active = []
            for g_idx in active_indices:
                if g_idx in game_orders:
                    games[g_idx].step(game_orders[g_idx])
                    if should_visualize and visualizers is not None:
                        visualizers[g_idx].capture_turn(
                            games[g_idx].game,
                            f"Rollout step {step_count}\n{'\n'.join(game_orders[g_idx])}",
                        )
                    # Check if it should continue next loop
                    if not (
                        games[g_idx].get_year() >= target_year or games[g_idx].is_done()
                    ):
                        next_active.append(g_idx)

            active_indices = next_active
            is_fork_step = False

        # 4. SCORING & RETURN
        # -------------------
        # Now we have G final states. We calculate rewards relative to the group.

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
                            "group_id": f"{main_game.game.game_id}_{power}_{current_year}",  # Unique ID for GRPO grouping
                        }
                    )
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

        return trajectories
    except Exception as e:
        logger.error(f"❌ Error running rollout: {e}")
        raise e
    finally:
        await axiom.flush()


# ==============================================================================
# 5. TRAINER (THE LEARNER)
# ==============================================================================


@app.function(
    image=gpu_image, gpu="H100", volumes={str(VOLUME_PATH): volume}, timeout=86400
)
def train_grpo():
    # ... (Trainer Logic Placeholder) ...
    pass
