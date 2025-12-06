from pathlib import Path
from typing import Any

import modal

from src.utils.observability import log_inference_request, log_inference_response

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
    "wandb",
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

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

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
        "huggingface_hub",
        "hf_transfer",
        "pynvml",
    )
    .env(
        {
            "VLLM_USE_V1": "1",  # Enable vLLM v1 engine
            "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True",  # Allow dynamic LoRA loading
            "VLLM_PLUGINS": "lora_filesystem_resolver",  # Use built-in resolver
            "VLLM_LORA_RESOLVER_CACHE_DIR": "/data/models",  # Where adapters are stored
            "HF_HUB_ENABLE_HF_TRANSFER": "1",  # Use faster hf-transfer for downloads (if available)
        }
    )
    .add_local_python_source("src")
)

# ==============================================================================
# 2. STORAGE & APP SETUP
# ==============================================================================

app = modal.App("diplomacy-grpo")
volume = modal.Volume.from_name("diplomacy-data", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("hf-cache", create_if_missing=True)
trace_volume = modal.Volume.from_name("diplomacy-traces", create_if_missing=True)


VOLUME_PATH = Path("/data")
MODELS_PATH = VOLUME_PATH / "models"
REPLAYS_PATH = VOLUME_PATH / "replays"
BENCHMARKS_PATH = VOLUME_PATH / "benchmarks"
HF_CACHE_PATH = Path("/hf-cache")
TRACE_PATH = Path("/traces")

# ==============================================================================
# 3. INFERENCE ENGINE (THE BRAIN)
# ==============================================================================


@app.cls(
    image=gpu_image,
    gpu="A100",
    volumes={
        str(VOLUME_PATH): volume,
        str(HF_CACHE_PATH): hf_cache_volume,  # Cache HuggingFace models
    },
    container_idle_timeout=60 * 10,  # 10 minutes
    # Allow multiple containers for parallel sweeps/experiments
    # Each container can batch requests internally via vLLM
    # Trade-off: More containers = more parallelism but less batching per container
    concurrency_limit=1,
    allow_concurrent_inputs=200,  # Per-container concurrent request limit
)
class InferenceEngine:
    model_id: str = modal.parameter(default="Qwen/Qwen2.5-7B-Instruct")

    @modal.enter()
    def setup(self):
        import asyncio
        import os

        from transformers import AutoTokenizer
        from vllm import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        # Import Processor (Works because of add_local_python_source)
        from src.inference.logits import DiplomacyLogitsProcessor

        print("🥶 Initializing vLLM v1 Engine...")

        # Set HuggingFace cache to use volume for persistence across cold starts
        # This allows model files to persist even if container restarts
        # First run will download (~2-3min), subsequent runs use cache (~instant)
        os.environ["HF_HOME"] = str(HF_CACHE_PATH)
        os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE_PATH / "transformers")
        os.environ["HF_HUB_CACHE"] = str(HF_CACHE_PATH / "hub")

        # Ensure cache directories exist
        for cache_dir in [
            HF_CACHE_PATH / "transformers",
            HF_CACHE_PATH / "hub",
        ]:
            cache_dir.mkdir(parents=True, exist_ok=True)

        # Use cached tokenizer (faster than downloading)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            cache_dir=str(HF_CACHE_PATH / "transformers"),
        )

        # Track loaded adapters to avoid redundant volume reloads
        # This prevents concurrent reload conflicts
        self._loaded_adapters: set[str] = set()
        self._adapter_lock = asyncio.Lock()

        # vLLM v1: Pass logits_processors at engine initialization
        # The processor is loaded once and handles all requests at batch level
        # NOTE: max_lora_rank must match or exceed the rank used during training
        engine_args = AsyncEngineArgs(
            model=self.model_id,
            enable_lora=True,
            max_loras=4,
            max_lora_rank=16,  # Must match training LoRA rank
            gpu_memory_utilization=0.85,  # Higher utilization for better throughput
            max_num_seqs=256,  # Allow more concurrent sequences for better batching
            disable_log_stats=False,
            logits_processors=[DiplomacyLogitsProcessor],
            # Optimization: Use cached model location
            download_dir=str(HF_CACHE_PATH / "hub"),
            # Optimization: Prefer safetensors (faster loading than pickle)
            load_format="auto",  # Auto-detects best format (safetensors preferred)
            # Optimization: Skip warmup if not needed (set to False for production stability)
            # Note: Warmup helps with first-request latency but adds to startup time
            # disable_log_requests=True,  # Reduces logging overhead during startup
        )
        self.engine = AsyncLLM.from_engine_args(engine_args)

        print("✅ Engine Ready (with dynamic LoRA support).")

    @modal.method()
    async def generate(
        self, prompts: list[str], valid_moves: list[dict], lora_name: str | None = None
    ):
        """
        Generate responses for the given prompts.

        Args:
            prompts: List of prompt strings
            valid_moves: List of valid moves dicts for each prompt
            lora_name: Name of the LoRA adapter (relative to VLLM_LORA_RESOLVER_CACHE_DIR).
                       e.g., "benchmark-20251130/adapter_v1" will load from
                       /data/models/benchmark-20251130/adapter_v1
        """
        import asyncio
        import os
        import time
        import uuid

        from vllm.lora.request import LoRARequest
        from vllm.sampling_params import SamplingParams

        batch_size = len(prompts)
        request_start = time.time()
        log_inference_request(
            rollout_id="inference-engine",
            batch_size=batch_size,
            phase="engine",
            step_type="engine",
        )

        try:
            lora_req = None
            if lora_name:
                full_path = f"/data/models/{lora_name}"

                # Use lock to serialize volume reloads and prevent concurrent conflicts
                # Only reload if we haven't seen this adapter before
                if lora_name not in self._loaded_adapters:
                    async with self._adapter_lock:
                        # Double-check after acquiring lock (another request might have loaded it)
                        if lora_name not in self._loaded_adapters:
                            print(f"📂 Reloading volume for NEW adapter: {lora_name}")
                            volume.reload()

                            if not os.path.exists(full_path):
                                # List what's in models dir for debugging
                                models_dir = "/data/models"
                                if os.path.exists(models_dir):
                                    contents = os.listdir(models_dir)
                                    print(f"📁 Contents of {models_dir}: {contents}")
                                raise RuntimeError(f"LoRA path does not exist: {full_path}")

                            # List adapter files for confirmation
                            adapter_files = os.listdir(full_path)
                            print(f"✅ LoRA adapter found. Files: {adapter_files}")

                            # Mark as loaded so future requests skip the reload
                            self._loaded_adapters.add(lora_name)
                        else:
                            print(f"📂 Adapter already loaded by another request: {lora_name}")
                else:
                    print(f"📂 Using cached adapter: {lora_name}")

                # Create LoRA request (safe to do concurrently)
                lora_int_id = abs(hash(lora_name)) % (2**31)
                lora_req = LoRARequest(lora_name, lora_int_id, full_path)
                print(f"🔧 Created LoRARequest: name={lora_name}, id={lora_int_id}")

            async def _generate_single(prompt: str, moves: dict) -> dict[str, object]:
                """Generate for a single prompt. Allows concurrent execution."""
                request_id = str(uuid.uuid4())
                # vLLM SamplingParams accepts these at runtime but type stubs may be incomplete
                sampling_params = SamplingParams(  # type: ignore[call-arg, misc]
                    temperature=0.7,  # type: ignore[arg-type]
                    max_tokens=200,  # type: ignore[arg-type]
                    extra_args={"valid_moves_dict": moves},  # type: ignore[arg-type]
                    stop=["</orders>", "</Orders>"],  # type: ignore[arg-type]
                )
                try:
                    generator = self.engine.generate(
                        prompt=prompt,
                        sampling_params=sampling_params,
                        request_id=request_id,
                        lora_request=lora_req,
                    )
                    final_output = None
                    async for output in generator:
                        final_output = output
                    text = ""
                    token_count = 0
                    if final_output and final_output.outputs:
                        text = final_output.outputs[0].text
                        token_count = len(final_output.outputs[0].token_ids)
                    return {"text": text, "token_count": token_count}
                except Exception as e:
                    # Log on GPU side for debugging
                    print(f"❌ Generation Error: {e}")
                    raise e

            # CRITICAL: Await all generators concurrently for proper batching
            # Old code awaited sequentially, killing vLLM's continuous batching
            responses = await asyncio.gather(
                *[_generate_single(p, m) for p, m in zip(prompts, valid_moves, strict=True)]
            )

            duration_ms = int((time.time() - request_start) * 1000)
            total_tokens = sum(
                int(token_count)
                if isinstance(token_count := resp.get("token_count"), int | str)
                else 0
                for resp in responses
            )
            tokens_per_second = total_tokens / (duration_ms / 1000) if duration_ms > 0 else None
            log_inference_response(
                rollout_id="inference-engine",
                batch_size=batch_size,
                duration_ms=duration_ms,
                tokens_generated=total_tokens,
                tokens_per_second=tokens_per_second,
            )

            return [str(resp.get("text", "")) for resp in responses]
        except Exception as e:
            error_msg = f"GPU Inference Failed: {type(e).__name__}: {str(e)}"
            print(error_msg)
            raise RuntimeError(error_msg) from None


# ==============================================================================
# 4. ROLLOUT WORKER (THE SIMULATION)
# ==============================================================================


CURRENT_ROLLOUT_LORA: str | None = None


@app.function(
    image=cpu_image,
    cpu=1.0,
    memory=2048,  # Increased memory to hold G game copies
    timeout=3600,
    secrets=[modal.Secret.from_name("axiom-secrets")],
    volumes={str(VOLUME_PATH): volume},
)
async def run_rollout(config_dict: dict, lora_name: str | None = None):
    """
    Run a single rollout (game simulation) with the given config.

    Args:
        config_dict: ExperimentConfig as dict
        lora_name: Name of the LoRA adapter (relative to /data/models).
                   e.g., "benchmark-20251130/adapter_v1"
                   The vLLM filesystem resolver will find and load it.
    """
    import os
    import random
    import time

    import cloudpickle

    from src.agents import LLMAgent, PromptConfig
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

    # NOTE: We use vLLM's lora_filesystem_resolver to load adapters dynamically
    # The resolver looks in VLLM_LORA_RESOLVER_CACHE_DIR (/data/models) for the adapter
    # We just need to reload the volume to see newly committed adapter files
    if lora_name:
        logger.info(f"📂 Using LoRA adapter: {lora_name}")
        global CURRENT_ROLLOUT_LORA
        if CURRENT_ROLLOUT_LORA != lora_name:
            volume.reload()  # Ensure we see the latest adapter files
            full_path = f"/data/models/{lora_name}"
            if os.path.exists(full_path):
                files = os.listdir(full_path)
                logger.info(f"✅ LoRA adapter found at {full_path}. Files: {files}")
                CURRENT_ROLLOUT_LORA = lora_name
            else:
                logger.error(f"❌ LoRA adapter NOT found at: {full_path}")
        else:
            logger.info(f"📂 Adapter already cached locally: {lora_name}")

    rollout_start_time = time.time()
    rollout_id = ""
    metrics: RolloutMetrics | None = None

    try:
        cfg = ExperimentConfig(**config_dict)
        should_visualize = random.random() < cfg.rollout_visualize_chance

        # Initialize the LLM agent for prompt generation
        prompt_config = PromptConfig(compact_mode=cfg.compact_prompts)
        agent = LLMAgent(config=prompt_config)

        # 1. THE WARMUP (Generate a random state)
        # ---------------------------------------
        warmup_phases = random.randint(0, 8)
        if random.random() < cfg.rollout_no_warmup_chance:
            warmup_phases = 0

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

                raw_responses = await InferenceEngine().generate.remote.aio(
                    prompts=inputs["prompts"],
                    valid_moves=inputs["valid_moves"],
                    lora_name=lora_name,
                )

            # Parse & Execute with observability
            all_orders = []
            phase = main_game.get_current_phase()

            for idx, (response, power) in enumerate(
                zip(raw_responses, inputs["power_names"], strict=True)
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
            visualizers = [cloudpickle.loads(frozen_vis) for _ in range(cfg.samples_per_group)]
        else:
            visualizers = None

        games = [cloudpickle.loads(frozen_state) for _ in range(cfg.samples_per_group)]
        current_year = main_game.get_year()
        target_year = current_year + cfg.rollout_horizon_years

        fork_data = {i: {} for i in range(len(games))}

        # 3. THE GROUP ROLLOUT (ASYNC FORK PIPELINE)
        # --------------------------------------------------
        # Convert to async tasks per cloned game for concurrent inference + parsing
        import asyncio

        async def run_game_async(g_idx: int, game: DiplomacyWrapper, vis_obj) -> dict:
            """Run a single game clone asynchronously until completion."""
            game_fork_data = {}
            step_count = 0

            while game.get_year() < target_year and not game.is_done():
                step_count += 1

                # Get inputs for all powers in this game
                inputs = game.get_all_inputs(agent=agent)

                # Submit inference request for all powers concurrently
                with stopwatch(
                    f"Game {g_idx} Step {step_count} (Batch Size: {len(inputs['prompts'])})"
                ):
                    metrics.inference_calls += 1
                    responses = await InferenceEngine().generate.remote.aio(
                        prompts=inputs["prompts"],
                        valid_moves=inputs["valid_moves"],
                        lora_name=lora_name,
                    )

                # Parse responses and collect orders concurrently
                all_orders = []
                phase = game.get_current_phase()

                for idx, (response, power) in enumerate(
                    zip(responses, inputs["power_names"], strict=True)
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

                    # Store fork data on first step
                    if step_count == 1:
                        game_fork_data[power] = {
                            "prompt": inputs["prompts"][idx],
                            "completion": response,
                        }

                    all_orders.extend(orders)

                # Step the game
                game.step(all_orders)

                # Update visualization if enabled
                if should_visualize and vis_obj is not None:
                    vis_obj.capture_turn(
                        game.game,
                        f"Rollout step {step_count}\n{chr(10).join(all_orders)}",
                    )

            return {"g_idx": g_idx, "fork_data": game_fork_data}

        # Run all games concurrently
        with stopwatch("Async Group Rollout"):
            game_tasks = [
                run_game_async(g_idx, game, visualizers[g_idx] if visualizers else None)
                for g_idx, game in enumerate(games)
            ]
            game_results = await asyncio.gather(*game_tasks)

        # Collect fork data from all games
        for result in game_results:
            fork_data[result["g_idx"]] = result["fork_data"]

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
                output_file = str(REPLAYS_PATH / f"rollout_{game.game.game_id}_{g_idx}.html")
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
# 5. TRAINER (THE LEARNER) - Consolidated from train_grpo and train_grpo_benchmark
# ==============================================================================


# ==============================================================================
# 7. PROFILING HELPERS
# ==============================================================================


@app.function(
    image=cpu_image,
    volumes={str(VOLUME_PATH): volume},
    timeout=120,
    secrets=[modal.Secret.from_name("axiom-secrets")],
)
def persist_profile_snapshot(profile_name: str, payload: dict):
    """Persist profiling payload to /data/benchmarks for later analysis."""
    import json
    from datetime import datetime

    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    safe_name = profile_name.replace("/", "_")
    output_dir = BENCHMARKS_PATH
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{safe_name}-{timestamp}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    volume.commit()
    return {"path": str(output_path)}


# ==============================================================================
# 6. MAIN TRAINER FUNCTION
# ==============================================================================


@app.function(
    image=gpu_image,
    gpu="H100",
    volumes={str(VOLUME_PATH): volume, str(TRACE_PATH): trace_volume},
    timeout=60 * 60 * 24,  # 24 hours max
    secrets=[
        modal.Secret.from_name("axiom-secrets"),
        modal.Secret.from_name("wandb-secret"),
    ],
)
def train_grpo(config_dict: dict | None = None, **kwargs) -> dict:
    """
    Main GRPO training function.

    This is the unified training entrypoint that supports both production runs
    and profiling/benchmarking. All parameters come from ExperimentConfig.

    Args:
        config_dict: Optional dict of ExperimentConfig values. If provided,
                    these take precedence over kwargs.
        **kwargs: Individual config overrides (matched to ExperimentConfig fields).
                 Common overrides:
                 - total_steps: Number of training steps
                 - num_groups_per_step: Rollout groups per step (G in GRPO)
                 - samples_per_group: Samples per group (N in GRPO)
                 - rollout_horizon_years: Years to simulate per rollout
                 - learning_rate: Learning rate for AdamW optimizer
                 - profiling_mode: "rollout", "trainer", or "e2e" for profiling
                 - experiment_tag: Tag for grouping runs in WandB

    Returns:
        Dict with timing, throughput, and training metrics

    Example:
        # From CLI via Modal
        modal run app.py::train_grpo --total-steps 10 --learning-rate 1e-5

        # From Python
        train_grpo.remote(config_dict={"total_steps": 10, "learning_rate": 1e-5})
    """
    import asyncio
    import time
    from datetime import datetime

    import torch
    import wandb
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.training.loss import GRPOLoss
    from src.training.trainer import process_trajectories
    from src.utils.config import ExperimentConfig
    from src.utils.observability import GPUStatsLogger, axiom, logger, stopwatch

    # Build config from dict or kwargs
    # Priority: config_dict > kwargs > defaults
    config_values = {}
    if config_dict:
        config_values.update(config_dict)
    config_values.update({k: v for k, v in kwargs.items() if v is not None})

    # Generate run_name if not provided
    if "run_name" not in config_values or config_values["run_name"] == "diplomacy-grpo-v1":
        config_values["run_name"] = f"grpo-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    cfg = ExperimentConfig(**config_values)

    # Pre-compute simulated years metrics for power law analysis
    sim_years_per_step = cfg.simulated_years_per_step

    # Profiling setup
    profile_enabled = cfg.profiling_mode in {"trainer", "e2e"}
    profile_snapshots: list[dict[str, float]] = []

    from contextlib import contextmanager

    from torch.profiler import (
        ProfilerActivity,
        tensorboard_trace_handler,
    )
    from torch.profiler import (
        profile as torch_profile,
    )
    from torch.profiler import (
        schedule as profiler_schedule,
    )

    @contextmanager
    def profile_section(step_profile: dict[str, Any], name: str):
        if not profile_enabled:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            key = f"{name}_ms"
            step_profile[key] = step_profile.get(key, 0.0) + elapsed_ms

    # Metrics collection
    metrics = {
        "config": cfg.model_dump(),
        "step_metrics": [],
        "timing": {},
    }

    benchmark_start = time.time()
    gpu_logger = GPUStatsLogger()
    gpu_logger.start(context=f"train_grpo_benchmark:{cfg.run_name}")
    profiler = None
    trace_subdir = None
    if profile_enabled:
        traces_root = TRACE_PATH / "trainer"
        trace_subdir = traces_root / (cfg.profile_run_name or cfg.run_name)  # type: ignore[arg-type]
        trace_subdir.mkdir(parents=True, exist_ok=True)
        profiler = torch_profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=profiler_schedule(
                wait=1,
                warmup=1,
                active=max(1, cfg.profiling_trace_steps - 2),
                repeat=0,
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
            on_trace_ready=tensorboard_trace_handler(str(trace_subdir)),
        )
        profiler.__enter__()

    # Initialize WandB
    wandb_tags = []
    if cfg.experiment_tag:
        wandb_tags.append(cfg.experiment_tag)
    wandb.init(
        project=cfg.wandb_project,
        name=cfg.run_name,
        tags=wandb_tags if wandb_tags else None,
        config={
            **cfg.model_dump(),
            "simulated_years_per_step": sim_years_per_step,
            "total_simulated_years": cfg.total_simulated_years,
        },
    )

    try:
        # ==========================================================================
        # 1. Model Loading (Timed)
        # ==========================================================================
        logger.info(f"🚀 Starting GRPO Training: {cfg.run_name}")

        model_load_start = time.time()

        tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_id)
        tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",  # PyTorch native scaled dot product attention
        )

        peft_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_rank * 2,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        policy_model = get_peft_model(base_model, peft_config)

        # Enable gradient checkpointing to reduce memory (trades compute for memory)
        policy_model.gradient_checkpointing_enable()  # type: ignore[attr-defined]
        logger.info("✅ Gradient checkpointing enabled")

        # NOTE: torch.compile() with reduce-overhead mode uses CUDA graphs which
        # conflict with LoRA's dynamic tensor operations. Disabling for now.
        # See: https://github.com/huggingface/peft/issues/1043

        optimizer = torch.optim.AdamW(policy_model.parameters(), lr=cfg.learning_rate)
        loss_fn = GRPOLoss(policy_model, beta=0.04)  # type: ignore[arg-type]

        model_load_time = time.time() - model_load_start
        metrics["timing"]["model_load_s"] = model_load_time
        logger.info(f"✅ Model loaded in {model_load_time:.2f}s")

        # ==========================================================================
        # 2. Training Loop (BUFFERED PIPELINE)
        # ==========================================================================
        # Buffering strategy: Keep `buffer_depth` batches of rollouts in flight
        # - This ensures we always have rollouts ready even when they take
        #   longer than training (common with longer horizons)
        # - Higher buffer_depth = more rollouts in flight = less GPU idle time
        #   but also more "stale" trajectories (trained on older adapter)
        #
        # Timeline visualization (buffer_depth=3):
        #   Rollout[0] ──────────────────────►
        #   Rollout[1] ──────────────────────────────────────►
        #   Rollout[2] ──────────────────────────────────────────────────►
        #                                      Train[0] ────► Train[1] ────►
        #                                                     ↑ No GPU idle!
        #
        # Adapter versioning:
        # - All pre-launched batches use base model (no adapter trained yet)
        # - After step N training, new batches use adapter_v{N+1}
        # ==========================================================================
        from collections import deque

        total_trajectories = 0
        all_rewards = []
        buffer_depth = cfg.buffer_depth

        logger.info(f"🚀 Starting BUFFERED pipelined training loop (buffer_depth={buffer_depth})")

        # Pre-launch `buffer_depth` batches of rollouts
        # All use base model initially (no adapter trained yet)
        logger.info(f"Step 0: Pre-launching {buffer_depth} batches of rollouts with base model")

        # Each entry is (handles_list, lora_name_used)
        rollout_buffer: deque[tuple[list, str | None]] = deque()
        for _ in range(buffer_depth):
            handles = [
                run_rollout.spawn(cfg.model_dump(), lora_name=None)
                for _ in range(cfg.num_groups_per_step)
            ]
            rollout_buffer.append((handles, None))

        total_in_flight = buffer_depth * cfg.num_groups_per_step
        logger.info(
            f"📦 Buffer initialized: {buffer_depth} batches ({total_in_flight} rollouts) in flight"
        )

        for step in range(cfg.total_steps):
            step_start = time.time()
            step_metrics: dict[str, Any] = {"step": step}
            step_profile: dict[str, Any] | None = {"step": step} if profile_enabled else None

            # A. Wait for OLDEST rollouts (front of buffer)
            current_handles, current_lora_name = rollout_buffer.popleft()
            rollout_start = time.time()
            with stopwatch(f"Benchmark_Rollout_{step}"):
                raw_trajectories = []
                for handle in current_handles:
                    result = handle.get()  # Block until this rollout completes
                    raw_trajectories.extend(result)

            rollout_time = time.time() - rollout_start
            step_metrics["rollout_time_s"] = rollout_time
            step_metrics["raw_trajectories"] = len(raw_trajectories)
            step_metrics["rollout_lora"] = current_lora_name or "base_model"
            step_metrics["buffer_depth_actual"] = (
                len(rollout_buffer) + 1
            )  # +1 for the one we just popped
            if step_profile is not None:
                step_profile["rollout_time_ms"] = rollout_time * 1000

            # B. Launch NEW batch to maintain buffer (if not near the end)
            # This batch runs during training, keeping the pipeline full
            steps_remaining = cfg.total_steps - step - 1
            if steps_remaining >= buffer_depth:
                # Determine which adapter to use for the new batch
                # After step 0 training completes, we'll have adapter_v1
                if step >= 1:
                    adapter_rel_path = f"{cfg.run_name}/adapter_v{step}"
                    adapter_full_path = MODELS_PATH / cfg.run_name / f"adapter_v{step}"
                    policy_model.save_pretrained(str(adapter_full_path))
                    volume.commit()
                    new_lora_name = adapter_rel_path
                    logger.info(f"Saved adapter to {adapter_full_path}")
                else:
                    new_lora_name = None  # Still use base model

                target_step = step + buffer_depth
                logger.info(
                    f"🔀 Launching rollouts for step {target_step} "
                    f"(using {'base model' if not new_lora_name else new_lora_name})"
                )
                new_handles = [
                    run_rollout.spawn(cfg.model_dump(), lora_name=new_lora_name)
                    for _ in range(cfg.num_groups_per_step)
                ]
                rollout_buffer.append((new_handles, new_lora_name))

            if not raw_trajectories:
                logger.warning(f"Step {step}: No trajectories, skipping")
                metrics["step_metrics"].append(step_metrics)
                continue

            total_trajectories += len(raw_trajectories)
            all_rewards.extend([t["reward"] for t in raw_trajectories])

            # C. Process trajectories
            process_start = time.time()
            profile_target = step_profile if step_profile is not None else {}
            with profile_section(profile_target, "tokenize"):
                batch_data, traj_stats = process_trajectories(raw_trajectories, tokenizer)
            process_time = time.time() - process_start
            step_metrics["process_time_s"] = process_time
            step_metrics["processed_trajectories"] = len(batch_data)

            if not batch_data:
                logger.warning(f"Step {step}: No valid batches")
                metrics["step_metrics"].append(step_metrics)
                continue

            # D. Training
            training_start = time.time()
            optimizer.zero_grad()

            accum_loss = 0.0
            accum_kl = 0.0
            num_chunks = 0

            for i in range(0, len(batch_data), cfg.chunk_size):
                chunk = batch_data[i : i + cfg.chunk_size]
                if not chunk:
                    break

                section_profile = step_profile if step_profile is not None else {}
                with profile_section(section_profile, "loss_forward"):
                    loss_output = loss_fn.compute_loss(chunk)
                scaled_loss = loss_output.loss / max(1, len(batch_data) // cfg.chunk_size)
                with profile_section(section_profile, "backward"):
                    scaled_loss.backward()

                accum_loss += loss_output.loss.item()
                accum_kl += loss_output.kl
                num_chunks += 1

            section_profile = step_profile if step_profile is not None else {}
            with profile_section(section_profile, "optimizer_step"):
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    policy_model.parameters(), cfg.max_grad_norm
                ).item()
                optimizer.step()

            training_time = time.time() - training_start

            # Record step metrics
            avg_loss = accum_loss / max(1, num_chunks)
            avg_kl = accum_kl / max(1, num_chunks)

            step_metrics["training_time_s"] = training_time
            step_metrics["loss"] = avg_loss
            step_metrics["kl"] = avg_kl
            step_metrics["grad_norm"] = grad_norm
            step_metrics["reward_mean"] = traj_stats.reward_mean
            step_metrics["reward_std"] = traj_stats.reward_std
            step_metrics["total_tokens"] = traj_stats.total_tokens
            if step_profile is not None:
                step_profile["training_time_ms"] = training_time * 1000
                step_profile["process_time_ms"] = process_time * 1000
                step_profile["trajectories"] = len(batch_data)
                step_profile["tokens"] = traj_stats.total_tokens

            step_total = time.time() - step_start
            step_metrics["total_time_s"] = step_total

            metrics["step_metrics"].append(step_metrics)

            logger.info(
                f"Step {step}: loss={avg_loss:.4f} | kl={avg_kl:.4f} | "
                f"reward={traj_stats.reward_mean:.2f}±{traj_stats.reward_std:.2f} | "
                f"trajectories={len(batch_data)} | time={step_total:.2f}s"
            )

            # Calculate pipeline efficiency: how much time was hidden by overlap
            # If rollout_time < training_time, we got good overlap
            pipeline_overlap = max(0, training_time - rollout_time) if step > 0 else 0
            step_metrics["pipeline_overlap_s"] = pipeline_overlap
            if step_profile is not None:
                step_profile["pipeline_overlap_ms"] = pipeline_overlap * 1000
                profile_snapshots.append(step_profile)

            # Calculate cumulative simulated years for power law X-axis
            cumulative_sim_years = (step + 1) * sim_years_per_step

            wandb.log(
                {
                    "benchmark/step": step,
                    "benchmark/loss": avg_loss,
                    "benchmark/kl": avg_kl,
                    "benchmark/reward_mean": traj_stats.reward_mean,
                    "benchmark/reward_std": traj_stats.reward_std,
                    "benchmark/rollout_time_s": rollout_time,
                    "benchmark/training_time_s": training_time,
                    "benchmark/trajectories": len(batch_data),
                    "benchmark/grad_norm": grad_norm,
                    "benchmark/pipeline_overlap_s": pipeline_overlap,
                    # Power Law metrics (for X-axis comparison across runs)
                    "power_law/cumulative_simulated_years": cumulative_sim_years,
                    "power_law/simulated_years_per_step": sim_years_per_step,
                    "power_law/reward_at_compute": traj_stats.reward_mean,
                }
            )
            if profiler is not None:
                profiler.step()

        # Save final adapter
        final_adapter_path = MODELS_PATH / cfg.run_name / f"adapter_v{cfg.total_steps}"
        policy_model.save_pretrained(str(final_adapter_path))
        volume.commit()
        logger.info(f"Saved final adapter to {final_adapter_path}")

        # ==========================================================================
        # 3. Final Metrics
        # ==========================================================================
        total_time = time.time() - benchmark_start
        metrics["timing"]["total_s"] = total_time

        # Compute summary
        total_simulated_years = (
            cfg.total_steps
            * cfg.num_groups_per_step
            * cfg.samples_per_group
            * cfg.rollout_horizon_years
        )

        # Calculate total pipeline savings
        total_pipeline_overlap = sum(
            m.get("pipeline_overlap_s", 0) for m in metrics["step_metrics"]
        )

        metrics["summary"] = {
            "total_trajectories": total_trajectories,
            "total_simulated_years": total_simulated_years,
            "trajectories_per_second": total_trajectories / max(0.001, total_time),
            "simulated_years_per_second": total_simulated_years / max(0.001, total_time),
            "reward_mean": sum(all_rewards) / len(all_rewards) if all_rewards else 0,
            "reward_min": min(all_rewards) if all_rewards else 0,
            "reward_max": max(all_rewards) if all_rewards else 0,
            "pipeline_overlap_total_s": total_pipeline_overlap,
            "run_name": cfg.run_name,
            "profiling_mode": cfg.profiling_mode,
            "profile_snapshots": profile_snapshots if profile_enabled else None,
            "trace_dir": str(trace_subdir) if trace_subdir else None,
        }

        # Get final step metrics
        if metrics["step_metrics"]:
            final = metrics["step_metrics"][-1]
            metrics["summary"]["final_loss"] = final.get("loss")
            metrics["summary"]["final_kl"] = final.get("kl")
            metrics["summary"]["final_reward_mean"] = final.get("reward_mean")

        logger.info(f"\n{'=' * 60}")
        logger.info("🏁 BENCHMARK COMPLETE")
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(f"Trajectories: {total_trajectories}")
        logger.info(f"Throughput: {metrics['summary']['trajectories_per_second']:.2f} traj/s")
        logger.info(f"Pipeline overlap (time saved): {total_pipeline_overlap:.2f}s")
        logger.info(f"{'=' * 60}")

        wandb.log({"benchmark/complete": True, **metrics["summary"]})

        return metrics["summary"]

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        wandb.log({"benchmark/error": str(e)})
        raise

    finally:
        gpu_logger.stop()
        if profiler is not None:
            profiler.__exit__(None, None, None)
        asyncio.run(axiom.flush())
        wandb.finish()


# ==============================================================================
# 8. POWER LAWS SWEEP ORCHESTRATOR
# ==============================================================================


@app.function(
    image=cpu_image,
    timeout=86400,  # 24 hours max for long sweeps
    secrets=[modal.Secret.from_name("axiom-secrets")],
)
def run_power_laws_sweep(
    total_steps: int = 100,
    num_groups_per_step: int = 8,
    learning_rate: float = 1e-5,
    model_id: str = "Qwen/Qwen2.5-7B-Instruct",
    run_configs: list[str] | None = None,  # ["A", "B", "C"] or None for all
    parallel: bool = False,
) -> dict:
    """
    Orchestrate the Power Laws experiment entirely on Modal.

    This function runs in the cloud, so you can close your laptop after launching.
    Progress is logged to Axiom and results are returned when complete.

    Args:
        total_steps: Training steps per configuration
        num_groups_per_step: Rollout groups per step
        learning_rate: Optimizer learning rate
        model_id: Model to use for inference
        run_configs: List of configs to run ["A", "B", "C"], or None for all
        parallel: If True, run configs in parallel (3x cost, 3x faster)

    Returns:
        Dict with results from all configurations and analysis
    """
    import time
    from datetime import datetime

    from src.utils.observability import axiom, logger

    # Define the three experimental configurations
    SWEEP_CONFIGS = {
        "A": {
            "name": "baseline",
            "tag": "power-laws-baseline",
            "rollout_horizon_years": 2,
            "samples_per_group": 8,
            "compute_multiplier": 1.0,
            "description": "Baseline: Fast & Loose (horizon=2, samples=8)",
        },
        "B": {
            "name": "deep-search",
            "tag": "power-laws-deep",
            "rollout_horizon_years": 4,
            "samples_per_group": 8,
            "compute_multiplier": 2.0,
            "description": "Deep Search: Time Scaling (horizon=4, samples=8)",
        },
        "C": {
            "name": "broad-search",
            "tag": "power-laws-broad",
            "rollout_horizon_years": 2,
            "samples_per_group": 16,
            "compute_multiplier": 2.0,
            "description": "Broad Search: Variance Scaling (horizon=2, samples=16)",
        },
    }

    # Determine which configs to run
    if run_configs is None:
        run_configs = ["A", "B", "C"]

    configs_to_run = [SWEEP_CONFIGS[k] for k in run_configs if k in SWEEP_CONFIGS]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    logger.info("=" * 60)
    logger.info("🔬 POWER LAWS SWEEP STARTING (Cloud Orchestrated)")
    logger.info("=" * 60)
    logger.info(f"Timestamp: {timestamp}")
    logger.info(f"Total Steps: {total_steps}")
    logger.info(f"Groups/Step: {num_groups_per_step}")
    logger.info(f"Model: {model_id}")
    logger.info(f"Parallel: {parallel}")
    logger.info(f"Configs: {run_configs}")

    # Log sweep start to Axiom
    axiom.log(
        {
            "event": "power_laws_sweep_start",
            "timestamp": timestamp,
            "total_steps": total_steps,
            "num_groups_per_step": num_groups_per_step,
            "model_id": model_id,
            "parallel": parallel,
            "configs": run_configs,
        }
    )

    sweep_start = time.time()
    results = []

    def run_single_config(config: dict) -> dict:
        """Run a single configuration and return results."""
        config_start = time.time()
        sim_years = (
            num_groups_per_step
            * config["samples_per_group"]
            * config["rollout_horizon_years"]
            * total_steps
        )

        logger.info(f"\n🚀 Starting: {config['description']}")
        logger.info(f"   Simulated Years: {sim_years}")

        # Log config start
        axiom.log(
            {
                "event": "power_laws_config_start",
                "config_name": config["name"],
                "timestamp": timestamp,
                "simulated_years": sim_years,
            }
        )

        try:
            result = train_grpo.remote(
                config_dict={
                    "total_steps": total_steps,
                    "num_groups_per_step": num_groups_per_step,
                    "samples_per_group": config["samples_per_group"],
                    "rollout_horizon_years": config["rollout_horizon_years"],
                    "learning_rate": learning_rate,
                    "rollout_visualize_chance": 0.0,
                    "compact_prompts": True,
                    "experiment_tag": config["tag"],
                    "base_model_id": model_id,
                }
            )

            duration = time.time() - config_start

            # Log config complete
            axiom.log(
                {
                    "event": "power_laws_config_complete",
                    "config_name": config["name"],
                    "timestamp": timestamp,
                    "duration_s": duration,
                    "final_reward_mean": result.get("final_reward_mean"),
                    "final_loss": result.get("final_loss"),
                }
            )

            logger.info(f"✅ {config['name']} complete in {duration:.1f}s")
            logger.info(f"   Final Reward: {result.get('final_reward_mean', 'N/A')}")

            return {
                "config": config,
                "result": result,
                "duration_s": duration,
                "simulated_years": sim_years,
            }

        except Exception as e:
            logger.error(f"❌ {config['name']} failed: {e}")
            axiom.log(
                {
                    "event": "power_laws_config_error",
                    "config_name": config["name"],
                    "error": str(e),
                }
            )
            return {
                "config": config,
                "error": str(e),
                "duration_s": time.time() - config_start,
                "simulated_years": sim_years,
            }

    if parallel:
        # Run all configs in parallel using spawn
        logger.info("\n🔀 Running configs in PARALLEL...")
        handles = [
            (
                config,
                train_grpo.spawn(
                    config_dict={
                        "total_steps": total_steps,
                        "num_groups_per_step": num_groups_per_step,
                        "samples_per_group": config["samples_per_group"],
                        "rollout_horizon_years": config["rollout_horizon_years"],
                        "learning_rate": learning_rate,
                        "rollout_visualize_chance": 0.0,
                        "compact_prompts": True,
                        "experiment_tag": config["tag"],
                        "base_model_id": model_id,
                    }
                ),
                time.time(),
            )
            for config in configs_to_run
        ]

        for config, handle, start_time in handles:
            sim_years = (
                num_groups_per_step
                * config["samples_per_group"]
                * config["rollout_horizon_years"]
                * total_steps
            )
            try:
                result = handle.get()
                duration = time.time() - start_time
                results.append(
                    {
                        "config": config,
                        "result": result,
                        "duration_s": duration,
                        "simulated_years": sim_years,
                    }
                )
                logger.info(f"✅ {config['name']} complete")
            except Exception as e:
                results.append(
                    {
                        "config": config,
                        "error": str(e),
                        "duration_s": time.time() - start_time,
                        "simulated_years": sim_years,
                    }
                )
    else:
        # Run configs sequentially
        logger.info("\n📋 Running configs SEQUENTIALLY...")
        for config in configs_to_run:
            result = run_single_config(config)
            results.append(result)

    # Analyze results
    total_duration = time.time() - sweep_start

    logger.info("\n" + "=" * 60)
    logger.info("📊 POWER LAWS SWEEP COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total Duration: {total_duration / 3600:.2f} hours")

    # Build comparison table
    valid_results = [r for r in results if "result" in r and r["result"]]
    comparison = []

    for r in results:
        config = r["config"]
        entry = {
            "name": config["name"],
            "description": config["description"],
            "compute_multiplier": config["compute_multiplier"],
            "simulated_years": r["simulated_years"],
            "duration_s": r["duration_s"],
        }
        if "result" in r and r["result"]:
            entry["final_reward_mean"] = r["result"].get("final_reward_mean")
            entry["final_loss"] = r["result"].get("final_loss")
            entry["final_kl"] = r["result"].get("final_kl")
            entry["run_name"] = r["result"].get("run_name")
        else:
            entry["error"] = r.get("error", "Unknown error")
        comparison.append(entry)

    # Determine winner
    analysis = {"winner": None, "interpretation": ""}
    if valid_results:
        best = max(
            valid_results, key=lambda r: r["result"].get("final_reward_mean") or float("-inf")
        )
        best_name = best["config"]["name"]
        analysis["winner"] = best_name
        analysis["best_reward"] = best["result"].get("final_reward_mean")

        if best_name == "deep-search":
            analysis["interpretation"] = (
                "HORIZON SCALING WINS: Simple reward works, increase rollout_horizon_years"
            )
        elif best_name == "broad-search":
            analysis["interpretation"] = (
                "VARIANCE SCALING WINS: Simple reward works, increase samples_per_group"
            )
        else:
            analysis["interpretation"] = (
                "BASELINE WINS: Scaling alone insufficient, consider reward engineering"
            )

        logger.info(f"\n🏆 Winner: {best_name}")
        logger.info(f"   {analysis['interpretation']}")

    # Log sweep complete
    axiom.log(
        {
            "event": "power_laws_sweep_complete",
            "timestamp": timestamp,
            "total_duration_s": total_duration,
            "winner": analysis.get("winner"),
            "results_count": len(results),
        }
    )

    import asyncio

    asyncio.run(axiom.flush())

    return {
        "timestamp": timestamp,
        "total_duration_s": total_duration,
        "total_duration_hours": total_duration / 3600,
        "comparison": comparison,
        "analysis": analysis,
        "config": {
            "total_steps": total_steps,
            "num_groups_per_step": num_groups_per_step,
            "learning_rate": learning_rate,
            "model_id": model_id,
            "parallel": parallel,
        },
    }


# ==============================================================================
# 9. EVALUATION RUNNER
# ==============================================================================

EVALS_PATH = VOLUME_PATH / "evals"


@app.function(
    image=cpu_image,
    timeout=86400,  # 24 hours max
    secrets=[
        modal.Secret.from_name("axiom-secrets"),
        modal.Secret.from_name("wandb-secret"),
    ],
    volumes={str(VOLUME_PATH): volume},
)
async def run_evaluation(
    checkpoint_path: str,
    opponents: list[str] | None = None,
    games_per_opponent: int = 10,
    max_years: int = 10,
    eval_powers: list[str] | None = None,
    model_id: str = "Qwen/Qwen2.5-7B-Instruct",
    visualize: bool = True,
    visualize_sample_rate: float = 0.3,
    log_to_wandb: bool = True,
    wandb_run_name: str | None = None,
) -> dict:
    """
    Evaluate a checkpoint against various opponents.

    This function runs evaluation games and logs results to WandB with
    HTML visualizations of sample games.

    Args:
        checkpoint_path: Path to checkpoint (relative to /data/models)
                        e.g., "benchmark-20251205/adapter_v10"
        opponents: List of opponent types ["random", "chaos"]
        games_per_opponent: Number of games per opponent type
        max_years: Maximum game length in years
        eval_powers: Which powers use the checkpoint (default: ["FRANCE"])
        model_id: Base model ID
        visualize: Whether to generate HTML visualizations
        visualize_sample_rate: Fraction of games to visualize
        log_to_wandb: Whether to log results to WandB
        wandb_run_name: Optional WandB run name

    Returns:
        Dict with evaluation metrics and visualization paths
    """
    import os
    import random
    import time
    from datetime import datetime

    import wandb

    from src.agents import LLMAgent, PromptConfig
    from src.agents.baselines import ChaosBot, RandomBot
    from src.engine.wrapper import DiplomacyWrapper
    from src.utils.observability import axiom, logger
    from src.utils.parsing import extract_orders
    from src.utils.vis import GameVisualizer

    # Ensure evals directory exists
    EVALS_PATH.mkdir(parents=True, exist_ok=True)

    # Default opponents
    if opponents is None:
        opponents = ["random", "chaos"]

    # Default eval powers (FRANCE for consistency)
    if eval_powers is None:
        eval_powers = ["FRANCE"]

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_name = checkpoint_path.split("/")[-1]

    logger.info("=" * 60)
    logger.info(f"🎯 EVALUATION: {checkpoint_path}")
    logger.info("=" * 60)
    logger.info(f"Opponents: {opponents}")
    logger.info(f"Games per opponent: {games_per_opponent}")
    logger.info(f"Eval powers: {eval_powers}")
    logger.info(f"Max years: {max_years}")

    # Log to Axiom
    axiom.log(
        {
            "event": "evaluation_start",
            "checkpoint_path": checkpoint_path,
            "opponents": opponents,
            "games_per_opponent": games_per_opponent,
            "timestamp": timestamp,
        }
    )

    # Initialize WandB
    if log_to_wandb:
        run_name = wandb_run_name or f"eval-{checkpoint_name}-{timestamp}"
        wandb.init(
            project="diplomacy-grpo",
            name=run_name,
            tags=["evaluation"],
            config={
                "checkpoint_path": checkpoint_path,
                "opponents": opponents,
                "games_per_opponent": games_per_opponent,
                "max_years": max_years,
                "eval_powers": eval_powers,
                "model_id": model_id,
            },
        )

    # Initialize LLM agent
    prompt_config = PromptConfig(compact_mode=True)
    llm_agent = LLMAgent(config=prompt_config)

    # Results storage
    all_results = []
    all_visualizations = []

    start_time = time.time()

    for opponent_type in opponents:
        logger.info(f"\n📊 Running {games_per_opponent} games vs {opponent_type}...")

        # Select opponent agent
        if opponent_type == "random":
            opponent_agent = RandomBot()
        elif opponent_type == "chaos":
            opponent_agent = ChaosBot()
        else:
            logger.warning(f"Unknown opponent type: {opponent_type}, using random")
            opponent_agent = RandomBot()

        opponent_results = []

        for game_idx in range(games_per_opponent):
            random.seed(42 + game_idx)  # Reproducible

            # Initialize game
            game = DiplomacyWrapper(horizon=max_years * 2)
            game_id = game.game.game_id

            # Visualization for this game
            should_viz = visualize and (random.random() < visualize_sample_rate)
            vis = GameVisualizer() if should_viz else None
            if vis:
                vis.capture_turn(game.game, f"Game Start vs {opponent_type}")

            # Track metrics
            checkpoint_centers_history = []
            all_powers = list(game.game.powers.keys())
            checkpoint_power_set = set(eval_powers)

            # Game loop
            year = 0
            while not game.is_done() and year < max_years:
                year = game.get_year() - 1901
                all_orders = []

                # Collect prompts for checkpoint powers
                checkpoint_prompts = []
                checkpoint_valid_moves = []
                active_checkpoint_powers = []

                for power_name in all_powers:
                    power = game.game.powers[power_name]
                    if len(power.units) == 0 and len(power.centers) == 0:
                        continue  # Eliminated

                    if power_name in checkpoint_power_set:
                        prompt, valid_moves = llm_agent.build_prompt(game, power_name)
                        checkpoint_prompts.append(prompt)
                        checkpoint_valid_moves.append(valid_moves)
                        active_checkpoint_powers.append(power_name)
                    else:
                        orders = opponent_agent.get_orders(game, power_name)
                        all_orders.extend(orders)

                # Batch inference for checkpoint powers
                if checkpoint_prompts:
                    responses = await InferenceEngine(model_id=model_id).generate.remote.aio(
                        prompts=checkpoint_prompts,
                        valid_moves=checkpoint_valid_moves,
                        lora_name=checkpoint_path,
                    )

                    for response, valid_moves in zip(
                        responses, checkpoint_valid_moves, strict=True
                    ):
                        orders = extract_orders(response)
                        if not orders:
                            # Fallback: use first valid move for each unit
                            orders = [moves[0] for moves in valid_moves.values() if moves]
                        all_orders.extend(orders)

                # Step game
                game.step(all_orders)

                # Track checkpoint centers
                total_ckpt_centers = sum(
                    len(game.game.powers[p].centers)
                    for p in checkpoint_power_set
                    if p in game.game.powers
                )
                checkpoint_centers_history.append(total_ckpt_centers)

                # Capture visualization
                if vis:
                    phase = game.get_current_phase()
                    vis.capture_turn(game.game, f"Year {game.get_year()} - {phase}")

            # Game complete - collect results
            game_result = {
                "game_id": game_id,
                "opponent": opponent_type,
                "num_years": game.get_year() - 1901,
                "checkpoint_powers": {},
                "opponent_powers": {},
            }

            for power_name in all_powers:
                power = game.game.powers[power_name]
                power_data = {
                    "final_centers": len(power.centers),
                    "final_units": len(power.units),
                    "survived": len(power.centers) > 0 or len(power.units) > 0,
                    "won": len(power.centers) >= 18,
                }

                if power_name in checkpoint_power_set:
                    game_result["checkpoint_powers"][power_name] = power_data
                else:
                    game_result["opponent_powers"][power_name] = power_data

            opponent_results.append(game_result)

            # Save visualization
            if vis:
                vis_filename = f"eval_{checkpoint_name}_{opponent_type}_{game_idx}.html"
                vis_path = str(EVALS_PATH / vis_filename)
                vis.save_html(vis_path)
                all_visualizations.append(
                    {
                        "path": vis_path,
                        "game_id": game_id,
                        "opponent": opponent_type,
                    }
                )
                logger.info(f"  Saved visualization: {vis_path}")

            # Log progress
            ckpt_centers = sum(
                p["final_centers"] for p in game_result["checkpoint_powers"].values()
            )
            logger.info(
                f"  Game {game_idx + 1}/{games_per_opponent}: "
                f"{ckpt_centers} centers, {game_result['num_years']} years"
            )

        # Compute metrics for this opponent
        total_games = len(opponent_results)
        total_wins = sum(
            1 for g in opponent_results for p in g["checkpoint_powers"].values() if p["won"]
        )
        total_survivals = sum(
            1 for g in opponent_results for p in g["checkpoint_powers"].values() if p["survived"]
        )
        total_checkpoint_powers = sum(len(g["checkpoint_powers"]) for g in opponent_results)
        avg_centers = (
            sum(
                p["final_centers"]
                for g in opponent_results
                for p in g["checkpoint_powers"].values()
            )
            / total_checkpoint_powers
            if total_checkpoint_powers > 0
            else 0
        )

        opponent_metrics = {
            "opponent": opponent_type,
            "games": total_games,
            "win_rate": total_wins / total_checkpoint_powers if total_checkpoint_powers > 0 else 0,
            "survival_rate": total_survivals / total_checkpoint_powers
            if total_checkpoint_powers > 0
            else 0,
            "avg_centers": avg_centers,
            "results": opponent_results,
        }

        all_results.append(opponent_metrics)

        logger.info(f"\n  vs {opponent_type} Summary:")
        logger.info(f"    Win Rate: {opponent_metrics['win_rate']:.1%}")
        logger.info(f"    Survival Rate: {opponent_metrics['survival_rate']:.1%}")
        logger.info(f"    Avg Centers: {opponent_metrics['avg_centers']:.1f}")

        # Log to WandB
        if log_to_wandb:
            wandb.log(
                {
                    f"eval/vs_{opponent_type}/win_rate": opponent_metrics["win_rate"],
                    f"eval/vs_{opponent_type}/survival_rate": opponent_metrics["survival_rate"],
                    f"eval/vs_{opponent_type}/avg_centers": opponent_metrics["avg_centers"],
                }
            )

    total_duration = time.time() - start_time

    # Commit visualizations to volume
    volume.commit()

    # Log visualizations to WandB
    if log_to_wandb and all_visualizations:
        logger.info("\n📊 Logging visualizations to WandB...")

        # Create artifact for all visualizations
        artifact = wandb.Artifact(
            f"eval-replays-{checkpoint_name}",
            type="evaluation-replays",
            description=f"Game replays for {checkpoint_path}",
        )

        for viz in all_visualizations:
            if os.path.exists(viz["path"]):
                artifact.add_file(viz["path"])

                # Also log inline HTML for first game of each opponent
                first_per_opponent = {}
                for v in all_visualizations:
                    if v["opponent"] not in first_per_opponent:
                        first_per_opponent[v["opponent"]] = v

                if viz == first_per_opponent.get(viz["opponent"]):
                    with open(viz["path"], encoding="utf-8") as f:
                        html_content = f.read()
                    wandb.log({f"eval/replay_vs_{viz['opponent']}": wandb.Html(html_content)})

        wandb.log_artifact(artifact)

        # Create summary table
        table = wandb.Table(
            columns=[
                "opponent",
                "games",
                "win_rate",
                "survival_rate",
                "avg_centers",
            ]
        )
        for r in all_results:
            table.add_data(
                r["opponent"],
                r["games"],
                r["win_rate"],
                r["survival_rate"],
                r["avg_centers"],
            )
        wandb.log({"eval/summary_table": table})

    # Log completion to Axiom
    axiom.log(
        {
            "event": "evaluation_complete",
            "checkpoint_path": checkpoint_path,
            "total_duration_s": total_duration,
            "results": [
                {
                    "opponent": r["opponent"],
                    "win_rate": r["win_rate"],
                    "survival_rate": r["survival_rate"],
                    "avg_centers": r["avg_centers"],
                }
                for r in all_results
            ],
        }
    )

    # Finish WandB
    if log_to_wandb:
        wandb.finish()

    # Flush Axiom
    await axiom.flush()

    logger.info("\n" + "=" * 60)
    logger.info("✅ EVALUATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Duration: {total_duration:.1f}s")
    logger.info(f"Visualizations saved: {len(all_visualizations)}")

    return {
        "checkpoint_path": checkpoint_path,
        "timestamp": timestamp,
        "total_duration_s": total_duration,
        "results": [
            {
                "opponent": r["opponent"],
                "games": r["games"],
                "win_rate": r["win_rate"],
                "survival_rate": r["survival_rate"],
                "avg_centers": r["avg_centers"],
            }
            for r in all_results
        ],
        "visualization_paths": [v["path"] for v in all_visualizations],
    }
