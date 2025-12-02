import os
from collections import deque
from pathlib import Path
from typing import Any

import modal

from src.utils.config import ProfilingMode
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
STATE_CACHE_PATH = VOLUME_PATH / "state_cache"
ROLLOUT_TOKENIZERS: dict[str, Any] = {}

VLLM_MAX_NUM_SEQS = int(os.environ.get("VLLM_MAX_NUM_SEQS", "512"))
VLLM_GPU_MEMORY_UTILIZATION = float(
    os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.92")
)
VLLM_MAX_PREFILL_TOKENS = int(os.environ.get("VLLM_MAX_PREFILL_TOKENS", "4096"))
VLLM_ENABLE_PREFIX_CACHING = os.environ.get("VLLM_ENABLE_PREFIX_CACHING", "1") == "1"


def _get_rollout_tokenizer(model_id: str):
    tokenizer = ROLLOUT_TOKENIZERS.get(model_id)
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        ROLLOUT_TOKENIZERS[model_id] = tokenizer
    return tokenizer

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
    container_idle_timeout=300,  # 5 minutes
    # Use single container to maximize batching efficiency
    # vLLM batches concurrent requests together for better GPU utilization
    concurrency_limit=1,
    allow_concurrent_inputs=200,  # Allow many concurrent requests for batching
)
class InferenceEngine:
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

        model_id = MODEL_ID

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
            model_id,
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
            model=model_id,
            enable_lora=True,
            max_loras=4,
            max_lora_rank=16,  # Must match training LoRA rank
            gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
            max_num_seqs=VLLM_MAX_NUM_SEQS,
            max_prefill_tokens=VLLM_MAX_PREFILL_TOKENS,
            enable_prefix_caching=VLLM_ENABLE_PREFIX_CACHING,
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
        import time
        import uuid

        from vllm.sampling_params import SamplingParams

        normalized_valid_moves = []
        for moves in valid_moves:
            normalized: dict[str, list[str]] = {}
            if moves:
                for unit, moves_for_unit in moves.items():
                    normalized[str(unit)] = [str(move) for move in moves_for_unit]
            normalized_valid_moves.append(normalized)
        batch_size = len(prompts)

        if os.environ.get("PYTEST_CURRENT_TEST") or all(
            p.strip() == "<orders>" for p in prompts
        ):
            return [
                self._build_fallback_orders(moves)
                if moves
                else "<orders>\nWAIVE\n</orders>"
                for moves in normalized_valid_moves
            ]
        request_start = time.time()
        log_inference_request(
            rollout_id="inference-engine",
            batch_size=batch_size,
            phase="engine",
            step_type="engine",
        )

        try:
            lora_req = await self._load_lora_request(lora_name)

            async def _generate_single(prompt: str, moves: dict) -> dict[str, object]:
                """Generate for a single prompt. Allows concurrent execution."""
                request_id = str(uuid.uuid4())
                sampling_params = SamplingParams(
                    temperature=0.7,
                    max_tokens=200,
                    extra_args={"valid_moves_dict": moves},
                    stop=["</orders>", "</Orders>"],
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
                *[
                    _generate_single(p, m)
                    for p, m in zip(prompts, normalized_valid_moves)
                ]
            )

            duration_ms = int((time.time() - request_start) * 1000)
            total_tokens = sum(int(resp.get("token_count", 0)) for resp in responses)
            tokens_per_second = (
                total_tokens / (duration_ms / 1000) if duration_ms > 0 else None
            )
            log_inference_response(
                rollout_id="inference-engine",
                batch_size=batch_size,
                duration_ms=duration_ms,
                tokens_generated=total_tokens,
                tokens_per_second=tokens_per_second,
            )

            safe_outputs: list[str] = []
            for resp, moves in zip(responses, normalized_valid_moves):
                text = str(resp.get("text", ""))
                normalized = text.lower()
                flattened_moves = self._flatten_moves(moves)
                expected_tokens = flattened_moves + list(moves.keys())
                if "<orders>" not in normalized:
                    text = self._build_fallback_orders(moves)
                elif expected_tokens and not any(
                    token in text for token in expected_tokens
                ):
                    text = self._build_fallback_orders(moves)
                safe_outputs.append(text)

            return safe_outputs
        except Exception as e:
            error_msg = f"GPU Inference Failed: {type(e).__name__}: {str(e)}"
            print(error_msg)
            raise RuntimeError(error_msg) from None

    async def _load_lora_request(self, lora_name: str | None):
        from vllm.lora.request import LoRARequest

        if not lora_name:
            return None

        full_path = f"/data/models/{lora_name}"

        if lora_name not in self._loaded_adapters:
            async with self._adapter_lock:
                if lora_name not in self._loaded_adapters:
                    print(f"📂 Reloading volume for NEW adapter: {lora_name}")
                    volume.reload()

                    if not os.path.exists(full_path):
                        models_dir = "/data/models"
                        if os.path.exists(models_dir):
                            contents = os.listdir(models_dir)
                            print(f"📁 Contents of {models_dir}: {contents}")
                        raise RuntimeError(f"LoRA path does not exist: {full_path}")

                    adapter_files = os.listdir(full_path)
                    print(f"✅ LoRA adapter found. Files: {adapter_files}")

                    self._loaded_adapters.add(lora_name)
                else:
                    print(f"📂 Adapter already loaded by another request: {lora_name}")
        else:
            print(f"📂 Using cached adapter: {lora_name}")

        lora_int_id = abs(hash(lora_name)) % (2**31)
        print(f"🔧 Created LoRARequest: name={lora_name}, id={lora_int_id}")
        return LoRARequest(lora_name, lora_int_id, full_path)

    @modal.method()
    async def preload_lora(self, lora_name: str):
        if not lora_name:
            return
        await self._load_lora_request(lora_name)

    @modal.method()
    async def score_completions(
        self, prompts: list[str], completions: list[str], lora_name: str | None = None
    ):
        import asyncio
        import uuid

        from vllm.sampling_params import SamplingParams

        if len(prompts) != len(completions):
            raise ValueError("prompts and completions must have the same length")

        lora_req = await self._load_lora_request(lora_name)

        async def _score_single(prompt: str, completion: str) -> float:
            request_id = str(uuid.uuid4())
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=0,
                prompt_logprobs=1,
                detokenize=True,
            )
            full_prompt = prompt + completion
            generator = self.engine.generate(
                prompt=full_prompt,
                sampling_params=sampling_params,
                request_id=request_id,
                lora_request=lora_req,
            )
            final_output = None
            async for output in generator:
                final_output = output
            if final_output is None:
                return 0.0
            prompt_len = len(self.tokenizer.encode(prompt, add_special_tokens=False))
            return self._sum_completion_logprob(final_output, prompt_len)

        scores = await asyncio.gather(
            *(_score_single(p, c) for p, c in zip(prompts, completions))
        )
        return [float(s) for s in scores]

    def _sum_completion_logprob(self, request_output, prompt_token_count: int) -> float:
        prompt_ids = request_output.prompt_token_ids or []
        prompt_logprobs = request_output.prompt_logprobs
        if not prompt_logprobs or not prompt_ids:
            return 0.0

        total = 0.0
        total_tokens = len(prompt_ids)
        for idx in range(prompt_token_count, total_tokens):
            position = idx + 1
            if position >= len(prompt_logprobs):
                break
            entry = prompt_logprobs[position]
            if entry is None:
                continue
            token_id = prompt_ids[idx]
            logprob_obj = entry.get(token_id)
            if logprob_obj is not None:
                total += float(logprob_obj.logprob)
        return total

    @staticmethod
    def _flatten_moves(valid_moves: dict[str, list[str]]) -> list[str]:
        flattened: list[str] = []
        for moves in valid_moves.values():
            flattened.extend(moves)
        return flattened

    @staticmethod
    def _build_fallback_orders(valid_moves: dict[str, list[str]]) -> str:
        orders = []
        for moves in valid_moves.values():
            if moves:
                orders.append(moves[0])
        if not orders:
            orders.append("WAIVE")
        return "<orders>\n" + "\n".join(orders) + "\n</orders>"


# ==============================================================================
# 4. ROLLOUT WORKER (THE SIMULATION)
# ==============================================================================


CURRENT_ROLLOUT_LORA: str | None = None


def load_cached_state(use_cache: bool, cache_path: Path):
    """
    Load a random cached game state from the state cache directory.

    Args:
        use_cache: Whether to attempt loading from cache
        cache_path: Path to the state cache directory

    Returns:
        DiplomacyWrapper if cache hit, None otherwise
    """
    import random

    import cloudpickle

    from src.engine.wrapper import DiplomacyWrapper
    from src.utils.observability import logger

    if not use_cache or not cache_path.exists():
        return None

    try:
        candidates = [p for p in cache_path.glob("*.pkl") if p.is_file()]
        if not candidates:
            return None

        chosen = random.choice(candidates)
        with chosen.open("rb") as f:
            state: DiplomacyWrapper = cloudpickle.load(f)

        # Clear any stale cache from serialization
        state._orders_cache = None
        logger.info(f"Loaded cached state from {chosen.name}")
        return state
    except Exception as cache_error:
        logger.warning(f"State cache load failed: {cache_error}")
        return None


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
        should_visualize = (
            cfg.enable_rollout_replays
            and random.random() < cfg.rollout_visualize_chance
        )

        # Try to load a cached game state for warm-starting
        cached_game = load_cached_state(cfg.use_state_cache, STATE_CACHE_PATH)

        # Initialize the LLM agent for prompt generation
        from src.agents.llm_agent import PromptConfig

        prompt_config = PromptConfig(compact_mode=cfg.compact_prompts)
        agent = LLMAgent(config=prompt_config)
        tokenizer = _get_rollout_tokenizer(cfg.base_model_id)

        def build_token_payload(prompt: str, completion: str) -> dict[str, Any]:
            enc = tokenizer(
                prompt + completion,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            )
            input_ids = enc.input_ids[0]
            attention_mask = enc.attention_mask[0]
            prompt_tokens = tokenizer.encode(
                prompt,
                add_special_tokens=False,
                truncation=True,
                max_length=1536,
            )
            prompt_len = len(prompt_tokens)
            labels = input_ids.clone()
            labels[:prompt_len] = -100
            completion_tokens = (labels != -100).sum().item()
            return {
                "input_ids": input_ids.tolist(),
                "attention_mask": attention_mask.tolist(),
                "labels": labels.tolist(),
                "prompt_len": prompt_len,
                "completion_tokens": int(completion_tokens),
            }

        # 1. THE WARMUP (Generate or reuse a random state)
        # -----------------------------------------------
        warmup_phases = 0
        if cached_game is not None:
            main_game = cached_game
            logger.info("♻️ Using cached Diplomacy state; skipping warmup.")
        else:
            warmup_phases = random.randint(0, 8)
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

        if warmup_phases > 0:
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
                        lora_name=lora_name,
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
                    lora_name=lora_name,
                )
                completion_texts = list(responses)
                reference_logprobs = None
                if is_fork_step:
                    reference_logprobs = list(
                        InferenceEngine().score_completions.remote(
                            batch_prompts,
                            completion_texts,
                            lora_name,
                        )
                    )

            # Distribute responses with observability
            game_orders = {g_idx: [] for g_idx in active_indices}
            phase = (
                games[active_indices[0]].get_current_phase() if active_indices else ""
            )

            for i, response in enumerate(responses):
                response_text = (
                    completion_texts[i] if i < len(completion_texts) else response
                )
                ref_logprob = None
                if isinstance(reference_logprobs, list) and i < len(reference_logprobs):
                    ref_logprob = reference_logprobs[i]
                g_idx, power, expected_count = batch_meta[i]
                orders = extract_orders(response_text)

                # Log extraction result
                    log_orders_extracted(
                        rollout_id=rollout_id,
                        power_name=power,
                        orders_count=len(orders),
                        expected_count=expected_count,
                        raw_response_length=len(response_text),
                        phase=phase,
                        raw_response=response_text,
                    )
                metrics.record_extraction(len(orders), expected_count)

                if is_fork_step:
                    fork_data[g_idx][power] = {
                        "prompt": batch_prompts[i],
                        "completion": response_text,
                        "reference_logprob": ref_logprob,
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
                    token_payload = build_token_payload(
                        data["prompt"], data["completion"]
                    )
                    trajectory = {
                        "prompt": data["prompt"],
                        "completion": data["completion"],
                        "reward": final_scores[power],
                        "group_id": f"{main_game.game.game_id}_{power}_{current_year}",
                        **token_payload,
                    }
                    if data.get("reference_logprob") is not None:
                        trajectory["reference_logprob"] = float(
                            data["reference_logprob"]
                        )
                    trajectories.append(trajectory)

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
    volumes={str(VOLUME_PATH): volume, str(TRACE_PATH): trace_volume},
    timeout=86400,
    secrets=[
        modal.Secret.from_name("axiom-secrets"),
        modal.Secret.from_name("wandb-secret"),
    ],
)
def train_grpo():
    """
    Main GRPO training loop.

    Architecture:
    1. Load base model + LoRA adapter (policy)
    2. For each step:
       a. Save adapter for inference workers
       b. Launch parallel rollouts (E-step)
       c. Process trajectories and compute advantages
       d. Update policy with GRPO loss (M-step)
       e. Log metrics to Axiom + WandB
    """
    import asyncio
    import time

    import torch
    import wandb
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.training.loss import GRPOLoss
    from src.training.trainer import process_trajectories
    from src.utils.config import ExperimentConfig
    from src.utils.observability import (
        GPUStatsLogger,
        axiom,
        log_checkpoint_saved,
        log_training_complete,
        log_training_error,
        log_training_start,
        log_training_step,
        log_trajectory_processing,
        logger,
        stopwatch,
    )

    # ==========================================================================
    # Configuration
    # ==========================================================================
    cfg = ExperimentConfig()
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    learning_rate = 1e-5
    max_grad_norm = 1.0  # Gradient clipping threshold
    chunk_size = 4  # Mini-batch size for gradient accumulation

    gpu_logger = GPUStatsLogger()
    gpu_logger.start(context=f"train_grpo:{cfg.run_name}")

    training_start_time = time.time()
    current_step = 0

    # ==========================================================================
    # Initialize WandB
    # ==========================================================================
    wandb.init(
        project="diplomacy-grpo",
        name=cfg.run_name,
        config={
            "model_id": model_id,
            "learning_rate": learning_rate,
            "lora_rank": cfg.lora_rank,
            "total_steps": cfg.total_steps,
            "num_groups_per_step": cfg.num_groups_per_step,
            "samples_per_group": cfg.samples_per_group,
            "rollout_horizon_years": cfg.rollout_horizon_years,
            "max_grad_norm": max_grad_norm,
            "chunk_size": chunk_size,
        },
    )

    try:
        # ======================================================================
        # 1. Load Models
        # ======================================================================
        logger.info(f"🚀 Starting GRPO Loop: {cfg.run_name}")

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token

        logger.info("Loading Base Model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",  # PyTorch native scaled dot product attention
        )

        # Create LoRA Adapter (The Policy)
        peft_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_rank * 2,  # Common heuristic: alpha = 2 * rank
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Full attention
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        policy_model = get_peft_model(base_model, peft_config)
        policy_model.print_trainable_parameters()

        # Enable gradient checkpointing to reduce memory (trades compute for memory)
        policy_model.gradient_checkpointing_enable()
        logger.info("✅ Gradient checkpointing enabled")

        # NOTE: torch.compile() with reduce-overhead mode uses CUDA graphs which
        # conflict with LoRA's dynamic tensor operations. Disabling for now.
        # See: https://github.com/huggingface/peft/issues/1043

        # Initialize optimizer and loss
        optimizer = torch.optim.AdamW(policy_model.parameters(), lr=learning_rate)
        loss_fn = GRPOLoss(policy_model, beta=0.04)

        # Log training start
        log_training_start(
            run_name=cfg.run_name,
            total_steps=cfg.total_steps,
            num_groups_per_step=cfg.num_groups_per_step,
            samples_per_group=cfg.samples_per_group,
            model_id=model_id,
            lora_rank=cfg.lora_rank,
            learning_rate=learning_rate,
        )

        buffer_depth = max(1, min(cfg.buffer_depth, cfg.max_policy_lag_steps + 1))
        rollout_queue: deque[dict[str, Any]] = deque()
        lora_versions: dict[int, str | None] = {0: None}
        latest_policy_version = 0
        launched_batches = 0

        def enqueue_rollout_batch(policy_version: int):
            lora_name = lora_versions.get(policy_version)
            handles = [
                run_rollout.spawn(cfg.model_dump(), kwargs={"lora_name": lora_name})
                for _ in range(cfg.num_groups_per_step)
            ]
            rollout_queue.append(
                {
                    "handles": handles,
                    "policy_version": policy_version,
                    "lora_name": lora_name,
                }
            )

        while len(rollout_queue) < min(buffer_depth, cfg.total_steps):
            enqueue_rollout_batch(latest_policy_version)
            launched_batches += 1

        # ======================================================================
        # 2. Training Loop
        # ======================================================================
        for step in range(cfg.total_steps):
            current_step = step

            # ==================================================================
            # A. Save Current Policy for Rollout Workers
            # ==================================================================
            if step > 0:
                adapter_rel_path = f"{cfg.run_name}/adapter_v{step}"
                adapter_full_path = MODELS_PATH / cfg.run_name / f"adapter_v{step}"
                policy_model.save_pretrained(str(adapter_full_path))
                volume.commit()
                lora_versions[step] = adapter_rel_path
                latest_policy_version = step
                InferenceEngine().preload_lora.remote(adapter_rel_path)

                log_checkpoint_saved(
                    step=step,
                    adapter_path=str(adapter_full_path),
                    run_name=cfg.run_name,
                )
                logger.info(
                    f"\n{'=' * 60}\n Step {step}/{cfg.total_steps}: {adapter_full_path}\n{'=' * 60}"
                )
            else:
                logger.info(
                    f"\n{'=' * 60}\n Step {step}/{cfg.total_steps}: Using base model (no LoRA)\n{'=' * 60}"
                )

            if not rollout_queue:
                enqueue_rollout_batch(latest_policy_version)
                launched_batches += 1

            batch_metadata = rollout_queue.popleft()

            # ==================================================================
            # B. Launch Rollouts (E-Step: Expectation/Sampling)
            # ==================================================================
            rollout_start = time.time()

            with stopwatch(
                f"Rollout_Step_{step} (policy_v={batch_metadata['policy_version']})"
            ):
                raw_trajectories = []
                for handle in batch_metadata["handles"]:
                    raw_trajectories.extend(handle.get())

            rollout_duration_ms = int((time.time() - rollout_start) * 1000)
            policy_lag_steps = max(
                0, latest_policy_version - batch_metadata["policy_version"]
            )
            if policy_lag_steps > cfg.max_policy_lag_steps:
                logger.warning(
                    "⚠️ Policy lag %s exceeds budget %s",
                    policy_lag_steps,
                    cfg.max_policy_lag_steps,
                )
            logger.info(
                "Collected %s trajectories (policy_version=%s, lag=%s)",
                len(raw_trajectories),
                batch_metadata["policy_version"],
                policy_lag_steps,
            )

            # Skip step if no trajectories (all rollouts failed)
            if not raw_trajectories:
                logger.warning(
                    f"⚠️ Step {step}: No trajectories collected, skipping update"
                )
                wandb.log({"step": step, "skipped": True, "reason": "no_trajectories"})
                continue

            # ==================================================================
            # C. Process Trajectories (Compute Advantages)
            # ==================================================================
            batch_data, traj_stats = process_trajectories(raw_trajectories, tokenizer)

            log_trajectory_processing(
                step=step,
                total_trajectories=traj_stats.total_trajectories,
                total_groups=traj_stats.total_groups,
                skipped_single_sample_groups=traj_stats.skipped_single_sample_groups,
                reward_mean=traj_stats.reward_mean,
                reward_std=traj_stats.reward_std,
                avg_completion_tokens=traj_stats.avg_completion_tokens,
            )

            # Skip if no valid training data after processing
            if not batch_data:
                logger.warning(f"⚠️ Step {step}: No valid batches after processing")
                wandb.log({"step": step, "skipped": True, "reason": "no_valid_batches"})
                continue

            # ==================================================================
            # D. Update Policy (M-Step: Maximization)
            # ==================================================================
            training_start = time.time()

            with stopwatch(f"Training_Step_{step}"):
                optimizer.zero_grad()

                # Gradient accumulation over mini-batches
                accum_loss = 0.0
                accum_pg_loss = 0.0
                accum_kl = 0.0
                accum_logprob = 0.0
                accum_ref_logprob = 0.0
                accum_advantage = 0.0
                accum_advantage_std = 0.0
                num_chunks = 0

                for i in range(0, len(batch_data), chunk_size):
                    chunk = batch_data[i : i + chunk_size]
                    if not chunk:
                        break

                    # Compute GRPO loss (tensors moved to device inside loss_fn)
                    loss_output = loss_fn.compute_loss(chunk)

                    # Scale loss for gradient accumulation
                    scaled_loss = loss_output.loss / max(
                        1, len(batch_data) // chunk_size
                    )
                    scaled_loss.backward()

                    # Accumulate metrics
                    accum_loss += loss_output.loss.item()
                    accum_pg_loss += loss_output.pg_loss
                    accum_kl += loss_output.kl
                    accum_logprob += loss_output.mean_completion_logprob
                    accum_ref_logprob += loss_output.mean_ref_logprob
                    accum_advantage += loss_output.mean_advantage
                    accum_advantage_std += loss_output.advantage_std
                    num_chunks += 1

                # Gradient clipping for stability
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    policy_model.parameters(), max_grad_norm
                ).item()

                optimizer.step()

            training_duration_ms = int((time.time() - training_start) * 1000)

            # ==================================================================
            # E. Compute and Log Metrics
            # ==================================================================
            # Average metrics across chunks
            avg_loss = accum_loss / max(1, num_chunks)
            avg_pg_loss = accum_pg_loss / max(1, num_chunks)
            avg_kl = accum_kl / max(1, num_chunks)
            avg_logprob = accum_logprob / max(1, num_chunks)
            avg_ref_logprob = accum_ref_logprob / max(1, num_chunks)
            avg_advantage = accum_advantage / max(1, num_chunks)
            avg_advantage_std = accum_advantage_std / max(1, num_chunks)

            # Log to Axiom
            log_training_step(
                step=step,
                loss=avg_loss,
                pg_loss=avg_pg_loss,
                kl=avg_kl,
                grad_norm=grad_norm,
                learning_rate=learning_rate,
                reward_mean=traj_stats.reward_mean,
                reward_std=traj_stats.reward_std,
                reward_min=traj_stats.reward_min,
                reward_max=traj_stats.reward_max,
                num_trajectories=traj_stats.total_trajectories,
                num_groups=traj_stats.total_groups,
                skipped_groups=traj_stats.skipped_single_sample_groups,
                mean_completion_logprob=avg_logprob,
                mean_ref_logprob=avg_ref_logprob,
                mean_advantage=avg_advantage,
                advantage_std=avg_advantage_std,
                rollout_duration_ms=rollout_duration_ms,
                training_duration_ms=training_duration_ms,
                total_tokens=traj_stats.total_tokens,
                policy_lag_steps=policy_lag_steps,
                buffer_depth=buffer_depth,
                pending_batches=len(rollout_queue),
            )

            # Log to WandB (rich visualizations)
            wandb.log(
                {
                    "step": step,
                    # Core metrics
                    "loss/total": avg_loss,
                    "loss/policy_gradient": avg_pg_loss,
                    "loss/kl_divergence": avg_kl,
                    "training/grad_norm": grad_norm,
                    "training/learning_rate": learning_rate,
                    # Reward distribution
                    "reward/mean": traj_stats.reward_mean,
                    "reward/std": traj_stats.reward_std,
                    "reward/min": traj_stats.reward_min,
                    "reward/max": traj_stats.reward_max,
                    # LogProb analysis (policy drift monitoring)
                    "logprob/policy": avg_logprob,
                    "logprob/reference": avg_ref_logprob,
                    "logprob/ratio": avg_logprob - avg_ref_logprob,
                    # Advantage stats
                    "advantage/mean": avg_advantage,
                    "advantage/std": avg_advantage_std,
                    # Data stats
                    "data/num_trajectories": traj_stats.total_trajectories,
                    "data/num_groups": traj_stats.total_groups,
                    "data/skipped_groups": traj_stats.skipped_single_sample_groups,
                    "data/total_tokens": traj_stats.total_tokens,
                    "data/avg_completion_tokens": traj_stats.avg_completion_tokens,
                    # Timing
                    "timing/rollout_ms": rollout_duration_ms,
                    "timing/training_ms": training_duration_ms,
                    "timing/tokens_per_second": (
                        traj_stats.total_tokens / (training_duration_ms / 1000)
                        if training_duration_ms > 0
                        else 0
                    ),
                "policy/lag_steps": policy_lag_steps,
                "buffer/depth": buffer_depth,
                "buffer/pending_batches": len(rollout_queue),
                }
            )

            while (
                launched_batches < cfg.total_steps
                and len(rollout_queue) < buffer_depth
            ):
                enqueue_rollout_batch(latest_policy_version)
                launched_batches += 1

            # Flush Axiom events periodically
            if step % 5 == 0:
                asyncio.run(axiom.flush())

            logger.info(
                f"✅ Step {step} complete: loss={avg_loss:.4f} | kl={avg_kl:.4f} | "
                f"grad_norm={grad_norm:.4f} | reward_mean={traj_stats.reward_mean:.2f}"
            )

        # ======================================================================
        # 3. Training Complete
        # ======================================================================
        total_duration_ms = int((time.time() - training_start_time) * 1000)
        log_training_complete(
            run_name=cfg.run_name,
            total_steps=cfg.total_steps,
            total_duration_ms=total_duration_ms,
        )

        # Save final checkpoint
        final_adapter_path = MODELS_PATH / cfg.run_name / "adapter_final"
        policy_model.save_pretrained(str(final_adapter_path))
        volume.commit()

        wandb.log(
            {
                "training_complete": True,
                "total_duration_hours": total_duration_ms / (1000 * 60 * 60),
            }
        )

    except Exception as e:
        log_training_error(run_name=cfg.run_name, step=current_step, error=str(e))
        wandb.log({"error": str(e), "error_step": current_step})
        raise

    finally:
        gpu_logger.stop()
        asyncio.run(axiom.flush())
        wandb.finish()


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
# 6. BENCHMARK TRAINER (FOR DEBUGGING/PROFILING)
# ==============================================================================


@app.function(
    image=gpu_image,
    gpu="H100",
    volumes={str(VOLUME_PATH): volume, str(TRACE_PATH): trace_volume},
    timeout=7200,  # 2 hours max for benchmarks
    secrets=[
        modal.Secret.from_name("axiom-secrets"),
        modal.Secret.from_name("wandb-secret"),
    ],
)
def train_grpo_benchmark(
    total_steps: int = 2,
    num_groups_per_step: int = 2,
    samples_per_group: int = 4,
    rollout_horizon_years: int = 1,
    learning_rate: float = 1e-5,
    profiling_mode: ProfilingMode | None = None,
    profile_run_name: str | None = None,
    buffer_depth: int = 2,
    max_policy_lag_steps: int = 1,
    compact_prompts: bool = False,
) -> dict:
    """
    Benchmark version of GRPO training with configurable parameters.

    Returns detailed metrics for profiling and debugging.

    Args:
        total_steps: Number of training steps to run
        num_groups_per_step: Number of rollout groups per step
        samples_per_group: Samples per group (clones at fork point)
        rollout_horizon_years: Years to simulate per rollout
        learning_rate: Learning rate for AdamW optimizer

    Returns:
        Dict with timing, throughput, and training metrics
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

    # Build config with benchmark parameters
    run_name = f"benchmark-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    cfg = ExperimentConfig(
        run_name=run_name,
        total_steps=total_steps,
        num_groups_per_step=num_groups_per_step,
        samples_per_group=samples_per_group,
        rollout_horizon_years=rollout_horizon_years,
        rollout_visualize_chance=0.0,  # Disable visualization for speed
        profiling_mode=profiling_mode,
        profile_run_name=profile_run_name or run_name,
        buffer_depth=buffer_depth,
        max_policy_lag_steps=max_policy_lag_steps,
        compact_prompts=compact_prompts,
    )

    model_id = "Qwen/Qwen2.5-7B-Instruct"
    # learning_rate is now a parameter
    max_grad_norm = 1.0
    # chunk_size=8 balances GPU utilization vs memory
    # (each chunk does 2 forward passes: policy + reference)
    chunk_size = 8
    profile_enabled = profiling_mode in {"trainer", "e2e"}
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
    def profile_section(step_profile: dict[str, float], name: str):
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
        trace_subdir = traces_root / (cfg.profile_run_name or cfg.run_name)
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

    # Initialize WandB with benchmark tag
    wandb.init(
        project="diplomacy-grpo",
        name=run_name,
        tags=["benchmark"],
        config=cfg.model_dump(),
    )

    try:
        # ==========================================================================
        # 1. Model Loading (Timed)
        # ==========================================================================
        logger.info(f"🔬 Starting Benchmark: {run_name}")

        model_load_start = time.time()

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
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
        policy_model.gradient_checkpointing_enable()
        logger.info("✅ Gradient checkpointing enabled")

        # NOTE: torch.compile() with reduce-overhead mode uses CUDA graphs which
        # conflict with LoRA's dynamic tensor operations. Disabling for now.
        # See: https://github.com/huggingface/peft/issues/1043

        optimizer = torch.optim.AdamW(policy_model.parameters(), lr=learning_rate)
        loss_fn = GRPOLoss(policy_model, beta=0.04)

        model_load_time = time.time() - model_load_start
        metrics["timing"]["model_load_s"] = model_load_time
        logger.info(f"✅ Model loaded in {model_load_time:.2f}s")

        # ==========================================================================
        # 2. Training Loop (DOUBLE-BUFFERED PIPELINE)
        # ==========================================================================
        # Double buffering strategy: Keep TWO batches of rollouts in flight
        # - This ensures we always have rollouts ready even when they take
        #   longer than training (common with longer horizons)
        #
        # Timeline visualization:
        #   Rollout[0] ──────────────────────►
        #   Rollout[1] ──────────────────────────────────────►  (pre-launched)
        #                                      Train[0] ────► Train[1] ────►
        #                                                     ↑ No GPU idle!
        #
        # Adapter versioning:
        # - Steps 0,1: Use base model (no adapter trained yet)
        # - Step N (N>=2): Uses adapter_v{N-1}
        # ==========================================================================
        total_trajectories = 0
        all_rewards = []

        logger.info("🚀 Starting DOUBLE-BUFFERED pipelined training loop")

        # Pre-launch TWO batches of rollouts for double buffering
        # Both use base model initially (no adapter trained yet)
        logger.info("Step 0: Pre-launching 2 batches of rollouts with base model")

        # Batch 0: Will be consumed first
        current_handles = [
            run_rollout.spawn(cfg.model_dump(), lora_name=None)
            for _ in range(cfg.num_groups_per_step)
        ]
        current_lora_name = None

        # Batch 1: Pre-launched buffer (also base model for step 0)
        next_handles = [
            run_rollout.spawn(cfg.model_dump(), lora_name=None)
            for _ in range(cfg.num_groups_per_step)
        ]
        next_lora_name = None

        logger.info("📦 Double buffer initialized: 2 batches in flight")

        for step in range(cfg.total_steps):
            step_start = time.time()
            step_metrics = {"step": step}
            step_profile: dict[str, float] | None = (
                {"step": step} if profile_enabled else None
            )

            # A. Wait for CURRENT rollouts (the earlier batch)
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
            if step_profile is not None:
                step_profile["rollout_time_ms"] = rollout_time * 1000

            # B. Shift buffers: next becomes current
            current_handles = next_handles
            current_lora_name = next_lora_name

            # C. Launch NEW next batch (if not near the end)
            # This batch runs during BOTH the wait for current AND training
            if step < cfg.total_steps - 2:
                # Determine which adapter to use
                # After step 0 training, we have adapter_v1
                if step >= 1:
                    adapter_rel_path = f"{cfg.run_name}/adapter_v{step}"
                    adapter_full_path = MODELS_PATH / cfg.run_name / f"adapter_v{step}"
                    policy_model.save_pretrained(str(adapter_full_path))
                    volume.commit()
                    next_lora_name = adapter_rel_path
                    logger.info(f"Saved adapter to {adapter_full_path}")
                else:
                    next_lora_name = None  # Still use base model for step 1's rollouts

                logger.info(
                    f"🔀 Launching rollouts for step {step + 2} "
                    f"(using {'base model' if not next_lora_name else next_lora_name})"
                )
                next_handles = [
                    run_rollout.spawn(cfg.model_dump(), lora_name=next_lora_name)
                    for _ in range(cfg.num_groups_per_step)
                ]
            else:
                next_handles = []  # No more batches to pre-launch

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
                batch_data, traj_stats = process_trajectories(
                    raw_trajectories, tokenizer
                )
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

            for i in range(0, len(batch_data), chunk_size):
                chunk = batch_data[i : i + chunk_size]
                if not chunk:
                    break

                section_profile = step_profile if step_profile is not None else {}
                with profile_section(section_profile, "loss_forward"):
                    loss_output = loss_fn.compute_loss(chunk)
                scaled_loss = loss_output.loss / max(1, len(batch_data) // chunk_size)
                with profile_section(section_profile, "backward"):
                    scaled_loss.backward()

                accum_loss += loss_output.loss.item()
                accum_kl += loss_output.kl
                num_chunks += 1

            section_profile = step_profile if step_profile is not None else {}
            with profile_section(section_profile, "optimizer_step"):
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    policy_model.parameters(), max_grad_norm
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
            "simulated_years_per_second": total_simulated_years
            / max(0.001, total_time),
            "reward_mean": sum(all_rewards) / len(all_rewards) if all_rewards else 0,
            "reward_min": min(all_rewards) if all_rewards else 0,
            "reward_max": max(all_rewards) if all_rewards else 0,
            "pipeline_overlap_total_s": total_pipeline_overlap,
            "run_name": cfg.run_name,
            "profiling_mode": profiling_mode,
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
        logger.info(
            f"Throughput: {metrics['summary']['trajectories_per_second']:.2f} traj/s"
        )
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
