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
    scaledown_window=60 * 10,  # 10 minutes - longer to prevent churn
    buffer_containers=3,  # Increased from 1: keep more containers warm for higher throughput
)
@modal.concurrent(max_inputs=512, target_inputs=400)  # Increased from 256/200 for higher throughput
class InferenceEngine:
    model_id: str = modal.parameter(default="Qwen/Qwen2.5-7B-Instruct")
    is_for_league_evaluation: bool = modal.parameter(default=False)

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

        # Performance tuning parameters (optimized for throughput)
        # These values balance memory usage with throughput for A100 GPUs
        gpu_memory_util = (
            0.92  # Increased from 0.85: more aggressive memory use for higher throughput
        )
        max_num_seqs = 512  # Increased from 256: allow larger batches for better GPU utilization
        max_num_batched_tokens = 16384  # Token-level batching limit (critical for throughput)
        # Higher value = better throughput but more memory
        # 16384 = ~64 tokens per request at max batch size (good for Diplomacy prompts)

        engine_args = AsyncEngineArgs(
            model=self.model_id,
            enable_lora=True,
            max_loras=8,  # League training: up to 7 opponents + 1 hero
            max_lora_rank=16,  # Must match training LoRA rank
            gpu_memory_utilization=gpu_memory_util,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,  # Critical for throughput
            enable_prefix_caching=True,  # Prefix caching already optimized for shared prompt prefixes
            disable_log_stats=False,
            logits_processors=[DiplomacyLogitsProcessor],
            # Optimization: Use cached model location
            download_dir=str(HF_CACHE_PATH / "hub"),
            # Optimization: Prefer safetensors (faster loading than pickle)
            load_format="auto",  # Auto-detects best format (safetensors preferred)
            # Optional: Enable FP8 quantization for memory efficiency (uncomment if needed)
            # quantization="fp8",  # Reduces memory by ~2x, allows larger batches
            # Note: Requires model support and may have slight accuracy impact
        )

        print(
            f"⚙️  vLLM Config: max_num_seqs={max_num_seqs}, "
            f"max_num_batched_tokens={max_num_batched_tokens}, "
            f"gpu_memory_util={gpu_memory_util}"
        )
        self.engine = AsyncLLM.from_engine_args(engine_args)

        print("✅ Engine Ready (with dynamic LoRA support).")

    @modal.method()
    def warmup(self) -> dict:
        """
        Lightweight warmup method to pre-spin containers.
        Call this in parallel to force Modal to spin up multiple containers.
        Returns container info for logging.
        """
        import os
        import socket

        return {
            "status": "ready",
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "model_id": self.model_id,
        }

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

        # Detailed timing breakdown for tracing
        timing_breakdown = {
            "adapter_load_time_s": 0.0,
            "generation_time_s": 0.0,
            "total_time_s": 0.0,
        }

        log_inference_request(
            rollout_id="inference-engine",
            batch_size=batch_size,
            phase="engine",
            step_type="engine",
        )

        try:
            adapter_load_start = time.time()
            lora_req = None
            if lora_name:
                full_path = f"/data/models/{lora_name}"

                # Use lock to serialize volume reloads and prevent concurrent conflicts
                # Only reload if we haven't seen this adapter before
                if lora_name not in self._loaded_adapters:
                    async with self._adapter_lock:
                        # Double-check after acquiring lock (another request might have loaded it)
                        if lora_name not in self._loaded_adapters:
                            print(f"📂 Loading NEW adapter: {lora_name}")

                            # Wait for adapter to appear with retries
                            # The trainer saves adapters to the volume, which takes time to sync
                            max_retries = 5
                            for attempt in range(max_retries):
                                # Only reload volume if path doesn't exist yet
                                if not os.path.exists(full_path):
                                    try:
                                        volume.reload()
                                    except Exception as e:
                                        print(f"⚠️ Volume reload warning: {e}")

                                if os.path.exists(full_path):
                                    break

                                if attempt < max_retries - 1:
                                    wait_time = 1 + attempt  # 1, 2, 3, 4 seconds
                                    print(
                                        f"⏳ Adapter not found (attempt {attempt + 1}), waiting {wait_time}s..."
                                    )
                                    await asyncio.sleep(wait_time)

                            if not os.path.exists(full_path):
                                # List what's in models dir for debugging
                                models_dir = "/data/models"
                                if os.path.exists(models_dir):
                                    contents = os.listdir(models_dir)
                                    print(f"📁 Contents of {models_dir}: {contents}")
                                raise RuntimeError(
                                    f"LoRA path does not exist after retries: {full_path}"
                                )

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

            timing_breakdown["adapter_load_time_s"] = time.time() - adapter_load_start

            async def _generate_single(prompt: str, moves: dict) -> dict[str, object]:
                """Generate for a single prompt. Allows concurrent execution."""
                request_id = str(uuid.uuid4())
                # vLLM SamplingParams accepts these at runtime but type stubs may be incomplete
                # logprobs=1 returns per-token log probabilities for the sampled token
                sampling_params = SamplingParams(  # type: ignore[call-arg, misc]
                    temperature=0.7,  # type: ignore[arg-type]
                    max_tokens=200,  # type: ignore[arg-type]
                    extra_args={"valid_moves_dict": moves},  # type: ignore[arg-type]
                    stop=["</orders>", "</Orders>"],  # type: ignore[arg-type]
                    logprobs=1,  # type: ignore[arg-type]  # Return logprobs for ref model opt
                )
                try:
                    gen_start = time.time()
                    generator = self.engine.generate(
                        prompt=prompt,
                        sampling_params=sampling_params,
                        request_id=request_id,
                        lora_request=lora_req,
                    )
                    first_token_time = None
                    final_output = None
                    async for output in generator:
                        if first_token_time is None and output.outputs:
                            first_token_time = time.time() - gen_start
                        final_output = output

                    # generation_time tracked but not used per-request (tracked at batch level)

                    text = ""
                    token_ids: list[int] = []
                    prompt_token_ids: list[int] = []
                    completion_logprobs: list[float] = []

                    if final_output:
                        # Get prompt token IDs for computing prompt_len
                        prompt_token_ids = final_output.prompt_token_ids or []

                        if final_output.outputs:
                            output = final_output.outputs[0]
                            text = output.text
                            token_ids = list(output.token_ids)

                            # Extract per-token logprobs (for reference model optimization)
                            if output.logprobs:
                                # SampleLogprobs can be FlatLogprobs or list[dict[int, Logprob]]
                                logprobs_data = output.logprobs
                                if hasattr(logprobs_data, "logprobs"):
                                    # FlatLogprobs - extract directly
                                    # The sampled token's logprob is the first in each position
                                    for i in range(len(logprobs_data)):
                                        pos_logprobs = logprobs_data[i]
                                        if pos_logprobs and token_ids[i] in pos_logprobs:
                                            completion_logprobs.append(
                                                pos_logprobs[token_ids[i]].logprob
                                            )
                                        else:
                                            completion_logprobs.append(0.0)
                                else:
                                    # list[dict[int, Logprob]]
                                    for i, pos_logprobs in enumerate(logprobs_data):
                                        if pos_logprobs and token_ids[i] in pos_logprobs:
                                            completion_logprobs.append(
                                                pos_logprobs[token_ids[i]].logprob
                                            )
                                        else:
                                            completion_logprobs.append(0.0)

                    return {
                        "text": text,
                        "token_count": len(token_ids),
                        "token_ids": token_ids,
                        "prompt_token_ids": prompt_token_ids,
                        "completion_logprobs": completion_logprobs,
                    }
                except Exception as e:
                    # Log on GPU side for debugging
                    print(f"❌ Generation Error: {e}")
                    raise e

            # CRITICAL: Await all generators concurrently for proper batching
            # Old code awaited sequentially, killing vLLM's continuous batching
            generation_start = time.time()
            responses = await asyncio.gather(
                *[_generate_single(p, m) for p, m in zip(prompts, valid_moves, strict=True)]
            )
            timing_breakdown["generation_time_s"] = time.time() - generation_start

            duration_ms = int((time.time() - request_start) * 1000)
            timing_breakdown["total_time_s"] = duration_ms / 1000.0

            total_tokens = sum(
                int(token_count)
                if isinstance(token_count := resp.get("token_count"), int | str)
                else 0
                for resp in responses
            )
            tokens_per_second = total_tokens / (duration_ms / 1000) if duration_ms > 0 else None

            # Enhanced logging with timing breakdown
            log_inference_response(
                rollout_id="inference-engine",
                batch_size=batch_size,
                duration_ms=duration_ms,
                tokens_generated=total_tokens,
                tokens_per_second=tokens_per_second,
            )

            # Log detailed timing breakdown to Axiom for analysis
            from src.utils.observability import axiom

            axiom.log(
                {
                    "event": "inference_timing_breakdown",
                    "batch_size": batch_size,
                    "lora_name": lora_name,
                    "adapter_load_time_s": timing_breakdown["adapter_load_time_s"],
                    "generation_time_s": timing_breakdown["generation_time_s"],
                    "total_time_s": timing_breakdown["total_time_s"],
                    "tokens_generated": total_tokens,
                    "tokens_per_second": tokens_per_second,
                }
            )

            # Debug logging for slow batches
            # Single-item batches are expected to be slower (2-3s is normal for vLLM)
            # Larger batches should be faster per-item, so we use adaptive thresholds
            if batch_size == 1:
                # Single-item batches: warn if > 5s (very slow)
                slow_threshold = 5.0
            elif batch_size <= 4:
                # Small batches: warn if > 3s per item
                slow_threshold = 3.0 * batch_size
            else:
                # Larger batches: warn if > 2s per item
                slow_threshold = 2.0 * batch_size

            if timing_breakdown["total_time_s"] > slow_threshold:
                print(
                    f"⏱️  Slow inference batch: {timing_breakdown['total_time_s']:.3f}s "
                    f"(adapter_load: {timing_breakdown['adapter_load_time_s']:.3f}s, "
                    f"generation: {timing_breakdown['generation_time_s']:.3f}s, "
                    f"batch_size={batch_size}, threshold={slow_threshold:.1f}s)"
                )

            # Return rich response structure with token data for trainer optimization
            return [
                {
                    "text": str(resp.get("text", "")),
                    "token_ids": resp.get("token_ids", []),
                    "prompt_token_ids": resp.get("prompt_token_ids", []),
                    "completion_logprobs": resp.get("completion_logprobs", []),
                }
                for resp in responses
            ]
        except Exception as e:
            error_msg = f"GPU Inference Failed: {type(e).__name__}: {str(e)}"
            print(error_msg)

            # Detect fatal vLLM errors that leave the engine in an unrecoverable state
            # Force container exit so Modal spins up a fresh one
            error_str = str(e).lower()
            error_type = type(e).__name__
            is_fatal = (
                "enginedeaderror" in error_type.lower()
                or "enginedead" in error_str
                or "engine" in error_str
                and "dead" in error_str
                or "cuda" in error_str
                and "out of memory" in error_str
                or "cudnn" in error_str
                and "error" in error_str
            )

            if is_fatal:
                import os

                print("💀 FATAL: vLLM engine is dead. Killing container for fresh restart...")
                os._exit(1)  # Force immediate exit, bypasses finally blocks

            raise RuntimeError(error_msg) from None

    @modal.method()
    async def score(
        self,
        prompts: list[str],
        completions: list[str],
        prompt_token_ids_list: list[list[int]],
    ) -> list[dict]:
        """
        Compute reference logprobs for completions using BASE MODEL (no LoRA).

        This enables skipping the reference forward pass in the trainer by capturing
        base model logprobs during rollouts.

        Uses vLLM's prompt_logprobs feature: pass full sequence as "prompt" with
        max_tokens=0 to score without generation.

        Args:
            prompts: List of prompt strings (for prefix caching benefit)
            completions: List of completion strings to score
            prompt_token_ids_list: List of prompt token IDs (to know where completion starts)

        Returns:
            List of dicts with 'ref_completion_logprobs' (sum of completion token logprobs)
        """
        import asyncio
        import time
        import uuid

        from vllm.sampling_params import SamplingParams

        request_start = time.time()

        async def _score_single(prompt: str, completion: str, prompt_token_ids: list[int]) -> dict:
            """Score a single sequence using base model."""
            request_id = str(uuid.uuid4())

            # Scoring params: generate 1 token (vLLM requires >= 1) but we only care about prompt_logprobs
            sampling_params = SamplingParams(  # type: ignore[call-arg, misc]
                max_tokens=1,  # type: ignore[arg-type]  # Min 1 required by vLLM, we ignore the output
                prompt_logprobs=1,  # type: ignore[arg-type]  # Get logprobs for all input tokens
            )

            # Full sequence = prompt + completion
            full_sequence = prompt + completion

            try:
                # Score with base model (no LoRA) to get reference logprobs
                generator = self.engine.generate(
                    prompt=full_sequence,
                    sampling_params=sampling_params,
                    request_id=request_id,
                    lora_request=None,  # CRITICAL: Use base model for reference
                )
                final_output = None
                async for output in generator:
                    final_output = output

                # Extract completion logprobs from prompt_logprobs
                completion_logprobs_sum = 0.0
                if final_output and final_output.prompt_logprobs:
                    prompt_len = len(prompt_token_ids)
                    # prompt_logprobs is list[dict[token_id, Logprob] | None]
                    # Index 0 is None (no logprob for first token)
                    # Completion tokens start at index prompt_len
                    for i in range(prompt_len, len(final_output.prompt_logprobs)):
                        pos_logprobs = final_output.prompt_logprobs[i]
                        if pos_logprobs:
                            # Get the logprob of the actual token at this position
                            # The token_id is the key with highest logprob (or we can get from prompt_token_ids)
                            # vLLM returns logprobs for top-k tokens, including the actual one
                            actual_token_id = (
                                final_output.prompt_token_ids[i]
                                if final_output.prompt_token_ids
                                else None
                            )
                            if actual_token_id is not None and actual_token_id in pos_logprobs:
                                completion_logprobs_sum += pos_logprobs[actual_token_id].logprob

                return {"ref_completion_logprobs": completion_logprobs_sum}

            except Exception as e:
                print(f"❌ Scoring Error: {e}")

                # Detect fatal vLLM errors that leave the engine unrecoverable
                error_str = str(e).lower()
                error_type = type(e).__name__
                is_fatal = (
                    "enginedeaderror" in error_type.lower()
                    or "enginedead" in error_str
                    or "engine" in error_str
                    and "dead" in error_str
                    or "cuda" in error_str
                    and "out of memory" in error_str
                )

                if is_fatal:
                    import os

                    print("💀 FATAL: vLLM engine is dead. Killing container for fresh restart...")
                    os._exit(1)

                # Return empty logprob on error - trainer will fall back to computing
                return {"ref_completion_logprobs": None}

        # Score all sequences concurrently
        results = await asyncio.gather(
            *[
                _score_single(p, c, pt)
                for p, c, pt in zip(prompts, completions, prompt_token_ids_list, strict=True)
            ]
        )

        duration_ms = int((time.time() - request_start) * 1000)
        print(f"📊 Scored {len(prompts)} sequences in {duration_ms}ms (base model)")

        return results


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
async def run_rollout(
    config_dict: dict,
    lora_name: str | None = None,
    *,
    power_adapters: dict[str, str | None] | None = None,
    hero_power: str | None = None,
):
    """
    Run a single rollout (game simulation) with the given config.

    Args:
        config_dict: ExperimentConfig as dict
        lora_name: (DEPRECATED) Name of the LoRA adapter for ALL powers.
                   Use power_adapters instead for league training.
        power_adapters: Mapping of power name to adapter. Values can be:
                       - "random_bot" or "chaos_bot": Use baseline bot (no LLM)
                       - None or "base_model": Use base LLM (no LoRA)
                       - str: LoRA adapter path (e.g., "run/adapter_v50")
                       If None, falls back to using lora_name for all powers.
        hero_power: Which power's trajectories to collect for training.
                   If None, collects trajectories for ALL powers (original behavior).
    """
    import os
    import random
    import time

    import cloudpickle

    from src.agents import LLMAgent, PromptConfig
    from src.agents.baselines import ChaosBot, RandomBot
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

    # Baseline bot instances (stateless, can be shared)
    BASELINE_BOTS = {
        "random_bot": RandomBot(),
        "chaos_bot": ChaosBot(),
    }

    # Build power_adapters from legacy lora_name if not provided
    POWERS = ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]
    if power_adapters is None:
        # Legacy mode: same adapter for all powers
        power_adapters = dict.fromkeys(POWERS, lora_name)

    # NOTE: We use vLLM's lora_filesystem_resolver to load adapters dynamically
    # The resolver looks in VLLM_LORA_RESOLVER_CACHE_DIR (/data/models) for the adapter
    # We just need to reload the volume to see newly committed adapter files
    volume_reload_time = 0.0

    # Collect unique LoRA adapters that need to be loaded
    unique_loras = {
        adapter
        for adapter in power_adapters.values()
        if adapter is not None and adapter not in BASELINE_BOTS and adapter != "base_model"
    }

    # Reload volume if there are any LoRAs to load
    if unique_loras:
        logger.info(f"📂 Using LoRA adapters: {unique_loras}")
        global CURRENT_ROLLOUT_LORA

        # Check if we need to reload (any new adapter not seen before)
        needs_reload = any(
            adapter != CURRENT_ROLLOUT_LORA and not os.path.exists(f"/data/models/{adapter}")
            for adapter in unique_loras
        )

        if needs_reload:
            reload_start = time.time()
            volume.reload()  # Ensure we see the latest adapter files
            volume_reload_time = time.time() - reload_start
            logger.info(f"⏱️ Volume reload took {volume_reload_time:.2f}s")

        # Verify all adapters exist
        for adapter in unique_loras:
            full_path = f"/data/models/{adapter}"
            if os.path.exists(full_path):
                files = os.listdir(full_path)
                logger.info(f"✅ LoRA adapter found at {full_path}. Files: {files}")
            else:
                logger.error(f"❌ LoRA adapter NOT found at: {full_path}")

        # Track the "main" adapter (for backwards compat in reference logprob logic)
        if unique_loras:
            CURRENT_ROLLOUT_LORA = next(iter(unique_loras))

    rollout_start_time = time.time()
    rollout_id = ""
    metrics: RolloutMetrics | None = None

    try:
        cfg = ExperimentConfig(**config_dict)
        should_visualize = random.random() < cfg.rollout_visualize_chance

        # Initialize the LLM agent for prompt generation
        prompt_config = PromptConfig(
            compact_mode=cfg.compact_prompts, prefix_cache_optimized=cfg.prefix_cache_optimized
        )
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
        # NOTE: Warmup uses base model for LLM powers (not per-power adapters)
        # This is fine since warmup is just for generating diverse game states
        for i in range(warmup_phases):
            if main_game.is_done():
                break

            all_orders = []
            phase = main_game.get_current_phase()

            # Get inputs for all powers
            inputs = main_game.get_all_inputs(agent=agent)

            # Separate baseline bots from LLM powers
            baseline_indices = []
            llm_indices = []
            for idx, power in enumerate(inputs["power_names"]):
                adapter = power_adapters.get(power)
                if adapter in BASELINE_BOTS:
                    baseline_indices.append(idx)
                else:
                    llm_indices.append(idx)

            # 1. Handle baseline bots directly (no LLM)
            for idx in baseline_indices:
                power = inputs["power_names"][idx]
                adapter = power_adapters.get(power)
                assert adapter is not None and adapter in BASELINE_BOTS
                bot = BASELINE_BOTS[adapter]
                orders = bot.get_orders(main_game, power)
                expected_count = len(inputs["valid_moves"][idx])

                log_orders_extracted(
                    rollout_id=rollout_id,
                    power_name=power,
                    orders_count=len(orders),
                    expected_count=expected_count,
                    raw_response_length=0,
                    phase=phase,
                    raw_response="[BASELINE BOT]",
                )
                metrics.record_extraction(len(orders), expected_count)
                all_orders.extend(orders)

            # 2. Handle LLM powers with batched inference
            if llm_indices:
                llm_prompts = [inputs["prompts"][idx] for idx in llm_indices]
                llm_valid_moves = [inputs["valid_moves"][idx] for idx in llm_indices]

                with stopwatch(f"Warmup Inference {i + 1}/{warmup_phases}"):
                    metrics.inference_calls += 1
                    raw_responses = await InferenceEngine(
                        model_id=cfg.base_model_id
                    ).generate.remote.aio(
                        prompts=llm_prompts,
                        valid_moves=llm_valid_moves,
                        lora_name=None,  # Use base model for warmup
                    )

                for resp_idx, game_idx in enumerate(llm_indices):
                    response_data = raw_responses[resp_idx]
                    power = inputs["power_names"][game_idx]
                    response_text = response_data["text"]
                    orders = extract_orders(response_text)
                    expected_count = len(inputs["valid_moves"][game_idx])

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
                phase = game.get_current_phase()

                # Results will be populated as we process each power
                # Format: {power_idx: {"orders": [...], "response_data": {...}}}
                power_results: dict[int, dict] = {}

                # 1. Handle baseline bots (no LLM inference needed)
                baseline_count = 0
                llm_count = 0
                for idx, power in enumerate(inputs["power_names"]):
                    adapter = power_adapters.get(power)
                    if adapter in BASELINE_BOTS:
                        baseline_count += 1
                        bot = BASELINE_BOTS[adapter]
                        orders = bot.get_orders(game, power)
                        power_results[idx] = {
                            "orders": orders,
                            "response_data": {
                                "text": "[BASELINE BOT]",
                                "prompt_token_ids": [],
                                "token_ids": [],
                                "completion_logprobs": [],
                            },
                        }
                    else:
                        llm_count += 1

                # 2. Group LLM powers by adapter for batched inference
                # Map adapter -> list of (original_idx, prompt, valid_moves)
                adapter_groups: dict[str | None, list[tuple[int, str, dict]]] = {}
                for idx, power in enumerate(inputs["power_names"]):
                    if idx in power_results:
                        continue  # Already handled by baseline bot

                    adapter = power_adapters.get(power)
                    # Normalize adapter: None and "base_model" both mean no LoRA
                    adapter_key = None if adapter in (None, "base_model") else adapter

                    if adapter_key not in adapter_groups:
                        adapter_groups[adapter_key] = []
                    adapter_groups[adapter_key].append(
                        (idx, inputs["prompts"][idx], inputs["valid_moves"][idx])
                    )

                # 3. Run batched inference for each adapter group
                # Note: batch_size=1 is common when:
                # - Only one power uses a particular adapter (league training)
                # - Powers are eliminated early in the game
                # - Using baseline bots for some powers
                # vLLM's continuous batching will still batch these across concurrent games
                for adapter_key, group_items in adapter_groups.items():
                    group_indices = [item[0] for item in group_items]
                    group_prompts = [item[1] for item in group_items]
                    group_valid_moves = [item[2] for item in group_items]
                    batch_size = len(group_prompts)

                    with stopwatch(
                        f"Game {g_idx} Step {step_count} Adapter={adapter_key} (Batch: {batch_size})"
                    ):
                        metrics.inference_calls += 1
                        responses = await InferenceEngine(
                            model_id=cfg.base_model_id
                        ).generate.remote.aio(
                            prompts=group_prompts,
                            valid_moves=group_valid_moves,
                            lora_name=adapter_key,
                        )

                    # Map responses back to original indices
                    for resp_idx, orig_idx in enumerate(group_indices):
                        response_data = responses[resp_idx]
                        response_text = response_data["text"]
                        orders = extract_orders(response_text)
                        power_results[orig_idx] = {
                            "orders": orders,
                            "response_data": response_data,
                        }

                # 4. Collect all orders and log metrics
                all_orders = []
                for idx, power in enumerate(inputs["power_names"]):
                    result = power_results[idx]
                    orders = result["orders"]
                    response_data = result["response_data"]
                    response_text = response_data["text"]
                    expected_count = len(inputs["valid_moves"][idx])

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

                    # Store fork data on first step for hero power only
                    # (or all powers if hero_power is None for backwards compat)
                    should_collect = (hero_power is None) or (power == hero_power)
                    is_baseline = power_adapters.get(power) in BASELINE_BOTS

                    if step_count == 1 and should_collect and not is_baseline:
                        game_fork_data[power] = {
                            "prompt": inputs["prompts"][idx],
                            "completion": response_text,
                            "prompt_token_ids": response_data.get("prompt_token_ids", []),
                            "completion_token_ids": response_data.get("token_ids", []),
                            "completion_logprobs": response_data.get("completion_logprobs", []),
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

        # 4. REFERENCE LOGPROBS (Optional - eliminates trainer reference forward pass)
        # ---------------------------------------------------------------------------
        # If enabled, compute base model logprobs for all completions
        # This adds ~1 forward pass to rollouts but saves ~1 forward pass in trainer
        ref_logprobs_map: dict[tuple[int, str], float | None] = {}

        # Check if any hero power uses a LoRA adapter (need ref logprobs)
        hero_uses_lora = False
        if hero_power:
            hero_adapter = power_adapters.get(hero_power)
            hero_uses_lora = (
                hero_adapter not in (None, "base_model") and hero_adapter not in BASELINE_BOTS
            )
        else:
            # Legacy mode: check if any unique LoRA is used
            hero_uses_lora = bool(unique_loras)

        if cfg.compute_ref_logprobs_in_rollout and hero_uses_lora:
            # Only needed when generating with LoRA (step > 0)
            # For step 0 (no LoRA), generation logprobs ARE reference logprobs
            with stopwatch("Reference Logprob Scoring"):
                # Batch all sequences for scoring
                score_prompts = []
                score_completions = []
                score_prompt_token_ids = []
                score_keys = []  # (g_idx, power) to map results back

                for g_idx in range(len(games)):
                    for power, data in fork_data[g_idx].items():
                        prompt_tids = data.get("prompt_token_ids", [])
                        if prompt_tids:  # Only if we have token data
                            score_prompts.append(data["prompt"])
                            score_completions.append(data["completion"])
                            score_prompt_token_ids.append(prompt_tids)
                            score_keys.append((g_idx, power))

                if score_prompts:
                    # Call scoring endpoint (uses base model, no LoRA)
                    score_results = await InferenceEngine(
                        model_id=cfg.base_model_id
                    ).score.remote.aio(
                        prompts=score_prompts,
                        completions=score_completions,
                        prompt_token_ids_list=score_prompt_token_ids,
                    )

                    # Map results back
                    for key, result in zip(score_keys, score_results, strict=True):
                        ref_logprobs_map[key] = result.get("ref_completion_logprobs")

                    logger.info(
                        f"📊 Computed {len(score_results)} reference logprobs (using base model)"
                    )

        # 5. SCORING & RETURN
        # -------------------
        trajectories = []

        for g_idx, game in enumerate(games):
            # Use win_bonus from config for scoring
            final_scores = calculate_final_scores(
                game,
                win_bonus=cfg.win_bonus,
                winner_threshold_sc=cfg.winner_threshold_sc,
            )

            for power, data in fork_data[g_idx].items():
                if power in final_scores:
                    # Get reference logprobs if computed
                    ref_logprobs = ref_logprobs_map.get((g_idx, power))

                    # If no LoRA used (step 0), generation logprobs ARE reference logprobs
                    power_adapter = power_adapters.get(power)
                    power_uses_lora = (
                        power_adapter not in (None, "base_model")
                        and power_adapter not in BASELINE_BOTS
                    )
                    if ref_logprobs is None and not power_uses_lora:
                        completion_logprobs = data.get("completion_logprobs", [])
                        if completion_logprobs:
                            ref_logprobs = sum(completion_logprobs)

                    traj = {
                        "prompt": data["prompt"],
                        "completion": data["completion"],
                        "reward": final_scores[power],
                        "group_id": f"{main_game.game.game_id}_{power}_{current_year}",
                        # Token data for trainer optimization (skip tokenization)
                        "prompt_token_ids": data.get("prompt_token_ids", []),
                        "completion_token_ids": data.get("completion_token_ids", []),
                        "completion_logprobs": data.get("completion_logprobs", []),
                    }

                    # Add reference logprobs if available (enables skipping trainer ref forward)
                    if ref_logprobs is not None:
                        traj["ref_logprobs"] = ref_logprobs

                    trajectories.append(traj)

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

        # Return trajectories with extraction stats for WandB aggregation
        return {
            "trajectories": trajectories,
            "extraction_stats": {
                "orders_expected": metrics.total_orders_expected,
                "orders_extracted": metrics.total_orders_extracted,
                "empty_responses": metrics.empty_responses,
                "partial_responses": metrics.partial_responses,
                "extraction_rate": metrics.get_extraction_rate(),
            },
            "timing": {
                "volume_reload_s": volume_reload_time,
                "total_s": time.time() - rollout_start_time,
            },
        }

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
    retries=0,  # CRITICAL: Don't auto-retry training - would restart from scratch
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

    try:
        # ==========================================================================
        # 0. Pre-warm Inference Containers via Autoscaler
        # ==========================================================================
        # vLLM cold start is slow (~30-60s). The InferenceEngine class is configured
        # with buffer_containers=1 at the decorator level, which keeps at least 1
        # container warm automatically. We spawn warmup calls to ensure containers
        # are actually ready.

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

        # ==========================================================================
        # Training State Checkpointing Helpers
        # ==========================================================================
        def save_training_state(step: int) -> None:
            """Save full training state for resume capability with atomic writes."""
            import tempfile

            run_path = MODELS_PATH / cfg.run_name
            run_path.mkdir(parents=True, exist_ok=True)
            state_path = run_path / f"training_state_v{step}.pt"

            state = {
                "step": step,
                "optimizer_state_dict": optimizer.state_dict(),
                "config": cfg.model_dump(),
                "seed": cfg.seed,  # Save seed for reproducibility
            }
            # Save wandb run ID if wandb is initialized (for resume)
            if wandb.run is not None:
                state["wandb_run_id"] = wandb.run.id

            # Atomic write: write to temp file first, then rename
            # This prevents corruption if save is interrupted
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=str(run_path), delete=False, suffix=".tmp"
            ) as tmp_file:
                tmp_path = tmp_file.name
                torch.save(state, tmp_path)

            # Atomic rename (should work on most filesystems)
            import os

            os.rename(tmp_path, str(state_path))
            volume.commit()
            logger.info(f"💾 Saved training state to {state_path}")

        def load_training_state(
            run_name: str, step: int | None = None, allow_fallback: bool = True
        ) -> tuple[int, str | None]:
            """
            Load training state and return the step to resume from and wandb run ID (if available).

            Args:
                run_name: Name of the run to load from
                step: Specific step to load (None = latest)
                allow_fallback: If True, try earlier checkpoints if latest is corrupted

            Returns:
                Tuple of (step, wandb_run_id)
            """
            import glob

            run_path = MODELS_PATH / run_name

            # Find latest checkpoint if step not specified
            if step is None:
                pattern = str(run_path / "training_state_v*.pt")
                state_files = glob.glob(pattern)
                if not state_files:
                    raise FileNotFoundError(f"No training states found in {run_path}")
                # Extract step numbers and find max
                steps = []
                for f in state_files:
                    try:
                        s = int(f.split("_v")[-1].replace(".pt", ""))
                        steps.append(s)
                    except ValueError:
                        pass
                if not steps:
                    raise FileNotFoundError(f"No valid training states found in {run_path}")
                step = max(steps)

            # At this point step is guaranteed to be an int (either passed in or from max(steps))
            assert step is not None, "step should be set by now"

            # Try loading checkpoint, with fallback to earlier ones if corrupted
            attempts = [step]
            if allow_fallback:
                # Get all available steps sorted descending
                pattern = str(run_path / "training_state_v*.pt")
                all_files = glob.glob(pattern)
                all_steps = sorted(
                    [
                        int(f.split("_v")[-1].replace(".pt", ""))
                        for f in all_files
                        if f.split("_v")[-1].replace(".pt", "").isdigit()
                    ],
                    reverse=True,
                )
                # Add earlier steps as fallbacks (up to 3 attempts)
                attempts.extend([s for s in all_steps if s < step][:2])

            last_error = None
            for attempt_step in attempts:
                state_path = run_path / f"training_state_v{attempt_step}.pt"
                if not state_path.exists():
                    continue

                try:
                    # Load and validate checkpoint
                    state = torch.load(str(state_path), weights_only=False)

                    # Validate required keys exist
                    required_keys = ["step", "optimizer_state_dict", "config"]
                    missing_keys = [k for k in required_keys if k not in state]
                    if missing_keys:
                        raise ValueError(f"Checkpoint missing required keys: {missing_keys}")

                    # Verify step matches filename
                    if state["step"] != attempt_step:
                        logger.warning(
                            f"⚠️ Step mismatch in checkpoint: filename={attempt_step}, "
                            f"state={state['step']}"
                        )

                    # Load optimizer state
                    optimizer.load_state_dict(state["optimizer_state_dict"])

                    # Load the adapter for this step
                    adapter_path = run_path / f"adapter_v{attempt_step}"
                    if adapter_path.exists():
                        policy_model.load_adapter(str(adapter_path), adapter_name="default")
                        logger.info(f"📂 Loaded adapter from {adapter_path}")
                    else:
                        logger.warning(
                            f"⚠️ Adapter not found at {adapter_path}, continuing with current weights"
                        )

                    # Extract wandb run ID if available (for resume)
                    wandb_run_id = state.get("wandb_run_id")

                    # Restore random seed if saved
                    saved_seed = state.get("seed")
                    if saved_seed is not None:
                        import random

                        import numpy as np

                        random.seed(saved_seed)
                        np.random.seed(saved_seed)
                        torch.manual_seed(saved_seed)
                        logger.info(f"🌱 Restored random seed: {saved_seed}")

                    # Warn about config mismatches (non-critical)
                    saved_config = state.get("config", {})
                    current_config = cfg.model_dump()
                    config_diff = {
                        k: (saved_config.get(k), current_config.get(k))
                        for k in set(saved_config.keys()) | set(current_config.keys())
                        if saved_config.get(k) != current_config.get(k)
                        and k not in ["run_name"]  # run_name can differ when forking
                    }
                    if config_diff:
                        logger.warning(
                            f"⚠️ Config differences detected (non-critical): {list(config_diff.keys())}"
                        )

                    if attempt_step != step:
                        logger.warning(
                            f"⚠️ Loaded checkpoint from step {attempt_step} "
                            f"(requested {step} was corrupted/unavailable)"
                        )

                    logger.info(f"✅ Resumed from step {attempt_step} (run: {run_name})")
                    return attempt_step, wandb_run_id

                except Exception as e:
                    last_error = e
                    logger.warning(f"⚠️ Failed to load checkpoint at step {attempt_step}: {e}")
                    if attempt_step == step:
                        logger.warning("   Will try earlier checkpoints if available...")
                    continue

            # All attempts failed
            raise FileNotFoundError(
                f"Failed to load any valid checkpoint from {run_path}. Last error: {last_error}"
            )

        # Resume from checkpoint if specified OR if checkpoints exist for this run
        start_step = 0
        wandb_run_id: str | None = None
        volume.reload()  # Ensure we see latest files

        if cfg.resume_from_run:
            # Explicit resume from another run
            try:
                start_step, wandb_run_id = load_training_state(
                    cfg.resume_from_run, cfg.resume_from_step
                )
                if cfg.resume_from_run != cfg.run_name:
                    logger.info(f"📦 Forking from {cfg.resume_from_run} to new run {cfg.run_name}")
                    # When forking, don't reuse the old wandb run ID
                    wandb_run_id = None
            except FileNotFoundError as e:
                logger.error(f"❌ Resume failed: {e}")
                raise
        elif not cfg.disable_auto_resume:
            # Auto-resume: Check if this run has existing checkpoints (crash recovery)
            import glob

            run_path = MODELS_PATH / cfg.run_name
            pattern = str(run_path / "training_state_v*.pt")
            existing_states = glob.glob(pattern)

            if existing_states:
                # Found existing checkpoints - auto-resume from latest
                logger.warning(
                    f"⚠️ Found {len(existing_states)} existing checkpoints for {cfg.run_name}"
                )
                logger.warning(
                    "🔄 AUTO-RESUMING from crash (use --disable-auto-resume to start fresh)"
                )
                try:
                    start_step, wandb_run_id = load_training_state(
                        cfg.run_name, cfg.resume_from_step
                    )
                except FileNotFoundError as e:
                    logger.error(f"❌ Auto-resume failed: {e}, starting fresh")
                    start_step = 0
                    wandb_run_id = None

        # Initialize WandB (after training state loading to support resume)
        wandb_tags = []
        if cfg.experiment_tag:
            wandb_tags.append(cfg.experiment_tag)

        # Resume existing wandb run if auto-resuming from crash
        wandb_init_kwargs: dict[str, Any] = {
            "project": cfg.wandb_project,
            "name": cfg.run_name,
            "tags": wandb_tags if wandb_tags else None,
            "config": {
                **cfg.model_dump(),
                "simulated_years_per_step": sim_years_per_step,
                "total_simulated_years": cfg.total_simulated_years,
            },
        }

        # If we have a wandb run ID from auto-resume, reuse that run
        if wandb_run_id:
            wandb_init_kwargs["id"] = wandb_run_id
            wandb_init_kwargs["resume"] = "must"
            logger.info(f"🔄 Resuming existing WandB run: {wandb_run_id}")
        elif start_step > 0:
            # Auto-resuming but no run ID saved (backward compatibility)
            # Try to resume by name, but allow creating new if not found
            wandb_init_kwargs["resume"] = "allow"
            logger.info("🔄 Attempting to resume WandB run by name (fallback)")

        wandb.init(**wandb_init_kwargs)

        model_load_time = time.time() - model_load_start
        metrics["timing"]["model_load_s"] = model_load_time
        logger.info(f"✅ Model loaded in {model_load_time:.2f}s")

        # ==========================================================================
        # 1.5 League Training Initialization (if enabled)
        # ==========================================================================
        league_registry = None
        pfsp_matchmaker = None
        last_registry_reload_step = -1  # Track last reload to throttle

        if cfg.league_training:
            from pathlib import Path

            from src.league import LeagueRegistry, PFSPConfig, PFSPMatchmaker

            logger.info("🏆 League training enabled - initializing registry and matchmaker")

            # Initialize or load league registry (default: per-run file)
            if cfg.league_registry_path:
                registry_path = Path(cfg.league_registry_path)
            else:
                # Per-run league file to avoid key collisions across runs
                registry_path = Path(f"/data/league_{cfg.run_name}.json")

            logger.info(f"📂 League registry path: {registry_path}")
            league_registry = LeagueRegistry(registry_path, run_name=cfg.run_name)

            # Optionally inherit opponents from a previous run (for curriculum learning)
            if cfg.league_inherit_from:
                inherit_path = Path(f"/data/league_{cfg.league_inherit_from}.json")
                if inherit_path.exists():
                    logger.info(f"📚 Inheriting opponents from {cfg.league_inherit_from}")
                    parent_registry = LeagueRegistry(inherit_path, run_name=cfg.league_inherit_from)
                    inherited_count = 0
                    for agent in parent_registry.get_checkpoints():
                        # Copy checkpoint as opponent (keep original key for path lookup)
                        if agent.name not in [a.name for a in league_registry.get_all_agents()]:
                            # Only copy if we have required fields (path, step)
                            if agent.path and agent.step is not None:
                                league_registry.add_checkpoint(
                                    name=agent.name,
                                    path=agent.path,
                                    step=agent.step,
                                    parent=agent.parent,
                                    initial_elo=agent.elo,
                                )
                                inherited_count += 1
                    logger.info(
                        f"✅ Inherited {inherited_count} checkpoints from {cfg.league_inherit_from}"
                    )
                else:
                    logger.warning(f"⚠️ Inherit league not found: {inherit_path}")

            # Configure PFSP with weights from config
            pfsp_config = PFSPConfig(
                self_play_weight=cfg.pfsp_self_play_weight,
                peer_weight=cfg.pfsp_peer_weight,
                exploitable_weight=cfg.pfsp_exploitable_weight,
                baseline_weight=cfg.pfsp_baseline_weight,
            )
            pfsp_matchmaker = PFSPMatchmaker(league_registry, pfsp_config)

            logger.info(
                f"📊 League status: {league_registry.num_checkpoints} checkpoints, "
                f"best Elo: {league_registry.best_elo:.0f} ({league_registry.best_agent})"
            )

            # Log league config to WandB
            wandb.config.update(
                {
                    "league_enabled": True,
                    "pfsp_weights": {
                        "self": cfg.pfsp_self_play_weight,
                        "peer": cfg.pfsp_peer_weight,
                        "exploitable": cfg.pfsp_exploitable_weight,
                        "baseline": cfg.pfsp_baseline_weight,
                    },
                }
            )

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

        # Helper function to spawn rollouts (handles both legacy and league modes)
        def spawn_rollouts_batch(hero_adapter_path: str | None) -> list:
            """
            Spawn a batch of rollouts with appropriate opponent sampling.

            Args:
                hero_adapter_path: Path to the hero's LoRA adapter (e.g., "run/adapter_v5"),
                                  or None for base model.
            """
            import random

            nonlocal last_registry_reload_step

            handles = []
            for _ in range(cfg.num_groups_per_step):
                if cfg.league_training and pfsp_matchmaker is not None:
                    # League training: hero uses latest adapter, opponents from registry
                    hero_power = random.choice(pfsp_matchmaker.POWERS)

                    # Sample opponents based on registry (cold start = all baselines)
                    if league_registry and league_registry.num_checkpoints > 0:
                        # Reload registry periodically to get latest Elo updates from evaluate_league
                        # Reload every 5 steps to balance freshness vs. performance
                        # (evaluate_league typically runs every 50 steps, so this ensures we see updates)
                        if step - last_registry_reload_step >= 5:  # type: ignore[possibly-unbound]
                            try:
                                # CRITICAL: Must reload volume first to see commits from evaluate_league
                                # Modal Volumes don't auto-sync - each container has a local view
                                volume.reload()
                                league_registry.reload()
                                last_registry_reload_step = step
                                logger.debug(
                                    f"🔄 Reloaded league registry at step {step} "
                                    f"(best Elo: {league_registry.best_elo:.0f})"
                                )
                            except Exception as e:
                                logger.warning(
                                    f"⚠️ Failed to reload registry: {e}, using cached state"
                                )

                        # Estimate hero's Elo for peer matching
                        # Hero adapter may not be registered yet (we checkpoint periodically)
                        # Use registry lookup if available, otherwise estimate from best_elo
                        hero_agent_name = hero_adapter_path or "base_model"
                        hero_info = league_registry.get_agent(hero_agent_name)
                        if hero_info:
                            estimated_hero_elo = hero_info.elo
                        else:
                            # Checkpoint not registered yet - use best_elo as estimate
                            # Current policy is likely similar strength to best registered checkpoint
                            estimated_hero_elo = league_registry.best_elo

                        # Use public API for clean opponent sampling
                        match_result = pfsp_matchmaker.sample_opponents(
                            hero_agent=hero_agent_name,
                            hero_power=hero_power,
                            num_opponents=6,
                            hero_elo_override=estimated_hero_elo,
                            hero_adapter_path=hero_adapter_path,  # For self-play when unregistered
                        )

                        # Use matched power_adapters, but override hero with exact training adapter
                        # (matchmaker may return a different path for the hero agent)
                        power_adapters = match_result.power_adapters.copy()
                        power_adapters[hero_power] = hero_adapter_path
                    else:
                        # Cold start: use matchmaker's cold start helper
                        match_result = pfsp_matchmaker.get_cold_start_opponents(hero_power)
                        power_adapters = match_result.power_adapters.copy()
                        power_adapters[hero_power] = hero_adapter_path

                    handles.append(
                        run_rollout.spawn(
                            cfg.model_dump(),
                            power_adapters=power_adapters,
                            hero_power=hero_power,
                        )
                    )
                else:
                    # Legacy mode: same adapter for all powers
                    handles.append(run_rollout.spawn(cfg.model_dump(), lora_name=hero_adapter_path))
            return handles

        # Each entry is (handles_list, hero_adapter_path)
        rollout_buffer: deque[tuple[list, str | None]] = deque()

        # Determine initial adapter for buffer (use resumed adapter if applicable)
        initial_adapter: str | None = None
        if start_step > 0:
            # Resuming: use the adapter from the resumed step
            initial_adapter = f"{cfg.run_name}/adapter_v{start_step}"
            logger.info(f"📦 Resuming - buffer will use adapter: {initial_adapter}")

        for _ in range(buffer_depth):
            handles = spawn_rollouts_batch(hero_adapter_path=initial_adapter)
            rollout_buffer.append((handles, initial_adapter))

        total_in_flight = buffer_depth * cfg.num_groups_per_step
        logger.info(
            f"📦 Buffer initialized: {buffer_depth} batches ({total_in_flight} rollouts) in flight"
        )

        for step in range(start_step, cfg.total_steps):
            step_start = time.time()
            step_metrics: dict[str, Any] = {"step": step}
            step_profile: dict[str, Any] | None = {"step": step} if profile_enabled else None

            # A. Wait for OLDEST rollouts (front of buffer)
            current_handles, current_lora_name = rollout_buffer.popleft()
            rollout_start = time.time()
            with stopwatch(f"Benchmark_Rollout_{step}"):
                raw_trajectories = []
                # Aggregate extraction stats across all rollouts in this batch
                step_extraction_stats: dict[str, int | float] = {
                    "orders_expected": 0,
                    "orders_extracted": 0,
                    "empty_responses": 0,
                    "partial_responses": 0,
                    "extraction_rate": 1.0,
                }
                # Aggregate timing stats from rollouts
                max_volume_reload_s = 0.0
                max_rollout_total_s = 0.0

                failed_rollouts = 0
                for handle in current_handles:
                    try:
                        result = handle.get()  # Block until this rollout completes
                    except Exception as e:
                        # Log the failure but continue with other rollouts
                        failed_rollouts += 1
                        logger.warning(
                            f"⚠️ Rollout failed (will continue with others): {type(e).__name__}: {e}"
                        )
                        continue

                    # Unpack new return format: {"trajectories": [...], "extraction_stats": {...}, "timing": {...}}
                    raw_trajectories.extend(result["trajectories"])
                    # Aggregate extraction stats
                    stats = result["extraction_stats"]
                    step_extraction_stats["orders_expected"] += stats["orders_expected"]  # type: ignore[operator]
                    step_extraction_stats["orders_extracted"] += stats["orders_extracted"]  # type: ignore[operator]
                    step_extraction_stats["empty_responses"] += stats["empty_responses"]  # type: ignore[operator]
                    step_extraction_stats["partial_responses"] += stats["partial_responses"]  # type: ignore[operator]
                    # Track max timing stats (slowest rollout determines wait time)
                    timing = result.get("timing", {})
                    max_volume_reload_s = max(
                        max_volume_reload_s, timing.get("volume_reload_s", 0.0)
                    )
                    max_rollout_total_s = max(max_rollout_total_s, timing.get("total_s", 0.0))

                if failed_rollouts > 0:
                    logger.warning(
                        f"⚠️ Step {step}: {failed_rollouts}/{len(current_handles)} rollouts failed"
                    )

                # Compute step-level extraction rate
                orders_expected = step_extraction_stats["orders_expected"]
                orders_extracted = step_extraction_stats["orders_extracted"]
                if isinstance(orders_expected, int) and orders_expected > 0:
                    step_extraction_stats["extraction_rate"] = float(orders_extracted) / float(
                        orders_expected
                    )

            rollout_time = time.time() - rollout_start
            step_metrics["rollout_time_s"] = rollout_time
            step_metrics["raw_trajectories"] = len(raw_trajectories)
            step_metrics["rollout_lora"] = current_lora_name or "base_model"
            step_metrics["buffer_depth_actual"] = (
                len(rollout_buffer) + 1
            )  # +1 for the one we just popped
            step_metrics["extraction_stats"] = step_extraction_stats
            step_metrics["failed_rollouts"] = failed_rollouts
            # Track slowest rollout timing (identifies bottlenecks)
            step_metrics["max_volume_reload_s"] = max_volume_reload_s
            step_metrics["max_rollout_total_s"] = max_rollout_total_s
            if step_profile is not None:
                step_profile["rollout_time_ms"] = rollout_time * 1000

            # B. Launch NEW batch to maintain buffer (if not near the end)
            # This batch runs during training, keeping the pipeline full
            steps_remaining = cfg.total_steps - step - 1
            if steps_remaining >= buffer_depth:
                # Determine which adapter to use for the new batch
                # After step 0 training completes, we'll have adapter_v1
                new_hero_agent: str | None = None

                if step >= 1:
                    adapter_rel_path = f"{cfg.run_name}/adapter_v{step}"
                    adapter_full_path = MODELS_PATH / cfg.run_name / f"adapter_v{step}"
                    policy_model.save_pretrained(str(adapter_full_path))
                    volume.commit()
                    logger.info(f"Saved adapter to {adapter_full_path}")

                    # League training: Add checkpoint to registry if criteria met
                    if cfg.league_training and league_registry is not None:
                        from src.league import should_add_to_league

                        # Use full path as key to ensure uniqueness across runs
                        # Format: "{run_name}/adapter_v{step}" e.g., "grpo-20251206/adapter_v50"
                        checkpoint_key = adapter_rel_path  # Already "{run_name}/adapter_v{step}"
                        parent_key = (
                            f"{cfg.run_name}/adapter_v{step - 1}" if step > 1 else "base_model"
                        )

                        if should_add_to_league(step, league_registry):
                            league_registry.add_checkpoint(
                                name=checkpoint_key,
                                path=adapter_rel_path,
                                step=step,
                                parent=parent_key,
                            )
                            # CRITICAL: Commit registry to volume so evaluate_league can see it
                            volume.commit()
                            logger.info(f"🏆 Added checkpoint {checkpoint_key} to league")

                            # Spawn async Elo evaluation if enabled
                            if (
                                cfg.elo_eval_every_n_steps > 0
                                and step % cfg.elo_eval_every_n_steps == 0
                            ):
                                logger.info(f"🎯 Spawning Elo evaluation for {checkpoint_key}")
                                # Get league registry path
                                registry_path_str = str(
                                    f"/data/league_{cfg.run_name}.json"
                                    if not cfg.league_registry_path
                                    else cfg.league_registry_path
                                )
                                # Spawn async - doesn't block training
                                evaluate_league.spawn(
                                    challenger_path=adapter_rel_path,
                                    league_registry_path=registry_path_str,
                                    games_per_opponent=cfg.elo_eval_games_per_opponent,
                                    max_years=cfg.rollout_horizon_years,
                                    model_id=cfg.base_model_id,
                                    wandb_run_id=wandb.run.id if wandb.run else None,
                                    training_step=step,
                                )

                    # Always use the adapter path directly for rollouts
                    # (checkpoint registry is for Elo tracking, not adapter loading)
                    new_hero_agent = adapter_rel_path

                target_step = step + buffer_depth
                logger.info(
                    f"🔀 Launching rollouts for step {target_step} "
                    f"(using {'base model' if not new_hero_agent else new_hero_agent})"
                )

                new_handles = spawn_rollouts_batch(hero_adapter_path=new_hero_agent)
                rollout_buffer.append((new_handles, new_hero_agent))

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

            extraction_rate_pct = float(step_extraction_stats["extraction_rate"]) * 100
            logger.info(
                f"Step {step}: loss={avg_loss:.4f} | kl={avg_kl:.4f} | "
                f"reward={traj_stats.reward_mean:.2f}±{traj_stats.reward_std:.2f} | "
                f"extraction={extraction_rate_pct:.1f}% | "
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
                    # Rollout timing breakdown (diagnose spikes)
                    "rollout/max_volume_reload_s": max_volume_reload_s,
                    "rollout/max_total_s": max_rollout_total_s,
                    "rollout/failed_count": failed_rollouts,
                    # Order extraction metrics (monitor prompt structure regressions)
                    "extraction/rate": step_extraction_stats["extraction_rate"],
                    "extraction/orders_expected": step_extraction_stats["orders_expected"],
                    "extraction/orders_extracted": step_extraction_stats["orders_extracted"],
                    "extraction/empty_responses": step_extraction_stats["empty_responses"],
                    "extraction/partial_responses": step_extraction_stats["partial_responses"],
                    # Power Law metrics (for X-axis comparison across runs)
                    "power_law/cumulative_simulated_years": cumulative_sim_years,
                    "power_law/simulated_years_per_step": sim_years_per_step,
                    "power_law/reward_at_compute": traj_stats.reward_mean,
                }
            )
            if profiler is not None:
                profiler.step()

            # Periodic checkpoint save for resume capability
            if cfg.save_state_every_n_steps > 0 and (step + 1) % cfg.save_state_every_n_steps == 0:
                save_training_state(step + 1)

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
        # Note: Autoscaler is configured via buffer_containers=1 at the class decorator level.
        # Containers will scale down automatically based on scaledown_window after inactivity.
        logger.info("🔄 Training complete. Containers will scale down after inactivity.")

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


# ------------------------------------------------------------------------------
# 9.1 Async Elo Evaluator (runs in background during training)
# ------------------------------------------------------------------------------


@app.function(
    image=cpu_image,
    timeout=60 * 60,  # 1 hour max per evaluation
    retries=0,  # Don't retry - if it fails, training continues
    secrets=[
        modal.Secret.from_name("axiom-secrets"),
        modal.Secret.from_name("wandb-secret"),
    ],
    volumes={str(VOLUME_PATH): volume},
)
async def evaluate_league(
    challenger_path: str,
    league_registry_path: str,
    games_per_opponent: int = 3,
    max_years: int = 5,
    model_id: str = "Qwen/Qwen2.5-7B-Instruct",
    wandb_run_id: str | None = None,
    training_step: int = 0,
) -> dict:
    """
    Run Elo evaluation for a new checkpoint against gatekeepers.

    This function is spawned asynchronously during training to update Elo
    ratings without blocking the training loop.

    Gatekeepers are:
    - All baseline bots (random_bot, chaos_bot)
    - Top N checkpoints by Elo from the registry

    Args:
        challenger_path: Path to the new checkpoint (e.g., "run-name/adapter_v50")
        league_registry_path: Path to league.json
        games_per_opponent: Games per gatekeeper (default 3 for speed)
        max_years: Max years per game
        model_id: Base model ID
        wandb_run_id: WandB run ID to log to (from training)
        training_step: Current training step (for logging)

    Returns:
        Dict with Elo updates and match results
    """
    import random
    import time
    from pathlib import Path

    import wandb

    from src.agents import LLMAgent, PromptConfig
    from src.agents.baselines import ChaosBot, RandomBot
    from src.engine.wrapper import DiplomacyWrapper
    from src.league import LeagueRegistry, update_elo_from_match
    from src.utils.observability import axiom, logger
    from src.utils.parsing import extract_orders
    from src.utils.scoring import calculate_final_scores

    eval_start = time.time()
    logger.info(f"🏆 Starting Elo evaluation for {challenger_path}")

    # Load league registry
    volume.reload()
    registry = LeagueRegistry(Path(league_registry_path))

    # Get gatekeepers: baselines + top checkpoints
    baselines = registry.get_baselines()
    checkpoints = registry.get_checkpoints()

    # Select top 3 checkpoints by Elo (excluding the challenger itself)
    top_checkpoints = sorted(
        [c for c in checkpoints if c.path != challenger_path],
        key=lambda x: x.elo,
        reverse=True,
    )[:3]

    gatekeepers = baselines + top_checkpoints
    logger.info(f"📊 Gatekeepers: {[g.name for g in gatekeepers]}")

    # Baseline bots
    BASELINE_BOTS = {
        "random_bot": RandomBot(),
        "chaos_bot": ChaosBot(),
        "base_model": None,  # Placeholder for base model
    }

    POWERS = ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]

    # Initialize agents
    prompt_config = PromptConfig(compact_mode=True, prefix_cache_optimized=True)
    llm_agent = LLMAgent(config=prompt_config)

    # Track all matches for Elo updates
    all_matches = []
    match_results = []

    for gatekeeper in gatekeepers:
        logger.info(f"  vs {gatekeeper.name}...")

        for game_idx in range(games_per_opponent):
            # Randomly assign challenger to a power
            challenger_power = random.choice(POWERS)
            opponent_powers = [p for p in POWERS if p != challenger_power]

            # Build power_adapters
            power_adapters: dict[str, str | None] = {}
            power_adapters[challenger_power] = challenger_path

            # Assign gatekeeper to all other powers
            for p in opponent_powers:
                if gatekeeper.agent_type.value == "baseline":
                    power_adapters[p] = gatekeeper.name  # "random_bot" or "chaos_bot"
                else:
                    power_adapters[p] = gatekeeper.path  # Checkpoint path

            # Run game
            game = DiplomacyWrapper(horizon=max_years * 2)
            step_count = 0
            step_timings = []  # Track timing per step for analysis

            while not game.is_done():
                step_start = time.time()
                step_count += 1
                all_orders = []

                # OPTIMIZATION: Batch inference calls by adapter (like run_rollout does)
                # This dramatically speeds up LLM vs LLM matchups
                adapter_groups: dict[str | None, list[tuple[str, str, dict]]] = {}
                baseline_orders: dict[str, list[str]] = {}

                # Phase 1: Collect prompts and group by adapter
                prompt_start = time.time()
                for power in POWERS:
                    adapter = power_adapters.get(power)

                    # Get orders based on adapter type
                    if adapter in BASELINE_BOTS and adapter is not None:
                        bot = BASELINE_BOTS[adapter]
                        if bot:
                            orders = bot.get_orders(game, power)
                            baseline_orders[power] = orders
                        else:
                            baseline_orders[power] = []
                    else:
                        # LLM power - collect prompt for batching
                        prompt, valid_moves = llm_agent.build_prompt(game, power)
                        if valid_moves:
                            adapter_key = adapter or "base_model"
                            if adapter_key not in adapter_groups:
                                adapter_groups[adapter_key] = []
                            adapter_groups[adapter_key].append((power, prompt, valid_moves))
                        else:
                            baseline_orders[power] = []

                prompt_time = time.time() - prompt_start

                # Phase 2: Batch inference for each adapter group
                inference_start = time.time()
                inference_results: dict[str, list[str]] = {}  # power -> orders

                for adapter_key, group_items in adapter_groups.items():
                    group_powers = [item[0] for item in group_items]
                    group_prompts = [item[1] for item in group_items]
                    group_valid_moves = [item[2] for item in group_items]

                    # Batch inference call (much faster than sequential)
                    # Use dedicated league evaluation engine pool (isolated from training)
                    # Note: batch_size=1 is common when only one power uses a particular adapter
                    # vLLM's continuous batching will still batch these across different calls
                    batch_start = time.time()
                    responses = await InferenceEngine(
                        model_id=model_id, is_for_league_evaluation=True
                    ).generate.remote.aio(
                        prompts=group_prompts,
                        valid_moves=group_valid_moves,
                        lora_name=adapter_key if adapter_key != "base_model" else None,
                    )
                    batch_time = time.time() - batch_start

                    # Extract orders for each power
                    for power, response_data in zip(group_powers, responses, strict=True):
                        orders = extract_orders(response_data["text"])
                        inference_results[power] = orders

                    # Log batch timing (only warn for very slow single-item batches)
                    batch_size = len(group_prompts)
                    if batch_size == 1 and batch_time > 5.0:
                        logger.warning(
                            f"  Very slow single-item batch: adapter={adapter_key}, "
                            f"time={batch_time:.3f}s (normal: 2-3s)"
                        )
                    else:
                        logger.debug(
                            f"  Batch inference: adapter={adapter_key}, "
                            f"batch_size={batch_size}, time={batch_time:.3f}s"
                        )

                inference_time = time.time() - inference_start

                # Phase 3: Combine all orders
                combine_start = time.time()
                for power in POWERS:
                    if power in baseline_orders:
                        all_orders.extend(baseline_orders[power])
                    elif power in inference_results:
                        all_orders.extend(inference_results[power])
                combine_time = time.time() - combine_start

                # Phase 4: Step game
                step_game_start = time.time()
                game.step(all_orders)
                step_game_time = time.time() - step_game_start

                step_total = time.time() - step_start
                step_timings.append(
                    {
                        "step": step_count,
                        "prompt_time_s": prompt_time,
                        "inference_time_s": inference_time,
                        "combine_time_s": combine_time,
                        "step_game_time_s": step_game_time,
                        "total_time_s": step_total,
                        "num_llm_powers": len(sum(adapter_groups.values(), [])),
                        "num_baseline_powers": len(baseline_orders),
                    }
                )

                # Log every 5 steps to avoid spam
                if step_count % 5 == 0:
                    logger.info(
                        f"  Step {step_count}: {step_total:.3f}s total "
                        f"(prompt: {prompt_time:.3f}s, inference: {inference_time:.3f}s, "
                        f"game: {step_game_time:.3f}s)"
                    )

            # Compute final scores
            final_scores = calculate_final_scores(game)

            # Log timing summary for this game
            if step_timings:
                total_time = sum(s["total_time_s"] for s in step_timings)
                avg_inference_time = sum(s["inference_time_s"] for s in step_timings) / len(
                    step_timings
                )
                avg_prompt_time = sum(s["prompt_time_s"] for s in step_timings) / len(step_timings)
                avg_game_time = sum(s["step_game_time_s"] for s in step_timings) / len(step_timings)

                logger.info(
                    f"  Game {game_idx} complete: {step_count} steps, {total_time:.2f}s total "
                    f"(avg: prompt={avg_prompt_time:.3f}s, inference={avg_inference_time:.3f}s, "
                    f"game={avg_game_time:.3f}s)"
                )

                # Log to Axiom for analysis
                axiom.log(
                    {
                        "event": "league_eval_game_timing",
                        "challenger_path": challenger_path,
                        "gatekeeper": gatekeeper.name,
                        "game_idx": game_idx,
                        "total_steps": step_count,
                        "total_time_s": total_time,
                        "avg_prompt_time_s": avg_prompt_time,
                        "avg_inference_time_s": avg_inference_time,
                        "avg_game_time_s": avg_game_time,
                        "num_llm_powers": step_timings[-1]["num_llm_powers"] if step_timings else 0,
                    }
                )

            # Build power_agents mapping
            power_agents = {p: power_adapters[p] or "base_model" for p in POWERS}

            # Record match
            match_data = {
                "power_agents": power_agents,
                "power_scores": final_scores,
                "step_timings": step_timings,  # Include timing data
            }
            all_matches.append(match_data)

            # Track for summary
            challenger_score = final_scores[challenger_power]
            gatekeeper_avg = sum(
                final_scores[p]
                for p in opponent_powers
                if power_adapters[p] == gatekeeper.path or power_adapters[p] == gatekeeper.name
            ) / len(opponent_powers)

            match_results.append(
                {
                    "gatekeeper": gatekeeper.name,
                    "game_idx": game_idx,
                    "challenger_score": challenger_score,
                    "gatekeeper_avg_score": gatekeeper_avg,
                    "win": challenger_score > gatekeeper_avg,
                }
            )

            logger.info(
                f"    Game {game_idx + 1}: Challenger {challenger_score:.1f} vs Gatekeeper avg {gatekeeper_avg:.1f}"
            )

    # Compute Elo updates from all matches
    logger.info("📈 Computing Elo updates...")

    # Get current Elos
    all_agents = registry.get_all_agents()
    current_elos = {a.name: a.elo for a in all_agents}

    # Add challenger to registry if not present yet (race condition: eval might start before checkpoint is added)
    challenger_name = challenger_path  # Name matches path for checkpoints
    if challenger_name not in [a.name for a in all_agents]:
        logger.warning(f"⚠️ Challenger {challenger_name} not in registry yet, adding it...")

        # Extract step and run_name from path (e.g., "grpo-20251209-200240/adapter_v20" -> 20)
        step = 0
        run_name = None
        try:
            parts = challenger_path.rsplit("/", 1)
            if len(parts) == 2:
                run_name = parts[0]
            step_str = challenger_path.split("adapter_v")[-1]
            step = int(step_str)
        except (ValueError, IndexError):
            logger.warning(f"Could not extract step from challenger_path: {challenger_path}")

        # Find parent checkpoint to inherit Elo
        # Parent is the previous step's checkpoint (e.g., adapter_v19 is parent of adapter_v20)
        parent_name = None
        parent_elo = 1000.0
        if step > 1 and run_name:
            # Try to find parent checkpoint (previous step)
            for prev_step in range(step - 1, 0, -1):
                potential_parent = f"{run_name}/adapter_v{prev_step}"
                if potential_parent in current_elos:
                    parent_name = potential_parent
                    parent_elo = current_elos[potential_parent]
                    logger.info(f"  Inheriting Elo {parent_elo:.0f} from parent {parent_name}")
                    break
        elif step == 1:
            parent_name = "base_model"
            parent_elo = current_elos.get("base_model", 1000.0)

        # Add challenger to registry
        registry.add_checkpoint(
            name=challenger_name,
            path=challenger_path,
            step=step,
            parent=parent_name,
            initial_elo=parent_elo,
        )
        # Reload current_elos to include the newly added challenger
        all_agents = registry.get_all_agents()
        current_elos = {a.name: a.elo for a in all_agents}

    # Apply Elo updates from all matches
    for match in all_matches:
        current_elos = update_elo_from_match(
            power_agents=match["power_agents"],
            power_scores=match["power_scores"],
            agent_elos=current_elos,
            k=32.0,
        )

    # Update registry with new Elos
    # Only update agents that exist in the registry (bulk_update_elos filters automatically)
    registry.bulk_update_elos(current_elos)

    # Also add match history for tracking
    from src.league.types import MatchResult

    for idx, match in enumerate(all_matches):
        # Calculate rankings from scores
        power_scores = match["power_scores"]
        sorted_powers = sorted(power_scores.items(), key=lambda x: x[1], reverse=True)
        rankings = {power: rank + 1 for rank, (power, _) in enumerate(sorted_powers)}

        # Count years (approximate from step count)
        num_years = len(match.get("step_timings", [])) // 2  # 2 phases per year

        # Determine winner (agent with highest score)
        winner_power = sorted_powers[0][0] if sorted_powers else None
        winner_agent = match["power_agents"].get(winner_power) if winner_power else None

        match_result = MatchResult(
            game_id=f"eval-{challenger_name}-{int(time.time())}-{idx}",
            step=training_step,
            power_agents=match["power_agents"],
            scores=power_scores,
            rankings=rankings,
            num_years=num_years,
            winner=winner_agent,
        )
        registry.add_match(match_result)

    # Final save to ensure match history is persisted
    registry._save()
    volume.commit()

    # Compute summary stats
    challenger_new_elo = current_elos.get(challenger_path, 1000.0)
    wins = sum(1 for m in match_results if m["win"])
    total_games = len(match_results)
    win_rate = wins / total_games if total_games > 0 else 0.0

    eval_duration = time.time() - eval_start

    logger.info(f"✅ Elo evaluation complete in {eval_duration:.1f}s")
    logger.info(f"   Challenger Elo: {challenger_new_elo:.0f}")
    logger.info(f"   Win rate: {win_rate:.1%} ({wins}/{total_games})")

    # Log to WandB if run ID provided
    if wandb_run_id:
        try:
            wandb.init(id=wandb_run_id, resume="allow", project="diplomacy-grpo")
            wandb.log(
                {
                    "elo/challenger": challenger_new_elo,
                    "elo/win_rate": win_rate,
                    "elo/games_played": total_games,
                    "elo/evaluation_step": training_step,
                }
            )
            # Log Elo for all tracked agents
            for agent_name, elo in current_elos.items():
                if agent_name in [a.name for a in all_agents]:
                    safe_name = agent_name.replace("/", "_")
                    wandb.log({f"elo/{safe_name}": elo})
        except Exception as e:
            logger.warning(f"Failed to log to WandB: {e}")

    await axiom.flush()

    return {
        "challenger_path": challenger_path,
        "challenger_elo": challenger_new_elo,
        "win_rate": win_rate,
        "games_played": total_games,
        "elo_updates": current_elos,
        "duration_s": eval_duration,
    }


# ------------------------------------------------------------------------------
# 9.2 Standard Evaluation Runner
# ------------------------------------------------------------------------------


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

                    for response_data, valid_moves in zip(
                        responses, checkpoint_valid_moves, strict=True
                    ):
                        orders = extract_orders(response_data["text"])
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
