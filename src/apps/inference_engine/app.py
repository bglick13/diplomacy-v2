import asyncio
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import modal
from transformers import AutoTokenizer

from src.apps.common.images import gpu_image
from src.apps.common.volumes import HF_CACHE_PATH, VOLUME_PATH, hf_cache_volume, volume
from src.apps.inference_engine.types import GenerateBatchResponseItem, GenerationResponse
from src.inference.logits import DiplomacyLogitsProcessor
from src.utils.observability import (
    axiom,
    log_inference_request,
    log_inference_response,
    log_prefix_cache_stats,
)

with gpu_image.imports():
    from vllm import AsyncEngineArgs
    from vllm.lora.request import LoRARequest
    from vllm.sampling_params import SamplingParams
    from vllm.v1.engine.async_llm import AsyncLLM

app = modal.App("diplomacy-grpo-inference-engine")


# ============================================================================
# ADAPTER MANAGEMENT
# ============================================================================


@dataclass
class AdapterManager:
    """Manages LoRA adapter loading and caching."""

    loaded_adapters: set[str] = field(default_factory=set)
    adapter_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def load_adapter(self, lora_name: str) -> LoRARequest:
        """
        Load a LoRA adapter, handling volume reloads and retries.

        Returns:
            LoRARequest for use with vLLM engine
        """
        full_path = f"/data/models/{lora_name}"

        # Use lock to serialize volume reloads and prevent concurrent conflicts
        if lora_name not in self.loaded_adapters:
            async with self.adapter_lock:
                # Double-check after acquiring lock
                if lora_name not in self.loaded_adapters:
                    print(f"📂 Loading NEW adapter: {lora_name}")
                    await self._wait_for_adapter(full_path, lora_name)
                    self.loaded_adapters.add(lora_name)
                else:
                    print(f"📂 Adapter already loaded by another request: {lora_name}")
        else:
            print(f"📂 Using cached adapter: {lora_name}")

        # Create LoRA request (safe to do concurrently)
        lora_int_id = abs(hash(lora_name)) % (2**31)
        lora_req = LoRARequest(lora_name, lora_int_id, full_path)
        print(f"🔧 Created LoRARequest: name={lora_name}, id={lora_int_id}")

        return lora_req

    async def load_adapters(self, lora_names: list[str]) -> dict[str, LoRARequest]:
        """
        Load multiple LoRA adapters concurrently.

        Args:
            lora_names: List of adapter names to load

        Returns:
            Dict mapping adapter name to LoRARequest
        """
        if not lora_names:
            return {}

        # Load all adapters concurrently
        tasks = [self.load_adapter(name) for name in lora_names]
        results = await asyncio.gather(*tasks)

        return dict(zip(lora_names, results, strict=True))

    async def _wait_for_adapter(self, full_path: str, lora_name: str) -> None:
        """Wait for adapter to appear with retries."""
        max_retries = 5

        for attempt in range(max_retries):
            # Only reload volume if path doesn't exist yet
            if not os.path.exists(full_path):
                try:
                    volume.reload()
                except Exception as e:
                    print(f"⚠️ Volume reload warning: {e}")

            if os.path.exists(full_path):
                adapter_files = os.listdir(full_path)
                print(f"✅ LoRA adapter found. Files: {adapter_files}")
                return

            if attempt < max_retries - 1:
                wait_time = 1 + attempt
                print(f"⏳ Adapter not found (attempt {attempt + 1}), waiting {wait_time}s...")
                await asyncio.sleep(wait_time)

        # Failed after all retries
        models_dir = "/data/models"
        if os.path.exists(models_dir):
            contents = os.listdir(models_dir)
            print(f"📁 Contents of {models_dir}: {contents}")

        raise RuntimeError(f"LoRA path does not exist after retries: {full_path}")


# ============================================================================
# CACHE STATS TRACKING
# ============================================================================


@dataclass
class CacheStatsTracker:
    """Tracks prefix cache statistics for observability."""

    total_queries: int = 0
    total_hits: int = 0
    total_prompt_tokens: int = 0
    batches_processed: int = 0
    batch_size_total: int = 0
    real_stats_available: bool = False

    def update(
        self,
        batch_size: int,
        prompt_tokens: int,
        cache_hit_rate: float | None = None,
        cache_queries: int | None = None,
        cache_hits: int | None = None,
    ) -> None:
        """Update cumulative stats from a batch."""
        self.batches_processed += 1
        self.total_prompt_tokens += prompt_tokens
        self.batch_size_total += batch_size

        # If we got real stats from vLLM, track them
        if cache_queries is not None:
            self.total_queries += cache_queries
            self.real_stats_available = True
        else:
            # Fallback: count requests as queries
            self.total_queries += batch_size

        if cache_hits is not None:
            self.total_hits += cache_hits
        elif cache_hit_rate is not None:
            # Estimate hits from hit rate
            self.total_hits += int(batch_size * cache_hit_rate)

    def to_dict(self) -> dict:
        """Export stats as dictionary."""
        return {
            "total_queries": self.total_queries,
            "total_hits": self.total_hits,
            "total_prompt_tokens": self.total_prompt_tokens,
            "batches_processed": self.batches_processed,
            "batch_size_total": self.batch_size_total,
            "real_stats_available": self.real_stats_available,
        }


def get_vllm_cache_stats() -> dict | None:
    """
    Extract prefix cache stats from vLLM's Prometheus metrics.

    vLLM v1 exposes prefix cache metrics as Prometheus counters:
    - vllm:prefix_cache_queries (Counter)
    - vllm:prefix_cache_hits (Counter)

    Returns:
        dict with hit_rate, queries, hits if available, else None
    """
    try:
        from prometheus_client import REGISTRY

        queries = 0
        hits = 0

        # Look for vLLM prefix cache counters
        for metric in REGISTRY.collect():
            metric_name = metric.name.replace(":", "_")

            if "prefix_cache_queries" in metric_name:
                for sample in metric.samples:
                    if sample.name.endswith("_total") or sample.name == metric.name:
                        queries = int(sample.value)
                        break

            elif "prefix_cache_hits" in metric_name:
                for sample in metric.samples:
                    if sample.name.endswith("_total") or sample.name == metric.name:
                        hits = int(sample.value)
                        break

        if queries > 0:
            hit_rate = hits / queries
            return {"hit_rate": hit_rate, "queries": queries, "hits": hits}

    except ImportError:
        pass
    except Exception as e:
        print(f"⚠️ Could not get prefix cache stats: {e}")

    return None


# ============================================================================
# LOGPROBS PARSING
# ============================================================================


def parse_completion_logprobs(logprobs_data: Any, token_ids: list[int]) -> list[float]:
    """
    Parse completion logprobs from vLLM output.

    Args:
        logprobs_data: SampleLogprobs from vLLM (FlatLogprobs or list[dict])
        token_ids: Token IDs for the completion

    Returns:
        List of logprobs for each token
    """
    completion_logprobs: list[float] = []

    if not logprobs_data:
        return completion_logprobs

    # Handle FlatLogprobs format
    if hasattr(logprobs_data, "logprobs"):
        for i in range(len(logprobs_data)):
            pos_logprobs = logprobs_data[i]
            if pos_logprobs and token_ids[i] in pos_logprobs:
                completion_logprobs.append(pos_logprobs[token_ids[i]].logprob)
            else:
                completion_logprobs.append(0.0)
    else:
        # Handle list[dict[int, Logprob]] format
        for i, pos_logprobs in enumerate(logprobs_data):
            if pos_logprobs and token_ids[i] in pos_logprobs:
                completion_logprobs.append(pos_logprobs[token_ids[i]].logprob)
            else:
                completion_logprobs.append(0.0)

    return completion_logprobs


def parse_prompt_logprobs(
    prompt_logprobs: list, prompt_token_ids: list[int], prompt_len: int
) -> float:
    """
    Parse completion logprobs from prompt_logprobs (for scoring).

    Args:
        prompt_logprobs: List of logprobs from vLLM output
        prompt_token_ids: Token IDs for the full sequence
        prompt_len: Length of prompt (completion starts after this)

    Returns:
        Sum of completion logprobs
    """
    completion_logprobs_sum = 0.0

    # prompt_logprobs is list[dict[token_id, Logprob] | None]
    # Index 0 is None (no logprob for first token)
    # Completion tokens start at index prompt_len
    for i in range(prompt_len, len(prompt_logprobs)):
        pos_logprobs = prompt_logprobs[i]
        if pos_logprobs:
            # Get the actual token at this position
            actual_token_id = prompt_token_ids[i] if i < len(prompt_token_ids) else None
            if actual_token_id is not None and actual_token_id in pos_logprobs:
                completion_logprobs_sum += pos_logprobs[actual_token_id].logprob

    return completion_logprobs_sum


# ============================================================================
# ERROR HANDLING
# ============================================================================


def is_fatal_vllm_error(error: Exception) -> bool:
    """Check if error is fatal and requires container restart."""
    error_str = str(error).lower()
    error_type = type(error).__name__

    return (
        "enginedeaderror" in error_type.lower()
        or "enginedead" in error_str
        or ("engine" in error_str and "dead" in error_str)
        or ("cuda" in error_str and "out of memory" in error_str)
        or ("cudnn" in error_str and "error" in error_str)
    )


def handle_fatal_error(error: Exception) -> None:
    """Handle fatal vLLM errors by forcing container restart."""
    if is_fatal_vllm_error(error):
        print("💀 FATAL: vLLM engine is dead. Killing container for fresh restart...")
        os._exit(1)  # Force immediate exit


# ============================================================================
# INFERENCE ENGINE
# ============================================================================


@app.cls(
    image=gpu_image,
    gpu="A100",
    volumes={
        str(VOLUME_PATH): volume,
        str(HF_CACHE_PATH): hf_cache_volume,
    },
    secrets=[modal.Secret.from_name("axiom-secrets")],
    scaledown_window=60 * 10,
    buffer_containers=2,
)
@modal.concurrent(max_inputs=20, target_inputs=8)
class InferenceEngine:
    model_id: str = modal.parameter(default="Qwen/Qwen2.5-7B-Instruct")
    is_for_league_evaluation: bool = modal.parameter(default=False)

    @modal.enter()
    def setup(self):
        """Initialize vLLM engine with optimized configuration."""
        print("🥶 Initializing vLLM v1 Engine...")

        # Setup HuggingFace cache
        self._setup_hf_cache()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            cache_dir=str(HF_CACHE_PATH / "transformers"),
        )

        # Initialize managers
        self.adapter_mgr = AdapterManager()
        self.cache_stats = CacheStatsTracker()

        # Initialize vLLM engine
        self.engine = self._create_engine()

        # Check stat logger availability
        self._check_stat_logger()

        print("✅ Engine Ready (with dynamic LoRA support).")

    def _setup_hf_cache(self) -> None:
        """Setup HuggingFace cache directories."""
        os.environ["HF_HOME"] = str(HF_CACHE_PATH)
        os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE_PATH / "transformers")
        os.environ["HF_HUB_CACHE"] = str(HF_CACHE_PATH / "hub")

        for cache_dir in [
            HF_CACHE_PATH / "transformers",
            HF_CACHE_PATH / "hub",
        ]:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def _create_engine(self) -> AsyncLLM:
        """Create vLLM engine with optimized settings."""
        # vLLM v1 Performance Tuning
        # See: https://docs.vllm.ai/en/latest/performance/tuning.html
        gpu_memory_util = 0.92  # High utilization for throughput
        max_num_seqs = 512  # Large batch size for concurrent requests
        max_num_batched_tokens = 16384  # Balance throughput vs latency

        engine_args = AsyncEngineArgs(
            model=self.model_id,  # pyright: ignore[reportCallIssue]
            # LoRA configuration for league training
            enable_lora=True,  # pyright: ignore[reportCallIssue]
            max_loras=8,  # Up to 7 opponents + 1 hero  # pyright: ignore[reportCallIssue]
            max_lora_rank=16,  # Must match training LoRA rank  # pyright: ignore[reportCallIssue]
            # Performance tuning
            gpu_memory_utilization=gpu_memory_util,  # pyright: ignore[reportCallIssue]
            max_num_seqs=max_num_seqs,  # pyright: ignore[reportCallIssue]
            max_num_batched_tokens=max_num_batched_tokens,  # pyright: ignore[reportCallIssue]
            # Prefix caching for shared prompt prefixes
            enable_prefix_caching=True,  # pyright: ignore[reportCallIssue]
            disable_log_stats=False,  # pyright: ignore[reportCallIssue]
            # Custom logits processor
            logits_processors=[DiplomacyLogitsProcessor],  # pyright: ignore[reportCallIssue]
            # Cache optimizations
            download_dir=str(HF_CACHE_PATH / "hub"),  # pyright: ignore[reportCallIssue]
            load_format="auto",  # Prefer safetensors for faster loading  # pyright: ignore[reportCallIssue]
        )

        print(
            f"⚙️  vLLM Config: max_num_seqs={max_num_seqs}, "
            f"max_num_batched_tokens={max_num_batched_tokens}, "
            f"gpu_memory_util={gpu_memory_util}"
        )

        return AsyncLLM.from_engine_args(engine_args)

    def _check_stat_logger(self) -> None:
        """Check if vLLM stat logger is available."""
        try:
            if self.engine.logger_manager:
                print("✅ vLLM stat_loggers available for metrics")
            else:
                print("⚠️ vLLM stat_loggers not available - using fallback metrics")
        except Exception as e:
            print(f"⚠️ Could not check stat_loggers: {e}")

    @modal.method()
    def get_cache_stats(self) -> dict:
        """Get current prefix cache statistics."""
        vllm_stats = get_vllm_cache_stats()
        return {
            "vllm_stats": vllm_stats,
            "cumulative": self.cache_stats.to_dict(),
        }

    @modal.method()
    async def generate(
        self,
        prompts: list[str],
        valid_moves: list[dict],
        lora_name: str | None = None,
        lora_names: list[str | None] | None = None,
        temperature: float = 0.8,
        max_new_tokens: int = 256,
    ) -> list[GenerateBatchResponseItem]:
        """
        Generate responses for the given prompts.

        Args:
            prompts: List of prompt strings
            valid_moves: List of valid moves dicts for each prompt
            lora_name: Single LoRA adapter for all prompts (legacy, for backwards compat)
            lora_names: Per-prompt LoRA adapters (preferred for mixed-adapter batches)
            temperature: Sampling temperature
            max_new_tokens: Maximum tokens to generate

        Returns:
            List of generation responses with text, tokens, and logprobs

        Note:
            If both lora_name and lora_names are provided, lora_names takes precedence.
            Use lora_names=["adapter1", None, "adapter2", ...] for mixed batches where
            some prompts use base model (None) and others use specific adapters.
        """
        batch_size = len(prompts)
        request_start = time.time()

        timing = {
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
            # Determine per-prompt adapters
            adapter_start = time.time()

            # Build per-prompt lora_reqs (type annotation needed for pyright)
            per_prompt_lora_reqs: list[LoRARequest | None]

            if lora_names is not None:
                # Per-prompt adapters (new batched mode)
                assert len(lora_names) == batch_size, (
                    f"lora_names length ({len(lora_names)}) must match prompts ({batch_size})"
                )
                # Get unique non-None adapter names
                unique_adapters = list({name for name in lora_names if name is not None})
                # Load all unique adapters concurrently
                adapter_map = await self.adapter_mgr.load_adapters(unique_adapters)
                # Build per-prompt lora_reqs list
                per_prompt_lora_reqs = [
                    adapter_map.get(name) if name else None for name in lora_names
                ]
                # For logging, show unique adapters
                lora_name_for_logging = f"mixed:{len(unique_adapters)}_adapters"
            elif lora_name is not None:
                # Single adapter for all prompts (legacy mode)
                lora_req = await self.adapter_mgr.load_adapter(lora_name)
                per_prompt_lora_reqs = [lora_req] * batch_size
                lora_name_for_logging = lora_name
            else:
                # No adapters (base model only)
                per_prompt_lora_reqs = [None] * batch_size
                lora_name_for_logging = None

            timing["adapter_load_time_s"] = time.time() - adapter_start

            # Generate responses
            generation_start = time.time()
            responses = await self._generate_batch(
                prompts, valid_moves, per_prompt_lora_reqs, temperature, max_new_tokens
            )
            timing["generation_time_s"] = time.time() - generation_start

            # Calculate metrics
            duration_ms = int((time.time() - request_start) * 1000)
            timing["total_time_s"] = duration_ms / 1000.0

            total_tokens = sum(resp["token_count"] for resp in responses)
            tokens_per_second = total_tokens / (duration_ms / 1000) if duration_ms > 0 else None
            total_prompt_tokens = sum(len(resp["prompt_token_ids"]) for resp in responses)

            # Get and update cache stats
            cache_stats = get_vllm_cache_stats()
            cache_hit_rate = cache_stats["hit_rate"] if cache_stats else None
            cache_queries = cache_stats.get("queries") if cache_stats else None
            cache_hits = cache_stats.get("hits") if cache_stats else None

            self.cache_stats.update(
                batch_size, total_prompt_tokens, cache_hit_rate, cache_queries, cache_hits
            )

            # Logging
            self._log_generation_metrics(
                batch_size,
                duration_ms,
                total_tokens,
                tokens_per_second,
                total_prompt_tokens,
                lora_name_for_logging,
                timing,
                cache_hit_rate,
                cache_queries,
                cache_hits,
            )

            # Flush axiom events asynchronously
            asyncio.create_task(axiom.flush())

            # Return formatted responses
            return [
                {
                    "text": resp["text"],
                    "token_ids": resp["token_ids"],
                    "prompt_token_ids": resp["prompt_token_ids"],
                    "completion_logprobs": resp["completion_logprobs"],
                }
                for resp in responses
            ]

        except Exception as e:
            error_msg = f"GPU Inference Failed: {type(e).__name__}: {str(e)}"
            print(error_msg)
            handle_fatal_error(e)
            raise RuntimeError(error_msg) from None

    async def _generate_batch(
        self,
        prompts: list[str],
        valid_moves: list[dict],
        lora_reqs: list[LoRARequest | None],
        temperature: float,
        max_new_tokens: int,
    ) -> list[GenerationResponse]:
        """Generate responses for a batch of prompts concurrently.

        Args:
            prompts: List of prompt strings
            valid_moves: List of valid moves dicts for each prompt
            lora_reqs: Per-prompt LoRA requests (None = use base model)
            temperature: Sampling temperature
            max_new_tokens: Maximum tokens to generate
        """

        async def _generate_single(
            prompt: str, moves: dict, lora_req: LoRARequest | None
        ) -> GenerationResponse:
            """Generate for a single prompt with optional LoRA adapter."""
            request_id = str(uuid.uuid4())
            sampling_params = SamplingParams(  # type: ignore[call-arg, misc]
                temperature=temperature,  # type: ignore[arg-type]
                max_tokens=max_new_tokens,  # type: ignore[arg-type]
                extra_args={"valid_moves_dict": moves, "start_active": True},
                stop=["</orders>", "</Orders>"],  # type: ignore[arg-type]
                logprobs=1,  # type: ignore[arg-type]
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

                # Parse output
                text = ""
                token_ids: list[int] = []
                prompt_token_ids: list[int] = []
                completion_logprobs: list[float] = []

                if final_output:
                    prompt_token_ids = final_output.prompt_token_ids or []

                    if final_output.outputs:
                        output = final_output.outputs[0]
                        text = output.text
                        token_ids = list(output.token_ids)

                        # Extract logprobs
                        if output.logprobs:
                            completion_logprobs = parse_completion_logprobs(
                                output.logprobs, token_ids
                            )

                return {
                    "text": text,
                    "token_count": len(token_ids),
                    "token_ids": token_ids,
                    "prompt_token_ids": prompt_token_ids,
                    "completion_logprobs": completion_logprobs,
                }

            except Exception as e:
                print(f"❌ Generation Error: {e}")
                raise e

        # Process all prompts concurrently - vLLM handles multi-adapter batching internally
        # with max_loras=8 and dynamic batching
        results = await asyncio.gather(
            *[
                _generate_single(prompt, moves, lora_req)
                for prompt, moves, lora_req in zip(prompts, valid_moves, lora_reqs, strict=True)
            ]
        )

        return list(results)

    def _log_generation_metrics(
        self,
        batch_size: int,
        duration_ms: int,
        total_tokens: int,
        tokens_per_second: float | None,
        total_prompt_tokens: int,
        lora_name: str | None,
        timing: dict,
        cache_hit_rate: float | None,
        cache_queries: int | None,
        cache_hits: int | None,
    ) -> None:
        """Log comprehensive generation metrics."""
        # Standard inference logging
        log_inference_response(
            rollout_id="inference-engine",
            batch_size=batch_size,
            duration_ms=duration_ms,
            tokens_generated=total_tokens,
            tokens_per_second=tokens_per_second,
            prefix_cache_hit_rate=cache_hit_rate,
            prefix_cache_queries=cache_queries,
            prefix_cache_hits=cache_hits,
        )

        # Calculate per-phase throughput
        gen_time_s = timing.get("generation_time_s", 0) or 0.001
        input_tps = total_prompt_tokens / gen_time_s
        output_tps = total_tokens / gen_time_s

        # Detailed timing breakdown
        axiom.log(
            {
                "event": "inference_timing_breakdown",
                "batch_size": batch_size,
                "lora_name": lora_name,
                "adapter_load_time_s": timing["adapter_load_time_s"],
                "generation_time_s": timing["generation_time_s"],
                "total_time_s": timing["total_time_s"],
                "tokens_generated": total_tokens,
                "tokens_per_second": tokens_per_second,
                "input_tokens_per_second": round(input_tps, 1),
                "output_tokens_per_second": round(output_tps, 1),
                "prompt_tokens": total_prompt_tokens,
                "prefix_cache_hit_rate": cache_hit_rate,
            }
        )

        # Log cache stats for meaningful batches
        if cache_hit_rate is not None and batch_size >= 4:
            log_prefix_cache_stats(
                batch_id=str(uuid.uuid4())[:8],
                hit_rate=cache_hit_rate,
                queries=cache_queries or 0,
                hits=cache_hits or 0,
                prompt_tokens_total=total_prompt_tokens,
                prompt_tokens_cached=int(total_prompt_tokens * cache_hit_rate),
            )

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

        Args:
            prompts: List of prompt strings
            completions: List of completion strings to score
            prompt_token_ids_list: List of prompt token IDs

        Returns:
            List of dicts with 'ref_completion_logprobs'
        """
        request_start = time.time()

        async def _score_single(prompt: str, completion: str, prompt_token_ids: list[int]) -> dict:
            """Score a single sequence using base model."""
            request_id = str(uuid.uuid4())

            # Validate prompt_token_ids - empty means we can't compute valid ref logprobs
            if not prompt_token_ids:
                print(
                    f"⚠️ Empty prompt_token_ids for request {request_id}, cannot compute ref logprobs"
                )
                return {"ref_completion_logprobs": None}

            # Scoring params: generate 1 token (min required) but only use prompt_logprobs
            sampling_params = SamplingParams(  # type: ignore[call-arg, misc]
                max_tokens=1,  # type: ignore[arg-type]
                prompt_logprobs=1,  # type: ignore[arg-type]
            )

            full_sequence = prompt + completion

            try:
                # Score with base model (no LoRA)
                generator = self.engine.generate(
                    prompt=full_sequence,
                    sampling_params=sampling_params,
                    request_id=request_id,
                    lora_request=None,  # Use base model for reference
                )

                final_output = None
                async for output in generator:
                    final_output = output

                # Extract completion logprobs - return None if data is incomplete
                if (
                    not final_output
                    or not final_output.prompt_logprobs
                    or not final_output.prompt_token_ids
                ):
                    print(
                        f"⚠️ Incomplete output for request {request_id}: "
                        f"output={bool(final_output)}, "
                        f"prompt_logprobs={bool(final_output and final_output.prompt_logprobs)}, "
                        f"prompt_token_ids={bool(final_output and final_output.prompt_token_ids)}"
                    )
                    return {"ref_completion_logprobs": None}

                prompt_len = len(prompt_token_ids)
                completion_logprobs_sum = parse_prompt_logprobs(
                    final_output.prompt_logprobs,  # pyright: ignore[reportArgumentType]
                    final_output.prompt_token_ids,
                    prompt_len,
                )

                return {"ref_completion_logprobs": completion_logprobs_sum}

            except Exception as e:
                print(f"❌ Scoring Error: {e}")
                handle_fatal_error(e)
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
