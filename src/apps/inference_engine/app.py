import asyncio
import os
import time
import uuid

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


@app.cls(
    image=gpu_image,
    gpu="A100",
    volumes={
        str(VOLUME_PATH): volume,
        str(HF_CACHE_PATH): hf_cache_volume,
    },
    scaledown_window=60 * 10,
    buffer_containers=1,
)
@modal.concurrent(max_inputs=512, target_inputs=400)
class InferenceEngine:
    model_id: str = modal.parameter(default="Qwen/Qwen2.5-7B-Instruct")
    is_for_league_evaluation: bool = modal.parameter(default=False)

    @modal.enter()
    def setup(self):
        print("🥶 Initializing vLLM v1 Engine...")

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

        # Prefix cache tracking for observability
        self._cache_stats = {
            "total_queries": 0,
            "total_hits": 0,
            "total_prompt_tokens": 0,
            "batches_processed": 0,
        }

        gpu_memory_util = 0.92
        max_num_seqs = 512
        max_num_batched_tokens = 16384

        engine_args = AsyncEngineArgs(
            model=self.model_id,  # pyright: ignore[reportCallIssue]
            enable_lora=True,  # pyright: ignore[reportCallIssue]
            max_loras=8,  # League training: up to 7 opponents + 1 hero  # pyright: ignore[reportCallIssue]
            max_lora_rank=16,  # Must match training LoRA rank  # pyright: ignore[reportCallIssue]
            gpu_memory_utilization=gpu_memory_util,  # pyright: ignore[reportCallIssue]
            max_num_seqs=max_num_seqs,  # pyright: ignore[reportCallIssue]
            max_num_batched_tokens=max_num_batched_tokens,  # Critical for throughput  # pyright: ignore[reportCallIssue]
            enable_prefix_caching=True,  # Prefix caching already optimized for shared prompt prefixes  # pyright: ignore[reportCallIssue]
            disable_log_stats=False,  # pyright: ignore[reportCallIssue]
            logits_processors=[DiplomacyLogitsProcessor],  # pyright: ignore[reportCallIssue]
            # Optimization: Use cached model location
            download_dir=str(HF_CACHE_PATH / "hub"),  # pyright: ignore[reportCallIssue]
            # Optimization: Prefer safetensors (faster loading than pickle)
            load_format="auto",  # Auto-detects best format (safetensors preferred)  # pyright: ignore[reportCallIssue]
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

        # Try to access stat loggers for metrics (vLLM v1)
        self._stat_logger_available = False
        try:
            if self.engine.logger_manager:
                self._stat_logger_available = True
                print("✅ vLLM stat_loggers available for metrics")
            else:
                print("⚠️ vLLM stat_loggers not available - using fallback metrics")
        except Exception as e:
            print(f"⚠️ Could not check stat_loggers: {e}")

        print("✅ Engine Ready (with dynamic LoRA support).")

    def _get_prefix_cache_stats(self) -> dict | None:
        """
        Extract prefix cache stats from vLLM's Prometheus metrics.

        vLLM v1 exposes prefix cache metrics as Prometheus counters:
        - vllm:prefix_cache_queries (Counter)
        - vllm:prefix_cache_hits (Counter)

        See: https://docs.vllm.ai/en/latest/design/metrics/#prefix-cache-hit-rate

        Returns dict with hit_rate, queries, hits if available, else None.
        """
        try:
            # vLLM uses prometheus_client - query the registry directly
            from prometheus_client import REGISTRY

            queries = 0
            hits = 0

            # Look for vLLM prefix cache counters in the registry
            for metric in REGISTRY.collect():
                # vLLM v1 uses "vllm:prefix_cache_queries" and "vllm:prefix_cache_hits"
                # prometheus_client may normalize colons to underscores
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
            # prometheus_client not available
            pass
        except Exception as e:
            # Don't fail generation for stats errors
            print(f"⚠️ Could not get prefix cache stats: {e}")

        return None

    @modal.method()
    def get_cache_stats(self) -> dict:
        """Get current prefix cache statistics."""
        stats = self._get_prefix_cache_stats()
        return {
            "vllm_stats": stats,
            "cumulative": self._cache_stats.copy(),
        }

    @modal.method()
    async def generate(
        self,
        prompts: list[str],
        valid_moves: list[dict],
        lora_name: str | None = None,
        temperature: float = 0.8,
        max_new_tokens: int = 256,
    ) -> list[GenerateBatchResponseItem]:
        import os

        """
        Generate responses for the given prompts.

        Args:
            prompts: List of prompt strings
            valid_moves: List of valid moves dicts for each prompt
            lora_name: Name of the LoRA adapter (relative to VLLM_LORA_RESOLVER_CACHE_DIR).
                       e.g., "benchmark-20251130/adapter_v1" will load from
                       /data/models/benchmark-20251130/adapter_v1
            temperature: Sampling temperature (default 0.8)
            max_new_tokens: Maximum tokens to generate (default 256)
        """

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

            async def _generate_single(prompt: str, moves: dict) -> GenerationResponse:
                """Generate for a single prompt. Allows concurrent execution."""
                request_id = str(uuid.uuid4())
                # vLLM SamplingParams accepts these at runtime but type stubs may be incomplete
                # logprobs=1 returns per-token log probabilities for the sampled token
                sampling_params = SamplingParams(  # type: ignore[call-arg, misc]
                    temperature=temperature,  # type: ignore[arg-type]
                    max_tokens=max_new_tokens,  # type: ignore[arg-type]
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

            generation_start = time.time()
            responses = await asyncio.gather(
                *[_generate_single(p, m) for p, m in zip(prompts, valid_moves, strict=True)]
            )
            timing_breakdown["generation_time_s"] = time.time() - generation_start

            duration_ms = int((time.time() - request_start) * 1000)
            timing_breakdown["total_time_s"] = duration_ms / 1000.0

            total_tokens = sum(resp["token_count"] for resp in responses)
            tokens_per_second = total_tokens / (duration_ms / 1000) if duration_ms > 0 else None

            # Calculate prompt tokens for cache tracking
            total_prompt_tokens = sum(len(resp["prompt_token_ids"]) for resp in responses)

            # Get prefix cache stats from vLLM (may return None if API not accessible)
            cache_stats = self._get_prefix_cache_stats()
            cache_hit_rate = cache_stats["hit_rate"] if cache_stats else None
            cache_queries = cache_stats.get("queries") if cache_stats else None
            cache_hits = cache_stats.get("hits") if cache_stats else None

            # Update cumulative stats - always track what we can measure directly
            self._cache_stats["batches_processed"] += 1
            self._cache_stats["total_prompt_tokens"] += total_prompt_tokens
            self._cache_stats["batch_size_total"] = (
                self._cache_stats.get("batch_size_total", 0) + batch_size
            )

            # If we got real stats from vLLM, track them
            if cache_queries is not None:
                self._cache_stats["total_queries"] += cache_queries
            else:
                # Fallback: count requests as queries
                self._cache_stats["total_queries"] += batch_size

            if cache_hits is not None:
                self._cache_stats["total_hits"] += cache_hits
            elif cache_hit_rate is not None:
                # Estimate hits from hit rate
                self._cache_stats["total_hits"] += int(batch_size * cache_hit_rate)

            # Track if we're getting real stats or estimates
            if cache_stats is not None:
                self._cache_stats["real_stats_available"] = True
            else:
                self._cache_stats.setdefault("real_stats_available", False)

            # Enhanced logging with timing breakdown and cache stats
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

            # Log detailed timing breakdown to Axiom for analysis
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
                    "prompt_tokens": total_prompt_tokens,
                    "prefix_cache_hit_rate": cache_hit_rate,
                }
            )

            # Log cache stats if we got them and hit rate is notable
            if cache_stats and batch_size >= 4:  # Only log for meaningful batches
                log_prefix_cache_stats(
                    batch_id=str(uuid.uuid4())[:8],
                    hit_rate=cache_hit_rate or 0.0,
                    queries=cache_queries or 0,
                    hits=cache_hits or 0,
                    prompt_tokens_total=total_prompt_tokens,
                    # Estimate cached tokens from hit rate
                    prompt_tokens_cached=int(total_prompt_tokens * (cache_hit_rate or 0.0)),
                )

            # Flush axiom events in a background thread so we don't block the main thread
            asyncio.create_task(axiom.flush())

            # Return rich response structure with token data for trainer optimization
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
