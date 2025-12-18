"""
Observability Layer for Diplomacy GRPO

Provides structured logging and metrics for:
- Rollout health monitoring
- LLM inference quality
- Error detection and alerting

Events are logged to both console and Axiom for dashboarding.
"""

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import aiohttp

# Configure logging to show up clearly in Modal logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("diplomacy")
console_logger = logging.getLogger("diplomacy")


# =============================================================================
# Event Types
# =============================================================================


class EventType(str, Enum):
    """Standard event types for Axiom ingestion."""

    # Timing/Performance
    SPAN_DURATION = "span_duration"

    # Rollout Events
    ROLLOUT_START = "rollout_start"
    ROLLOUT_COMPLETE = "rollout_complete"
    ROLLOUT_ERROR = "rollout_error"

    # Inference Events
    INFERENCE_REQUEST = "inference_request"
    INFERENCE_RESPONSE = "inference_response"

    # Order Parsing Events
    ORDERS_EXTRACTED = "orders_extracted"
    ORDERS_EMPTY = "orders_empty"  # CRITICAL: No orders parsed
    ORDERS_PARTIAL = "orders_partial"  # Less orders than units
    ORDERS_INVALID = "orders_invalid"  # Orders rejected by engine

    # Game State Events
    GAME_PHASE = "game_phase"
    GAME_ELIMINATED = "game_eliminated"

    # Health Checks
    HEALTH_CHECK = "health_check"

    # Prefix Cache Events
    PREFIX_CACHE_STATS = "prefix_cache_stats"
    PREFIX_CACHE_BATCH = "prefix_cache_batch"

    # Training Events
    TRAINING_STEP = "training_step"
    TRAINING_START = "training_start"
    TRAINING_COMPLETE = "training_complete"
    TRAINING_ERROR = "training_error"
    CHECKPOINT_SAVED = "checkpoint_saved"
    TRAJECTORY_PROCESSING = "trajectory_processing"
    GPU_STATS = "gpu_stats"


# =============================================================================
# Axiom Handler
# =============================================================================


class AxiomHandler:
    """Lightweight async logger for Axiom."""

    def __init__(self):
        self.token = os.environ.get("AXIOM_TOKEN")
        self.dataset = os.environ.get("AXIOM_DATASET", "diplomacy")
        self.batch: list[dict] = []
        self._service = os.environ.get("MODAL_FUNCTION_NAME", "unknown")

    def log(self, event: dict):
        """Add event to batch for later flushing."""
        if not self.token:
            return

        # Enrich with metadata
        event.update(
            {
                "_time": datetime.utcnow().isoformat() + "Z",
                "service": self._service,
            }
        )
        self.batch.append(event)

    async def flush(self):
        """Flush all batched events to Axiom."""
        if not self.token or not self.batch:
            console_logger.debug("Axiom batch is empty, skipping flush")
            return

        console_logger.info(f"📊 Flushing {len(self.batch)} events to Axiom")

        payload = list(self.batch)
        self.batch = []  # Clear immediately

        url = f"https://api.axiom.co/v1/datasets/{self.dataset}/ingest"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as resp:
                    if resp.status != 200:
                        console_logger.error(f"Axiom Flush Failed: {resp.status}")
                    else:
                        console_logger.debug(f"Axiom Flush Success: {len(payload)} events")
        except Exception as e:
            console_logger.error(f"Axiom Error: {e}")


# Global singleton
axiom = AxiomHandler()


# =============================================================================
# Stopwatch Context Manager
# =============================================================================


class stopwatch:
    """Context manager to measure and log execution time of blocks."""

    def __init__(self, name: str, metadata: dict | None = None):
        self.name = name
        self.start_time = 0.0
        self.duration = 0.0
        self.metadata = metadata or {}

    def __enter__(self):
        self.start_time = time.time()
        logger.info(f"⏳ [START] {self.name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.duration = time.time() - self.start_time
        status = "error" if exc_type else "success"

        # Console Log
        if exc_type:
            console_logger.error(f"❌ [FAIL] {self.name} ({self.duration:.2f}s)")
        else:
            console_logger.info(f"✅ [DONE] {self.name} ({self.duration:.2f}s)")

        # Axiom Log
        axiom.log(
            {
                "event": EventType.SPAN_DURATION,
                "span_name": self.name,
                "duration_ms": int(self.duration * 1000),
                "status": status,
                "error": str(exc_val) if exc_val else None,
                **self.metadata,
            }
        )


class tracked_timing:
    """
    Context manager to measure time and add it to a TimingBreakdown.

    Usage:
        timing = TimingBreakdown()
        with tracked_timing(timing, "inference"):
            # do inference work
            pass
        # timing.inference_s is now updated
    """

    def __init__(
        self,
        timing: "TimingBreakdown",
        category: str,
        log_name: str | None = None,
        silent: bool = False,
    ):
        self.timing = timing
        self.category = category
        self.log_name = log_name or category
        self.silent = silent
        self.start_time = 0.0
        self.duration = 0.0

    def __enter__(self):
        self.start_time = time.time()
        if not self.silent:
            logger.debug(f"⏱️ [START] {self.log_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.duration = time.time() - self.start_time
        self.timing.add(self.category, self.duration)
        if not self.silent:
            logger.debug(f"⏱️ [DONE] {self.log_name} ({self.duration:.3f}s)")


class GPUStatsLogger:
    """Background sampler that logs GPU utilization to Axiom."""

    def __init__(self, interval_s: float = 5.0, device_index: int = 0):
        self.interval_s = interval_s
        self.device_index = device_index
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._context = ""

    def start(self, context: str):
        if self._thread and self._thread.is_alive():
            return
        self._context = context
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1)

    def _run(self):
        try:
            import pynvml  # type: ignore[import-untyped]

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            while not self._stop_event.wait(self.interval_s):
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    axiom.log(
                        {
                            "event": EventType.GPU_STATS,
                            "context": self._context,
                            "gpu_utilization": util.gpu,
                            "memory_utilization": util.memory,
                            "memory_used": mem_info.used,
                            "memory_total": mem_info.total,
                        }
                    )
                except Exception as stats_error:
                    console_logger.debug(f"GPU stats logging error: {stats_error}")
        except Exception as e:
            console_logger.debug(f"Unable to initialize GPU stats logger: {e}")
        finally:
            try:
                import pynvml  # type: ignore[import-untyped]

                pynvml.nvmlShutdown()
            except Exception:
                pass


# =============================================================================
# Rollout Events
# =============================================================================


def log_rollout_start(
    rollout_id: str,
    warmup_phases: int,
    samples_per_group: int,
    horizon_years: int,
):
    """Log the start of a rollout."""
    console_logger.info(
        f"🚀 Rollout {rollout_id}: warmup={warmup_phases}, samples={samples_per_group}, horizon={horizon_years}"
    )
    axiom.log(
        {
            "event": EventType.ROLLOUT_START,
            "rollout_id": rollout_id,
            "warmup_phases": warmup_phases,
            "samples_per_group": samples_per_group,
            "horizon_years": horizon_years,
        }
    )


def log_rollout_complete(
    rollout_id: str,
    trajectories_count: int,
    total_inference_calls: int,
    total_duration_ms: int,
):
    """Log successful rollout completion."""
    console_logger.info(
        f"✅ Rollout {rollout_id} complete: {trajectories_count} trajectories, {total_inference_calls} inferences"
    )
    axiom.log(
        {
            "event": EventType.ROLLOUT_COMPLETE,
            "rollout_id": rollout_id,
            "trajectories_count": trajectories_count,
            "total_inference_calls": total_inference_calls,
            "total_duration_ms": total_duration_ms,
        }
    )


def log_rollout_error(rollout_id: str, error: str, phase: str = ""):
    """Log rollout error."""
    console_logger.error(f"❌ Rollout {rollout_id} error: {error}")
    axiom.log(
        {
            "event": EventType.ROLLOUT_ERROR,
            "rollout_id": rollout_id,
            "error": error,
            "phase": phase,
        }
    )


# =============================================================================
# Inference Events
# =============================================================================


def log_inference_request(
    rollout_id: str,
    batch_size: int,
    phase: str,
    step_type: str,  # "warmup" or "rollout"
):
    """Log an inference request being sent."""
    axiom.log(
        {
            "event": EventType.INFERENCE_REQUEST,
            "rollout_id": rollout_id,
            "batch_size": batch_size,
            "phase": phase,
            "step_type": step_type,
        }
    )


def log_inference_response(
    rollout_id: str,
    batch_size: int,
    duration_ms: int,
    tokens_generated: int | None = None,
    tokens_per_second: float | None = None,
    # Prefix cache stats
    prefix_cache_hit_rate: float | None = None,
    prefix_cache_queries: int | None = None,
    prefix_cache_hits: int | None = None,
):
    """Log inference response received."""
    event_data = {
        "event": EventType.INFERENCE_RESPONSE,
        "rollout_id": rollout_id,
        "batch_size": batch_size,
        "duration_ms": duration_ms,
        "tokens_generated": tokens_generated,
        "tokens_per_second": tokens_per_second,
    }
    # Add cache stats if available
    if prefix_cache_hit_rate is not None:
        event_data["prefix_cache_hit_rate"] = prefix_cache_hit_rate
        event_data["prefix_cache_queries"] = prefix_cache_queries
        event_data["prefix_cache_hits"] = prefix_cache_hits
    axiom.log(event_data)


def log_prefix_cache_stats(
    batch_id: str,
    hit_rate: float,
    queries: int,
    hits: int,
    prompt_tokens_total: int,
    prompt_tokens_cached: int,
    cache_utilization: float | None = None,
):
    """
    Log prefix cache performance for a batch.

    Args:
        batch_id: Identifier for the batch
        hit_rate: Fraction of queries that hit cache (0.0 - 1.0)
        queries: Total cache queries
        hits: Cache hits
        prompt_tokens_total: Total prompt tokens in batch
        prompt_tokens_cached: Prompt tokens served from cache
        cache_utilization: KV cache memory utilization (0.0 - 1.0)
    """
    cache_efficiency = (
        prompt_tokens_cached / prompt_tokens_total if prompt_tokens_total > 0 else 0.0
    )

    # Console log for visibility
    if hit_rate > 0.5:
        console_logger.info(
            f"🎯 Prefix Cache: {hit_rate * 100:.1f}% hit rate | "
            f"{prompt_tokens_cached}/{prompt_tokens_total} tokens cached ({cache_efficiency * 100:.1f}%)"
        )
    else:
        console_logger.warning(
            f"⚠️ Prefix Cache: {hit_rate * 100:.1f}% hit rate (low!) | "
            f"{prompt_tokens_cached}/{prompt_tokens_total} tokens cached"
        )

    axiom.log(
        {
            "event": EventType.PREFIX_CACHE_STATS,
            "batch_id": batch_id,
            "hit_rate": hit_rate,
            "queries": queries,
            "hits": hits,
            "prompt_tokens_total": prompt_tokens_total,
            "prompt_tokens_cached": prompt_tokens_cached,
            "cache_efficiency": cache_efficiency,
            "cache_utilization": cache_utilization,
        }
    )


def log_prefix_cache_batch_summary(
    step: int,
    batches_processed: int,
    avg_hit_rate: float,
    total_prompt_tokens: int,
    total_cached_tokens: int,
):
    """
    Log aggregated prefix cache stats for a training step.

    This is logged once per step to WandB for easy visualization.
    """
    cache_efficiency = total_cached_tokens / total_prompt_tokens if total_prompt_tokens > 0 else 0.0

    console_logger.info(
        f"📊 Step {step} Cache Summary: "
        f"avg_hit_rate={avg_hit_rate * 100:.1f}%, "
        f"cache_efficiency={cache_efficiency * 100:.1f}%, "
        f"tokens_saved={total_cached_tokens:,}"
    )

    axiom.log(
        {
            "event": EventType.PREFIX_CACHE_BATCH,
            "step": step,
            "batches_processed": batches_processed,
            "avg_hit_rate": avg_hit_rate,
            "total_prompt_tokens": total_prompt_tokens,
            "total_cached_tokens": total_cached_tokens,
            "cache_efficiency": cache_efficiency,
        }
    )


# =============================================================================
# Order Parsing Events - CRITICAL for debugging
# =============================================================================


def log_orders_extracted(
    rollout_id: str,
    power_name: str,
    orders_count: int,
    expected_count: int,
    raw_response_length: int,
    phase: str,
    raw_response: str = "",
):
    """Log successful order extraction."""
    status = "full" if orders_count == expected_count else "partial"

    # Truncate raw response for logging (first 200 chars)
    response_preview = raw_response[:200] if raw_response else ""

    if orders_count == 0 and expected_count > 0:
        # CRITICAL: This is the bug we want to catch!
        # Only warn if orders were expected but none were generated.
        # During ADJUSTMENTS phase, expected_count == 0 is legitimate when
        # a power has exactly as many units as supply centers (no builds/disbands needed).
        console_logger.warning(
            f"⚠️ ZERO ORDERS for {power_name} in {phase}! "
            f"Expected {expected_count}, got 0. "
            f"Response ({raw_response_length} chars): {response_preview!r}"
        )
        axiom.log(
            {
                "event": EventType.ORDERS_EMPTY,
                "rollout_id": rollout_id,
                "power_name": power_name,
                "expected_count": expected_count,
                "raw_response_length": raw_response_length,
                "raw_response_preview": response_preview,
                "phase": phase,
            }
        )
    elif orders_count == 0 and expected_count == 0:
        # Legitimate case: no orders needed (e.g., ADJUSTMENTS with units == supply centers)
        axiom.log(
            {
                "event": EventType.ORDERS_EXTRACTED,
                "rollout_id": rollout_id,
                "power_name": power_name,
                "orders_count": 0,
                "expected_count": 0,
                "status": "no_orders_needed",
                "phase": phase,
            }
        )
    elif orders_count < expected_count:
        console_logger.warning(
            f"⚠️ Partial orders for {power_name}: {orders_count}/{expected_count}"
        )
        axiom.log(
            {
                "event": EventType.ORDERS_PARTIAL,
                "rollout_id": rollout_id,
                "power_name": power_name,
                "orders_count": orders_count,
                "expected_count": expected_count,
                "raw_response_preview": response_preview,
                "phase": phase,
            }
        )
    else:
        axiom.log(
            {
                "event": EventType.ORDERS_EXTRACTED,
                "rollout_id": rollout_id,
                "power_name": power_name,
                "orders_count": orders_count,
                "expected_count": expected_count,
                "status": status,
                "phase": phase,
            }
        )


def log_orders_invalid(
    rollout_id: str,
    power_name: str,
    order: str,
    error: str,
    phase: str,
):
    """Log when an order is rejected by the game engine."""
    console_logger.warning(f"⚠️ Invalid order for {power_name}: {order} - {error}")
    axiom.log(
        {
            "event": EventType.ORDERS_INVALID,
            "rollout_id": rollout_id,
            "power_name": power_name,
            "order": order,
            "error": error,
            "phase": phase,
        }
    )


# =============================================================================
# Training Events
# =============================================================================


def log_training_start(
    run_name: str,
    total_steps: int,
    num_groups_per_step: int,
    samples_per_group: int,
    model_id: str,
    lora_rank: int,
    learning_rate: float,
):
    """Log the start of a training run."""
    console_logger.info(
        f"🚀 Training Started: {run_name} | {total_steps} steps | "
        f"{num_groups_per_step}x{samples_per_group} samples"
    )
    axiom.log(
        {
            "event": EventType.TRAINING_START,
            "run_name": run_name,
            "total_steps": total_steps,
            "num_groups_per_step": num_groups_per_step,
            "samples_per_group": samples_per_group,
            "model_id": model_id,
            "lora_rank": lora_rank,
            "learning_rate": learning_rate,
        }
    )


def log_training_step(
    step: int,
    loss: float,
    pg_loss: float,
    kl: float,
    grad_norm: float,
    learning_rate: float,
    # Reward stats
    reward_mean: float,
    reward_std: float,
    reward_min: float,
    reward_max: float,
    # Trajectory stats
    num_trajectories: int,
    num_groups: int,
    skipped_groups: int,
    # Loss component details
    mean_completion_logprob: float,
    mean_ref_logprob: float,
    mean_advantage: float,
    advantage_std: float,
    # Timing
    rollout_duration_ms: int,
    training_duration_ms: int,
    total_tokens: int,
):
    """Log a training step with comprehensive metrics."""
    console_logger.info(
        f"📈 Step {step}: loss={loss:.4f} | pg={pg_loss:.4f} | kl={kl:.4f} | "
        f"grad_norm={grad_norm:.4f} | reward_mean={reward_mean:.2f}"
    )
    axiom.log(
        {
            "event": EventType.TRAINING_STEP,
            "step": step,
            # Core losses
            "loss": loss,
            "pg_loss": pg_loss,
            "kl": kl,
            "grad_norm": grad_norm,
            "learning_rate": learning_rate,
            # Reward stats
            "reward_mean": reward_mean,
            "reward_std": reward_std,
            "reward_min": reward_min,
            "reward_max": reward_max,
            # Trajectory stats
            "num_trajectories": num_trajectories,
            "num_groups": num_groups,
            "skipped_groups": skipped_groups,
            # LogProb details
            "mean_completion_logprob": mean_completion_logprob,
            "mean_ref_logprob": mean_ref_logprob,
            "mean_advantage": mean_advantage,
            "advantage_std": advantage_std,
            # Timing & throughput
            "rollout_duration_ms": rollout_duration_ms,
            "training_duration_ms": training_duration_ms,
            "total_tokens": total_tokens,
            "tokens_per_second": (
                total_tokens / (training_duration_ms / 1000) if training_duration_ms > 0 else 0
            ),
        }
    )


def log_checkpoint_saved(step: int, adapter_path: str, run_name: str):
    """Log when a model checkpoint is saved."""
    console_logger.info(f"💾 Checkpoint saved: {adapter_path}")
    axiom.log(
        {
            "event": EventType.CHECKPOINT_SAVED,
            "step": step,
            "adapter_path": adapter_path,
            "run_name": run_name,
        }
    )


def log_trajectory_processing(
    step: int,
    total_trajectories: int,
    total_groups: int,
    skipped_single_sample_groups: int,
    reward_mean: float,
    reward_std: float,
    avg_completion_tokens: float,
):
    """Log trajectory processing results."""
    if skipped_single_sample_groups > 0:
        console_logger.warning(f"⚠️ Skipped {skipped_single_sample_groups} single-sample groups")
    axiom.log(
        {
            "event": EventType.TRAJECTORY_PROCESSING,
            "step": step,
            "total_trajectories": total_trajectories,
            "total_groups": total_groups,
            "skipped_single_sample_groups": skipped_single_sample_groups,
            "reward_mean": reward_mean,
            "reward_std": reward_std,
            "avg_completion_tokens": avg_completion_tokens,
        }
    )


def log_training_complete(run_name: str, total_steps: int, total_duration_ms: int):
    """Log successful training completion."""
    hours = total_duration_ms / (1000 * 60 * 60)
    console_logger.info(f"✅ Training Complete: {run_name} | {total_steps} steps | {hours:.2f}h")
    axiom.log(
        {
            "event": EventType.TRAINING_COMPLETE,
            "run_name": run_name,
            "total_steps": total_steps,
            "total_duration_ms": total_duration_ms,
        }
    )


def log_training_error(run_name: str, step: int, error: str):
    """Log training error."""
    console_logger.error(f"❌ Training Error at step {step}: {error}")
    axiom.log(
        {
            "event": EventType.TRAINING_ERROR,
            "run_name": run_name,
            "step": step,
            "error": error,
        }
    )


# =============================================================================
# Aggregate Metrics Helper
# =============================================================================


@dataclass
class TimingBreakdown:
    """
    Tracks detailed timing breakdown for rollouts.

    Categories are designed for waterfall chart visualization (mutually exclusive):
    - inference: Time spent calling the LLM inference engine
    - game_engine: Time spent processing moves in the diplomacy engine
    - prompt_building: Time spent building prompts for the LLM
    - ref_scoring: Time spent computing reference logprobs
    - visualization: Time spent capturing/saving game visualizations
    - forking: Time spent forking game state
    - other: Uncategorized time (should be minimal)
    """

    inference_s: float = 0.0
    game_engine_s: float = 0.0
    prompt_building_s: float = 0.0
    ref_scoring_s: float = 0.0
    visualization_s: float = 0.0
    forking_s: float = 0.0
    other_s: float = 0.0

    def add(self, category: str, duration: float):
        """Add duration to a category."""
        attr = f"{category}_s"
        if hasattr(self, attr):
            setattr(self, attr, getattr(self, attr) + duration)
        else:
            self.other_s += duration

    def to_dict(self) -> dict[str, float]:
        """Return as dict for JSON serialization."""
        return {
            "inference_s": self.inference_s,
            "game_engine_s": self.game_engine_s,
            "prompt_building_s": self.prompt_building_s,
            "ref_scoring_s": self.ref_scoring_s,
            "visualization_s": self.visualization_s,
            "forking_s": self.forking_s,
            "other_s": self.other_s,
        }

    def total(self) -> float:
        """Total tracked time."""
        return sum(self.to_dict().values())


@dataclass
class RolloutMetrics:
    """Accumulator for rollout-level metrics."""

    rollout_id: str
    start_time: float = field(default_factory=time.time)

    # Counters
    inference_calls: int = 0
    total_orders_expected: int = 0
    total_orders_extracted: int = 0
    total_orders_overflow: int = 0
    empty_responses: int = 0
    partial_responses: int = 0
    invalid_orders: int = 0

    # Detailed timing breakdown
    timing: TimingBreakdown = field(default_factory=TimingBreakdown)

    def record_extraction(self, extracted: int, expected: int):
        """Record order extraction result."""
        self.total_orders_expected += expected
        if extracted > expected:
            self.total_orders_overflow += extracted - expected
            extracted = expected
        self.total_orders_extracted += extracted
        if extracted == 0:
            self.empty_responses += 1
        elif extracted < expected:
            self.partial_responses += 1

    def record_invalid_order(self):
        """Record an invalid order."""
        self.invalid_orders += 1

    def get_extraction_rate(self) -> float:
        """Get percentage of expected orders that were extracted."""
        if self.total_orders_expected == 0:
            return 1.0
        return self.total_orders_extracted / self.total_orders_expected

    def get_summary(self) -> dict:
        """Get summary dict for logging."""
        duration_ms = int((time.time() - self.start_time) * 1000)
        return {
            "rollout_id": self.rollout_id,
            "duration_ms": duration_ms,
            "inference_calls": self.inference_calls,
            "extraction_rate": self.get_extraction_rate(),
            "empty_responses": self.empty_responses,
            "partial_responses": self.partial_responses,
            "invalid_orders": self.invalid_orders,
            "over_extracted": self.total_orders_overflow,
        }

    def log_summary(self):
        """Log the final summary to Axiom."""
        summary = self.get_summary()
        console_logger.info(f"📊 Rollout {self.rollout_id} metrics: {summary}")
        axiom.log({"event": "rollout_metrics", **summary})
