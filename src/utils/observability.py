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
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import aiohttp

# Configure logging to show up clearly in Modal logs
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
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
                        console_logger.debug(
                            f"Axiom Flush Success: {len(payload)} events"
                        )
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
        self.metadata = metadata or {}

    def __enter__(self):
        self.start_time = time.time()
        logger.info(f"⏳ [START] {self.name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        status = "error" if exc_type else "success"

        # Console Log
        if exc_type:
            console_logger.error(f"❌ [FAIL] {self.name} ({duration:.2f}s)")
        else:
            console_logger.info(f"✅ [DONE] {self.name} ({duration:.2f}s)")

        # Axiom Log
        axiom.log(
            {
                "event": EventType.SPAN_DURATION,
                "span_name": self.name,
                "duration_ms": int(duration * 1000),
                "status": status,
                "error": str(exc_val) if exc_val else None,
                **self.metadata,
            }
        )


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
):
    """Log inference response received."""
    axiom.log(
        {
            "event": EventType.INFERENCE_RESPONSE,
            "rollout_id": rollout_id,
            "batch_size": batch_size,
            "duration_ms": duration_ms,
            "tokens_generated": tokens_generated,
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

    if orders_count == 0:
        # CRITICAL: This is the bug we want to catch!
        console_logger.warning(
            f"⚠️ ZERO ORDERS for {power_name} in {phase}! "
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
# Aggregate Metrics Helper
# =============================================================================


@dataclass
class RolloutMetrics:
    """Accumulator for rollout-level metrics."""

    rollout_id: str
    start_time: float = field(default_factory=time.time)

    # Counters
    inference_calls: int = 0
    total_orders_expected: int = 0
    total_orders_extracted: int = 0
    empty_responses: int = 0
    partial_responses: int = 0
    invalid_orders: int = 0

    def record_extraction(self, extracted: int, expected: int):
        """Record order extraction result."""
        self.total_orders_expected += expected
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
        }

    def log_summary(self):
        """Log the final summary to Axiom."""
        summary = self.get_summary()
        console_logger.info(f"📊 Rollout {self.rollout_id} metrics: {summary}")
        axiom.log({"event": "rollout_metrics", **summary})
