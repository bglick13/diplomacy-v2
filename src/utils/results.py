"""
Result types for errors-as-values pattern in distributed ML training.

These types are designed to cross Modal process boundaries safely
using cloudpickle serialization via to_dict()/from_dict() methods.

Usage:
    # In rollouts - return typed results
    if success:
        return RolloutSuccess(...).to_dict()
    else:
        return RolloutFailure(
            error_code=ErrorCode.ROLLOUT_INFERENCE_TIMEOUT,
            message="Inference timed out",
            fix_instruction="Check InferenceEngine health",
        ).to_dict()

    # In trainer - parse results
    result = parse_rollout_result(raw_dict)
    if isinstance(result, RolloutFailure):
        logger.error(f"[{result.error_code}] {result.message}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ErrorCode(str, Enum):
    """Categorized error codes for actionable diagnostics.

    Naming convention: {CATEGORY}_{SPECIFIC_ERROR}
    Categories: CONFIG, ROLLOUT, INFERENCE
    """

    # Configuration errors (caught at preflight)
    CONFIG_LORA_RANK_MISMATCH = "CONFIG_LORA_RANK_MISMATCH"
    CONFIG_ADAPTER_NOT_FOUND = "CONFIG_ADAPTER_NOT_FOUND"
    CONFIG_MODEL_NOT_FOUND = "CONFIG_MODEL_NOT_FOUND"
    CONFIG_VOLUME_NOT_ACCESSIBLE = "CONFIG_VOLUME_NOT_ACCESSIBLE"
    CONFIG_INVALID_VALUE = "CONFIG_INVALID_VALUE"

    # Runtime errors (rollout)
    ROLLOUT_INFERENCE_TIMEOUT = "ROLLOUT_INFERENCE_TIMEOUT"
    ROLLOUT_ADAPTER_LOAD_FAILED = "ROLLOUT_ADAPTER_LOAD_FAILED"
    ROLLOUT_GAME_ENGINE_ERROR = "ROLLOUT_GAME_ENGINE_ERROR"
    ROLLOUT_WARMUP_FAILED = "ROLLOUT_WARMUP_FAILED"

    # Runtime errors (inference)
    INFERENCE_VLLM_ENGINE_DEAD = "INFERENCE_VLLM_ENGINE_DEAD"
    INFERENCE_CUDA_OOM = "INFERENCE_CUDA_OOM"
    INFERENCE_GENERATION_FAILED = "INFERENCE_GENERATION_FAILED"

    # Unknown/fallback
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class ValidationError:
    """A single validation error with actionable fix instructions.

    All validation errors MUST include a fix_instruction that tells the user
    exactly how to resolve the issue.
    """

    code: ErrorCode
    message: str
    fix_instruction: str
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "fix_instruction": self.fix_instruction,
            "context": self.context,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ValidationError:
        return cls(
            code=ErrorCode(data["code"]),
            message=data["message"],
            fix_instruction=data["fix_instruction"],
            context=data.get("context", {}),
        )

    def __str__(self) -> str:
        return f"[{self.code.value}] {self.message}\n  Fix: {self.fix_instruction}"


class PreflightError(Exception):
    """Raised when preflight validation fails. Contains structured errors."""

    def __init__(self, message: str, errors: tuple[ValidationError, ...]):
        super().__init__(message)
        self.errors = errors


@dataclass(frozen=True)
class PreflightResult:
    """Result of preflight validation - must pass before any GPU work.

    Usage:
        result = run_full_preflight(cfg)
        for warning in result.warnings:
            logger.warning(str(warning))
        result.raise_if_failed()  # Hard error if any failures
    """

    success: bool
    errors: tuple[ValidationError, ...] = field(default_factory=tuple)
    warnings: tuple[ValidationError, ...] = field(default_factory=tuple)

    def raise_if_failed(self) -> None:
        """Raise PreflightError if validation failed."""
        if not self.success:
            error_msgs = "\n".join(str(e) for e in self.errors)
            raise PreflightError(
                f"Preflight validation failed with {len(self.errors)} error(s):\n{error_msgs}",
                errors=self.errors,
            )

    def log_warnings(self, logger) -> None:
        """Log all warnings using provided logger."""
        for warning in self.warnings:
            logger.warning(f"[PREFLIGHT WARNING] {warning}")

    @staticmethod
    def merge(*results: PreflightResult) -> PreflightResult:
        """Merge multiple preflight results into one."""
        all_errors: list[ValidationError] = []
        all_warnings: list[ValidationError] = []
        for r in results:
            all_errors.extend(r.errors)
            all_warnings.extend(r.warnings)
        return PreflightResult(
            success=len(all_errors) == 0,
            errors=tuple(all_errors),
            warnings=tuple(all_warnings),
        )


# =============================================================================
# Rollout Result Types
# =============================================================================


@dataclass(frozen=True)
class RolloutSuccess:
    """Successful rollout result with all trajectory data.

    This replaces the bare dict return from run_rollout() with a typed structure.
    """

    trajectories: list[dict[str, Any]]
    extraction_stats: dict[str, Any]
    game_stats: dict[str, Any]
    timing: dict[str, float]
    match_results: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Serialize for Modal transport."""
        return {
            "status": "success",
            "trajectories": self.trajectories,
            "extraction_stats": self.extraction_stats,
            "game_stats": self.game_stats,
            "timing": self.timing,
            "match_results": self.match_results,
        }


@dataclass(frozen=True)
class RolloutFailure:
    """Failed rollout with error context and actionable fix.

    All failures MUST include:
    - error_code: For categorization and dashboards
    - message: Human-readable description
    - fix_instruction: Actionable steps to resolve
    """

    error_code: ErrorCode
    message: str
    fix_instruction: str
    partial_data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for Modal transport."""
        return {
            "status": "failure",
            "error_code": self.error_code.value,
            "message": self.message,
            "fix_instruction": self.fix_instruction,
            "partial_data": self.partial_data,
        }

    def __str__(self) -> str:
        return f"[{self.error_code.value}] {self.message}\n  Fix: {self.fix_instruction}"


# Union type for rollout results
RolloutResult = RolloutSuccess | RolloutFailure


def parse_rollout_result(data: dict[str, Any]) -> RolloutResult:
    """Parse a dict from Modal serialization back into typed result.

    Args:
        data: Raw dict returned from rollout worker

    Returns:
        RolloutSuccess or RolloutFailure based on status field
    """
    if data.get("status") == "failure":
        # Handle unknown error codes gracefully by falling back to UNKNOWN
        error_code_str = data.get("error_code", "UNKNOWN")
        try:
            error_code = ErrorCode(error_code_str)
        except ValueError:
            error_code = ErrorCode.UNKNOWN

        return RolloutFailure(
            error_code=error_code,
            message=data.get("message", "Unknown error"),
            fix_instruction=data.get("fix_instruction", "Check logs for details."),
            partial_data=data.get("partial_data", {}),
        )
    else:
        # Assume success if not explicitly failure (backwards compatibility)
        return RolloutSuccess(
            trajectories=data.get("trajectories", []),
            extraction_stats=data.get("extraction_stats", {}),
            game_stats=data.get("game_stats", {}),
            timing=data.get("timing", {}),
            match_results=data.get("match_results", []),
        )


def is_rollout_failure(data: dict[str, Any]) -> bool:
    """Quick check if a rollout result dict represents a failure.

    Useful for filtering before full parsing.
    """
    return data.get("status") == "failure"
