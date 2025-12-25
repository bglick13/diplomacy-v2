"""
Preflight validation that runs BEFORE any GPU work.

This module validates configuration consistency across components
to catch mismatches that would otherwise cause silent timeouts or failures.

Usage:
    from src.utils.preflight import run_full_preflight

    preflight = run_full_preflight(cfg)
    preflight.log_warnings(logger)
    preflight.raise_if_failed()  # Hard error if validation fails

The key insight is that many training failures are caused by configuration
mismatches between components (trainer, rollouts, inference engine). By
validating these BEFORE any GPU work, we fail fast with actionable errors.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from src.utils.results import ErrorCode, PreflightResult, ValidationError

if TYPE_CHECKING:
    from src.utils.config import ExperimentConfig

# =============================================================================
# Infrastructure Constants
# =============================================================================
# These MUST match the values in src/apps/inference_engine/app.py.
# The inference engine also validates at runtime (see AdapterManager._validate_adapter_config)
# but we validate here at preflight to fail-fast before any GPU work.

# Must match INFERENCE_ENGINE_MAX_LORA_RANK in src/apps/inference_engine/app.py:39
INFERENCE_ENGINE_MAX_LORA_RANK = 32

# Must match max_loras in src/apps/inference_engine/app.py:533
INFERENCE_ENGINE_MAX_LORAS = 8

# Standard Modal volume path
MODELS_VOLUME_PATH = Path("/data/models")


# =============================================================================
# Config Consistency Validation
# =============================================================================


def validate_lora_config(config: ExperimentConfig) -> PreflightResult:
    """
    Validate LoRA configuration is consistent with inference engine limits.

    This catches the bug where lora_rank exceeds max_lora_rank, causing
    silent inference timeouts.
    """
    errors: list[ValidationError] = []
    warnings: list[ValidationError] = []

    # Check lora_rank vs max_lora_rank
    if config.lora_rank > INFERENCE_ENGINE_MAX_LORA_RANK:
        errors.append(
            ValidationError(
                code=ErrorCode.CONFIG_LORA_RANK_MISMATCH,
                message=(
                    f"lora_rank={config.lora_rank} exceeds InferenceEngine "
                    f"max_lora_rank={INFERENCE_ENGINE_MAX_LORA_RANK}"
                ),
                fix_instruction=(
                    f"Either:\n"
                    f"  1. Set lora_rank <= {INFERENCE_ENGINE_MAX_LORA_RANK} in your config, OR\n"
                    f"  2. Update max_lora_rank={config.lora_rank} in "
                    f"src/apps/inference_engine/app.py:490 and redeploy"
                ),
                context={
                    "config_lora_rank": config.lora_rank,
                    "max_lora_rank": INFERENCE_ENGINE_MAX_LORA_RANK,
                },
            )
        )

    # Check lora_alpha sanity (warning only)
    effective_alpha = config.lora_alpha or (config.lora_rank * 2)
    if effective_alpha > config.lora_rank * 4:
        warnings.append(
            ValidationError(
                code=ErrorCode.CONFIG_INVALID_VALUE,
                message=(
                    f"lora_alpha={effective_alpha} is unusually high "
                    f"(> 4x lora_rank={config.lora_rank})"
                ),
                fix_instruction=(
                    "Typical lora_alpha is 1-2x lora_rank. "
                    "High values may cause training instability. "
                    "Consider setting lora_alpha explicitly."
                ),
                context={"lora_alpha": effective_alpha, "lora_rank": config.lora_rank},
            )
        )

    return PreflightResult(
        success=len(errors) == 0,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def validate_adapter_exists(adapter_path: str | None) -> PreflightResult:
    """
    Validate that a LoRA adapter exists on the volume.

    This catches the case where training is resumed but the checkpoint
    doesn't exist, preventing silent degradation to base model.
    """
    if adapter_path is None:
        return PreflightResult(success=True)

    errors: list[ValidationError] = []
    full_path = MODELS_VOLUME_PATH / adapter_path

    if not full_path.exists():
        errors.append(
            ValidationError(
                code=ErrorCode.CONFIG_ADAPTER_NOT_FOUND,
                message=f"LoRA adapter not found at {full_path}",
                fix_instruction=(
                    f"Ensure the adapter exists:\n"
                    f"  1. Check if training saved the adapter: ls -la {full_path}\n"
                    f"  2. If resuming, verify resume_from_run points to existing run\n"
                    f"  3. Volume may need reload - check Modal dashboard"
                ),
                context={"adapter_path": adapter_path, "full_path": str(full_path)},
            )
        )
    else:
        # Check for required adapter files
        config_file = full_path / "adapter_config.json"
        if not config_file.exists():
            errors.append(
                ValidationError(
                    code=ErrorCode.CONFIG_ADAPTER_NOT_FOUND,
                    message=f"Adapter directory exists but missing adapter_config.json at {full_path}",
                    fix_instruction=(
                        f"The adapter at {full_path} may be corrupted or incomplete.\n"
                        "Check if save_pretrained() completed successfully during training."
                    ),
                    context={"adapter_path": adapter_path, "missing_file": "adapter_config.json"},
                )
            )

        # Check for model weights
        weights_file = full_path / "adapter_model.safetensors"
        weights_bin = full_path / "adapter_model.bin"
        if not weights_file.exists() and not weights_bin.exists():
            errors.append(
                ValidationError(
                    code=ErrorCode.CONFIG_ADAPTER_NOT_FOUND,
                    message=f"Adapter directory missing weight files at {full_path}",
                    fix_instruction=(
                        "Expected adapter_model.safetensors or adapter_model.bin.\n"
                        "The adapter may not have been saved correctly."
                    ),
                    context={"adapter_path": adapter_path},
                )
            )

    return PreflightResult(success=len(errors) == 0, errors=tuple(errors))


def validate_volume_accessible() -> PreflightResult:
    """
    Validate that the Modal volume is accessible.

    This catches permission issues or volume mount failures early.
    """
    errors: list[ValidationError] = []

    if not MODELS_VOLUME_PATH.exists():
        errors.append(
            ValidationError(
                code=ErrorCode.CONFIG_VOLUME_NOT_ACCESSIBLE,
                message=f"Models volume not mounted at {MODELS_VOLUME_PATH}",
                fix_instruction=(
                    "The Modal volume is not accessible. Check:\n"
                    "  1. Volume is attached in Modal app configuration\n"
                    "  2. Container has correct volume mount path\n"
                    "  3. Volume exists in Modal dashboard"
                ),
                context={"path": str(MODELS_VOLUME_PATH)},
            )
        )

    return PreflightResult(success=len(errors) == 0, errors=tuple(errors))


# =============================================================================
# Main Preflight Entry Point
# =============================================================================


def run_full_preflight(config: ExperimentConfig) -> PreflightResult:
    """
    Run all preflight validations before training starts.

    This is the main entry point called by train_grpo() before any GPU work.
    All validations are run and errors are aggregated for a complete report.

    Args:
        config: The experiment configuration to validate

    Returns:
        PreflightResult with all errors and warnings

    Usage:
        preflight = run_full_preflight(cfg)
        preflight.log_warnings(logger)
        preflight.raise_if_failed()  # Raises PreflightError if any errors
    """
    results: list[PreflightResult] = []

    # 1. LoRA configuration consistency
    results.append(validate_lora_config(config))

    # 2. Volume accessibility (only if running in Modal context)
    # Skip if running locally (path won't exist)
    if os.environ.get("MODAL_ENVIRONMENT"):
        results.append(validate_volume_accessible())

        # 3. Resume adapter exists (if resuming)
        if config.resume_from_run:
            step = config.resume_from_step or "latest"
            # Handle "latest" by checking for any adapter
            if step == "latest":
                adapter_path = config.resume_from_run
            else:
                adapter_path = f"{config.resume_from_run}/adapter_v{step}"
            results.append(validate_adapter_exists(adapter_path))

    return PreflightResult.merge(*results)


def run_rollout_preflight(
    adapter_paths: list[str],
    hero_adapter: str | None = None,
) -> PreflightResult:
    """
    Run preflight checks specific to rollout workers.

    Called at the start of each rollout to validate adapters exist.

    Args:
        adapter_paths: List of adapter paths that will be loaded
        hero_adapter: The hero adapter path (if any)

    Returns:
        PreflightResult with adapter existence checks
    """
    results: list[PreflightResult] = []

    for path in adapter_paths:
        if path:  # Skip None entries
            results.append(validate_adapter_exists(path))

    if hero_adapter:
        results.append(validate_adapter_exists(hero_adapter))

    if not results:
        return PreflightResult(success=True)

    return PreflightResult.merge(*results)
