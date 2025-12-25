"""
Tests for preflight validation and result types.

This module tests:
- Preflight validation logic (lora_rank, adapter existence)
- Result types serialization (to_dict/from_dict)
- Error code categorization
"""

from unittest.mock import MagicMock, patch

import pytest

from src.utils.preflight import (
    INFERENCE_ENGINE_MAX_LORA_RANK,
    run_full_preflight,
    validate_adapter_exists,
    validate_lora_config,
)
from src.utils.results import (
    ErrorCode,
    PreflightError,
    PreflightResult,
    RolloutFailure,
    RolloutSuccess,
    ValidationError,
    is_rollout_failure,
    parse_rollout_result,
)


class TestErrorCode:
    """Tests for ErrorCode enum."""

    def test_error_codes_are_strings(self):
        """All error codes should be string enums."""
        for code in ErrorCode:
            assert isinstance(code.value, str)
            assert code.value == code.name

    def test_common_error_codes_exist(self):
        """Key error codes should be defined."""
        assert ErrorCode.CONFIG_LORA_RANK_MISMATCH
        assert ErrorCode.ROLLOUT_INFERENCE_TIMEOUT
        assert ErrorCode.UNKNOWN


class TestValidationError:
    """Tests for ValidationError dataclass."""

    def test_to_dict_serialization(self):
        """ValidationError should serialize to dict correctly."""
        error = ValidationError(
            code=ErrorCode.CONFIG_LORA_RANK_MISMATCH,
            message="Test error",
            fix_instruction="Fix this by doing X",
            context={"key": "value"},
        )

        data = error.to_dict()

        assert data["code"] == "CONFIG_LORA_RANK_MISMATCH"
        assert data["message"] == "Test error"
        assert data["fix_instruction"] == "Fix this by doing X"
        assert data["context"] == {"key": "value"}

    def test_from_dict_deserialization(self):
        """ValidationError should deserialize from dict correctly."""
        data = {
            "code": "CONFIG_LORA_RANK_MISMATCH",
            "message": "Test error",
            "fix_instruction": "Fix this",
            "context": {"key": "value"},
        }

        error = ValidationError.from_dict(data)

        assert error.code == ErrorCode.CONFIG_LORA_RANK_MISMATCH
        assert error.message == "Test error"
        assert error.fix_instruction == "Fix this"
        assert error.context == {"key": "value"}

    def test_str_representation(self):
        """ValidationError __str__ should include code and fix."""
        error = ValidationError(
            code=ErrorCode.CONFIG_LORA_RANK_MISMATCH,
            message="Test error",
            fix_instruction="Fix this",
        )

        error_str = str(error)

        assert "CONFIG_LORA_RANK_MISMATCH" in error_str
        assert "Test error" in error_str
        assert "Fix this" in error_str


class TestPreflightResult:
    """Tests for PreflightResult dataclass."""

    def test_success_result(self):
        """Successful result should not raise."""
        result = PreflightResult(success=True)
        result.raise_if_failed()  # Should not raise

    def test_failure_result_raises(self):
        """Failed result should raise PreflightError."""
        error = ValidationError(
            code=ErrorCode.CONFIG_LORA_RANK_MISMATCH,
            message="Test",
            fix_instruction="Fix",
        )
        result = PreflightResult(success=False, errors=(error,))

        with pytest.raises(PreflightError) as exc_info:
            result.raise_if_failed()

        assert len(exc_info.value.errors) == 1
        assert "CONFIG_LORA_RANK_MISMATCH" in str(exc_info.value)

    def test_merge_results(self):
        """Multiple results should merge correctly."""
        error1 = ValidationError(
            code=ErrorCode.CONFIG_LORA_RANK_MISMATCH,
            message="Error 1",
            fix_instruction="Fix 1",
        )
        error2 = ValidationError(
            code=ErrorCode.CONFIG_ADAPTER_NOT_FOUND,
            message="Error 2",
            fix_instruction="Fix 2",
        )
        warning = ValidationError(
            code=ErrorCode.CONFIG_INVALID_VALUE,
            message="Warning",
            fix_instruction="Check",
        )

        result1 = PreflightResult(success=False, errors=(error1,))
        result2 = PreflightResult(success=False, errors=(error2,))
        result3 = PreflightResult(success=True, warnings=(warning,))

        merged = PreflightResult.merge(result1, result2, result3)

        assert not merged.success
        assert len(merged.errors) == 2
        assert len(merged.warnings) == 1

    def test_merge_all_success(self):
        """Merging all successful results should be successful."""
        result1 = PreflightResult(success=True)
        result2 = PreflightResult(success=True)

        merged = PreflightResult.merge(result1, result2)

        assert merged.success
        assert len(merged.errors) == 0


class TestValidateLoraConfig:
    """Tests for validate_lora_config function."""

    def test_valid_lora_rank(self):
        """Valid lora_rank should pass."""
        config = MagicMock()
        config.lora_rank = 16
        config.lora_alpha = 32

        result = validate_lora_config(config)

        assert result.success
        assert len(result.errors) == 0

    def test_lora_rank_exceeds_max(self):
        """lora_rank exceeding max should fail."""
        config = MagicMock()
        config.lora_rank = INFERENCE_ENGINE_MAX_LORA_RANK + 1  # e.g., 33
        config.lora_alpha = 64

        result = validate_lora_config(config)

        assert not result.success
        assert len(result.errors) == 1
        assert result.errors[0].code == ErrorCode.CONFIG_LORA_RANK_MISMATCH

    def test_lora_rank_at_max(self):
        """lora_rank exactly at max should pass."""
        config = MagicMock()
        config.lora_rank = INFERENCE_ENGINE_MAX_LORA_RANK  # 32
        config.lora_alpha = 64

        result = validate_lora_config(config)

        assert result.success

    def test_high_lora_alpha_warning(self):
        """Unusually high lora_alpha should generate warning."""
        config = MagicMock()
        config.lora_rank = 8
        config.lora_alpha = 64  # 8x rank, > 4x threshold

        result = validate_lora_config(config)

        assert result.success  # Warning, not error
        assert len(result.warnings) == 1
        assert "lora_alpha" in result.warnings[0].message


class TestRolloutResultTypes:
    """Tests for RolloutSuccess and RolloutFailure."""

    def test_rollout_success_to_dict(self):
        """RolloutSuccess should serialize correctly."""
        success = RolloutSuccess(
            trajectories=[{"prompt": "test", "reward": 1.0}],
            extraction_stats={"rate": 0.95},
            game_stats={"sc_counts": [10]},
            timing={"total_s": 60.0},
            match_results=[{"game_id": "test"}],
        )

        data = success.to_dict()

        assert data["status"] == "success"
        assert len(data["trajectories"]) == 1
        assert data["extraction_stats"]["rate"] == 0.95

    def test_rollout_failure_to_dict(self):
        """RolloutFailure should serialize correctly."""
        failure = RolloutFailure(
            error_code=ErrorCode.ROLLOUT_INFERENCE_TIMEOUT,
            message="Inference timed out",
            fix_instruction="Check InferenceEngine health",
            partial_data={"rollout_id": "test123"},
        )

        data = failure.to_dict()

        assert data["status"] == "failure"
        assert data["error_code"] == "ROLLOUT_INFERENCE_TIMEOUT"
        assert data["message"] == "Inference timed out"

    def test_parse_rollout_success(self):
        """parse_rollout_result should parse success correctly."""
        data = {
            "status": "success",
            "trajectories": [{"prompt": "test"}],
            "extraction_stats": {},
            "game_stats": {},
            "timing": {},
            "match_results": [],
        }

        result = parse_rollout_result(data)

        assert isinstance(result, RolloutSuccess)
        assert len(result.trajectories) == 1

    def test_parse_rollout_failure(self):
        """parse_rollout_result should parse failure correctly."""
        data = {
            "status": "failure",
            "error_code": "ROLLOUT_INFERENCE_TIMEOUT",
            "message": "Timed out",
            "fix_instruction": "Check engine",
        }

        result = parse_rollout_result(data)

        assert isinstance(result, RolloutFailure)
        assert result.error_code == ErrorCode.ROLLOUT_INFERENCE_TIMEOUT

    def test_is_rollout_failure(self):
        """is_rollout_failure should correctly identify failures."""
        success_data = {"status": "success", "trajectories": []}
        failure_data = {"status": "failure", "error_code": "UNKNOWN"}

        assert not is_rollout_failure(success_data)
        assert is_rollout_failure(failure_data)

    def test_parse_unknown_error_code(self):
        """Unknown error codes should map to UNKNOWN."""
        data = {
            "status": "failure",
            "error_code": "NONEXISTENT_ERROR_CODE",
            "message": "Something",
        }

        # Should not raise, should map to UNKNOWN
        result = parse_rollout_result(data)

        assert isinstance(result, RolloutFailure)
        assert result.error_code == ErrorCode.UNKNOWN

    def test_rollout_failure_str(self):
        """RolloutFailure __str__ should be informative."""
        failure = RolloutFailure(
            error_code=ErrorCode.ROLLOUT_INFERENCE_TIMEOUT,
            message="Timed out",
            fix_instruction="Fix it",
        )

        failure_str = str(failure)

        assert "ROLLOUT_INFERENCE_TIMEOUT" in failure_str
        assert "Timed out" in failure_str


class TestValidateAdapterExists:
    """Tests for validate_adapter_exists function."""

    def test_none_adapter_passes(self):
        """None adapter path should pass (no adapter needed)."""
        result = validate_adapter_exists(None)
        assert result.success

    @patch("src.utils.preflight.MODELS_VOLUME_PATH")
    def test_missing_adapter_fails(self, mock_path):
        """Missing adapter should fail."""
        mock_path.__truediv__ = MagicMock(return_value=MagicMock(exists=lambda: False))

        result = validate_adapter_exists("nonexistent/adapter")

        assert not result.success
        assert result.errors[0].code == ErrorCode.CONFIG_ADAPTER_NOT_FOUND


class TestRunFullPreflight:
    """Tests for run_full_preflight function."""

    def test_valid_config_passes(self):
        """Valid configuration should pass preflight."""
        config = MagicMock()
        config.lora_rank = 16
        config.lora_alpha = 32
        config.resume_from_run = None

        # Not in Modal environment, so volume checks are skipped
        with patch.dict("os.environ", {}, clear=True):
            result = run_full_preflight(config)

        assert result.success

    def test_invalid_lora_rank_fails(self):
        """Invalid lora_rank should fail preflight."""
        config = MagicMock()
        config.lora_rank = 64  # Exceeds max of 32
        config.lora_alpha = 128
        config.resume_from_run = None

        with patch.dict("os.environ", {}, clear=True):
            result = run_full_preflight(config)

        assert not result.success
        assert any(e.code == ErrorCode.CONFIG_LORA_RANK_MISMATCH for e in result.errors)
