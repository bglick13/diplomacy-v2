"""
Unit tests for the GRPO training module.

Tests cover:
- GRPOLoss computation and correctness
- process_trajectories edge cases and normalization
- TrajectoryStats aggregation
"""

import numpy as np
import pytest
import torch

# =============================================================================
# Test Fixtures
# =============================================================================


class TokenizerOutput:
    """Simple class to mimic HuggingFace tokenizer output with attribute access."""

    def __init__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        self.input_ids = input_ids
        self.attention_mask = attention_mask


class MockTokenizer:
    """Mock tokenizer for testing without loading real models."""

    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1

    def __call__(self, text: str, return_tensors: str = "pt", **kwargs) -> TokenizerOutput:
        """Simulate tokenization - return fixed length based on text length."""
        # Simple heuristic: ~4 chars per token
        num_tokens = max(1, len(text) // 4)
        if kwargs.get("max_length"):
            num_tokens = min(num_tokens, kwargs["max_length"])

        return TokenizerOutput(
            input_ids=torch.randint(2, self.vocab_size, (1, num_tokens)),
            attention_mask=torch.ones(1, num_tokens, dtype=torch.long),
        )

    def encode(self, text: str, add_special_tokens: bool = True, **kwargs) -> list[int]:
        """Simulate encoding."""
        num_tokens = max(1, len(text) // 4)
        if kwargs.get("max_length"):
            num_tokens = min(num_tokens, kwargs["max_length"])
        return list(range(2, 2 + num_tokens))


class ModelOutput:
    """Simple class to mimic HuggingFace model output with attribute access."""

    def __init__(self, logits: torch.Tensor):
        self.logits = logits


class MockPeftModel(torch.nn.Module):
    """Mock PeftModel that supports disable_adapter context manager."""

    def __init__(self, hidden_size: int = 64, vocab_size: int = 1000):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        # Use a linear layer so we have real parameters with gradients
        self.embed = torch.nn.Embedding(vocab_size, hidden_size)
        self.linear = torch.nn.Linear(hidden_size, vocab_size)
        self._adapter_enabled = True

    def forward(self, input_ids, attention_mask=None):
        """Return mock logits with proper gradient support."""
        # Create logits through actual computation so gradients flow
        embeddings = self.embed(input_ids)  # (batch, seq, hidden)
        logits = self.linear(embeddings)  # (batch, seq, vocab)

        # When adapter is disabled, add a small offset to simulate
        # reference model behavior (detached to not affect gradients)
        if not self._adapter_enabled:
            logits = logits + 0.1

        return ModelOutput(logits=logits)

    def disable_adapter(self):
        """Context manager to disable adapter."""
        return _DisableAdapterContext(self)


class _DisableAdapterContext:
    """Context manager for disabling adapter."""

    def __init__(self, model: MockPeftModel):
        self.model = model
        self.was_enabled = model._adapter_enabled

    def __enter__(self):
        self.model._adapter_enabled = False
        return self

    def __exit__(self, *args):
        self.model._adapter_enabled = self.was_enabled


@pytest.fixture
def mock_tokenizer():
    return MockTokenizer()


@pytest.fixture
def mock_model():
    return MockPeftModel()


@pytest.fixture
def sample_trajectories():
    """Create sample trajectory data for testing."""
    return [
        {
            "prompt": "You are France. The board state is...",
            "completion": "<analysis>I should move to Burgundy</analysis><orders>A PAR - BUR</orders>",
            "reward": 3.0,
            "group_id": "game1_FRANCE_1901",
        },
        {
            "prompt": "You are France. The board state is...",
            "completion": "<analysis>I should hold</analysis><orders>A PAR H</orders>",
            "reward": 2.0,
            "group_id": "game1_FRANCE_1901",
        },
        {
            "prompt": "You are France. The board state is...",
            "completion": "<analysis>Move to Picardy</analysis><orders>A PAR - PIC</orders>",
            "reward": 4.0,
            "group_id": "game1_FRANCE_1901",
        },
        {
            "prompt": "You are England. Different state...",
            "completion": "<analysis>Fleet to North Sea</analysis><orders>F LON - NTH</orders>",
            "reward": 5.0,
            "group_id": "game1_ENGLAND_1901",
        },
        {
            "prompt": "You are England. Different state...",
            "completion": "<analysis>Hold position</analysis><orders>F LON H</orders>",
            "reward": 1.0,
            "group_id": "game1_ENGLAND_1901",
        },
    ]


# =============================================================================
# Tests for process_trajectories
# =============================================================================


class TestProcessTrajectories:
    """Tests for the trajectory processing function."""

    def test_basic_processing(self, mock_tokenizer, sample_trajectories):
        """Test basic trajectory processing with valid data."""
        from src.training.trainer import process_trajectories

        batch, stats = process_trajectories(sample_trajectories, mock_tokenizer)

        # Should have processed all 5 trajectories (2 groups)
        assert stats.total_trajectories == 5
        assert stats.total_groups == 2
        assert stats.skipped_single_sample_groups == 0
        assert len(batch) == 5

        # Check batch structure
        for item in batch:
            assert "input_ids" in item
            assert "attention_mask" in item
            assert "labels" in item
            assert "advantages" in item
            assert isinstance(item["advantages"], float)

    def test_empty_trajectories(self, mock_tokenizer):
        """Test handling of empty trajectory list."""
        from src.training.trainer import process_trajectories

        batch, stats = process_trajectories([], mock_tokenizer)

        assert batch == []
        assert stats.total_trajectories == 0
        assert stats.total_groups == 0

    def test_single_sample_groups_skipped(self, mock_tokenizer):
        """Test that single-sample groups are skipped."""
        from src.training.trainer import process_trajectories

        trajectories = [
            {"prompt": "p1", "completion": "c1", "reward": 1.0, "group_id": "g1"},
            # Only one sample in g1 - should be skipped
        ]

        batch, stats = process_trajectories(trajectories, mock_tokenizer)

        assert len(batch) == 0
        assert stats.skipped_single_sample_groups == 1

    def test_advantage_normalization(self, mock_tokenizer):
        """Test that advantages are properly normalized within groups."""
        from src.training.trainer import process_trajectories

        # Create group with known rewards
        trajectories = [
            {"prompt": "p", "completion": "c", "reward": 10.0, "group_id": "g"},
            {"prompt": "p", "completion": "c", "reward": 20.0, "group_id": "g"},
            {"prompt": "p", "completion": "c", "reward": 30.0, "group_id": "g"},
        ]

        batch, stats = process_trajectories(trajectories, mock_tokenizer)

        # Mean = 20, Std = 8.165
        # Advantages should be normalized
        advantages = [item["advantages"] for item in batch]

        # Sum of normalized advantages should be ~0
        assert abs(sum(advantages)) < 0.01

        # Std of advantages should be ~1
        assert 0.9 < np.std(advantages) < 1.1

    def test_identical_rewards_handling(self, mock_tokenizer):
        """Test handling when all rewards in a group are identical.

        Groups with identical rewards have zero variance, so they provide
        no gradient signal and are correctly skipped during processing.
        """
        from src.training.trainer import process_trajectories

        trajectories = [
            {"prompt": "p", "completion": "c", "reward": 5.0, "group_id": "g"},
            {"prompt": "p", "completion": "c", "reward": 5.0, "group_id": "g"},
        ]

        batch, stats = process_trajectories(trajectories, mock_tokenizer)

        # Groups with zero variance are skipped (no gradient signal)
        assert len(batch) == 0
        assert stats.skipped_zero_variance_groups == 1

    def test_reward_statistics(self, mock_tokenizer, sample_trajectories):
        """Test that reward statistics are computed correctly."""
        from src.training.trainer import process_trajectories

        batch, stats = process_trajectories(sample_trajectories, mock_tokenizer)

        # Rewards: [3.0, 2.0, 4.0, 5.0, 1.0]
        assert stats.reward_mean == pytest.approx(3.0)
        assert stats.reward_min == 1.0
        assert stats.reward_max == 5.0
        assert stats.reward_std > 0

    def test_labels_mask_prompt(self, mock_tokenizer):
        """Test that labels properly mask the prompt portion."""
        from src.training.trainer import process_trajectories

        trajectories = [
            {
                "prompt": "This is a long prompt",
                "completion": "Short",
                "reward": 1.0,
                "group_id": "g",
            },
            {
                "prompt": "This is a long prompt",
                "completion": "Short",
                "reward": 2.0,
                "group_id": "g",
            },
        ]

        batch, stats = process_trajectories(trajectories, mock_tokenizer)

        for item in batch:
            labels = item["labels"]
            # Some tokens should be masked (-100)
            assert (labels == -100).any()
            # Some tokens should NOT be masked (completion)
            assert (labels != -100).any()


# =============================================================================
# Tests for GRPOLoss
# =============================================================================


class TestGRPOLoss:
    """Tests for the GRPO loss computation."""

    def test_loss_output_structure(self, mock_model):
        """Test that loss output has correct structure."""
        from src.training.loss import GRPOLoss, GRPOLossOutput

        loss_fn = GRPOLoss(mock_model, beta=0.04)

        batch = [
            {
                "input_ids": torch.randint(0, 100, (20,)),
                "attention_mask": torch.ones(20),
                "labels": torch.cat([torch.full((10,), -100), torch.randint(0, 100, (10,))]),
                "advantages": 0.5,
            },
            {
                "input_ids": torch.randint(0, 100, (20,)),
                "attention_mask": torch.ones(20),
                "labels": torch.cat([torch.full((10,), -100), torch.randint(0, 100, (10,))]),
                "advantages": -0.5,
            },
        ]

        output = loss_fn.compute_loss(batch)

        assert isinstance(output, GRPOLossOutput)
        assert isinstance(output.loss, torch.Tensor)
        assert isinstance(output.pg_loss, float)
        assert isinstance(output.kl, float)
        assert isinstance(output.mean_completion_logprob, float)
        assert isinstance(output.mean_ref_logprob, float)
        assert isinstance(output.mean_advantage, float)
        assert isinstance(output.advantage_std, float)
        assert isinstance(output.num_tokens, int)

    def test_loss_is_differentiable(self, mock_model):
        """Test that loss can be backpropagated."""
        from src.training.loss import GRPOLoss

        loss_fn = GRPOLoss(mock_model, beta=0.04)

        batch = [
            {
                "input_ids": torch.randint(0, 100, (15,)),
                "attention_mask": torch.ones(15),
                "labels": torch.cat([torch.full((5,), -100), torch.randint(0, 100, (10,))]),
                "advantages": 1.0,
            },
        ]

        output = loss_fn.compute_loss(batch)

        # Should be able to backpropagate
        output.loss.backward()

        # Gradients should exist
        for param in mock_model.parameters():
            assert param.grad is not None

    def test_kl_is_non_negative(self, mock_model):
        """Test that KL divergence is non-negative."""
        from src.training.loss import GRPOLoss

        loss_fn = GRPOLoss(mock_model, beta=0.04)

        batch = [
            {
                "input_ids": torch.randint(0, 100, (20,)),
                "attention_mask": torch.ones(20),
                "labels": torch.cat([torch.full((10,), -100), torch.randint(0, 100, (10,))]),
                "advantages": 0.0,
            },
        ]

        output = loss_fn.compute_loss(batch)

        # KL should always be >= 0
        assert output.kl >= 0

    def test_beta_affects_loss(self):
        """Test that beta parameter affects the loss when KL is non-zero."""
        from src.training.loss import GRPOLoss

        # Create a model that produces different logits for policy vs reference
        class DifferentRefModel(MockPeftModel):
            def forward(self, input_ids, attention_mask=None):
                embeddings = self.embed(input_ids)
                logits = self.linear(embeddings)
                # When adapter is disabled, produce very different logits
                # to ensure non-zero KL
                if not self._adapter_enabled:
                    logits = logits * 2 + 5.0
                return ModelOutput(logits=logits)

        model = DifferentRefModel()

        batch = [
            {
                "input_ids": torch.randint(0, 100, (20,)),
                "attention_mask": torch.ones(20),
                "labels": torch.cat([torch.full((10,), -100), torch.randint(0, 100, (10,))]),
                "advantages": 1.0,
            },
        ]

        loss_fn_low_beta = GRPOLoss(model, beta=0.01)  # type: ignore[arg-type]
        loss_fn_high_beta = GRPOLoss(model, beta=10.0)  # type: ignore[arg-type]

        # Reset model state
        model.zero_grad()
        output_low = loss_fn_low_beta.compute_loss(batch)

        model.zero_grad()
        output_high = loss_fn_high_beta.compute_loss(batch)

        # Both should have non-zero KL
        assert output_low.kl > 0, "KL should be positive with different ref model"
        assert output_high.kl > 0, "KL should be positive with different ref model"

        # Policy gradient loss should be the same regardless of beta
        assert abs(output_low.pg_loss - output_high.pg_loss) < 0.01

        # Total loss should be different due to different beta * KL contribution
        # With higher beta, the loss should be higher (since KL > 0)
        assert output_high.loss.item() > output_low.loss.item()

    def test_advantage_direction(self, mock_model):
        """Test that positive advantages encourage actions, negative discourage."""
        from src.training.loss import GRPOLoss

        loss_fn = GRPOLoss(mock_model, beta=0.0)  # Disable KL for clarity

        # Same sequence, different advantages
        base_batch = {
            "input_ids": torch.randint(0, 100, (20,)),
            "attention_mask": torch.ones(20),
            "labels": torch.cat([torch.full((10,), -100), torch.randint(0, 100, (10,))]),
        }

        batch_positive = [{**base_batch, "advantages": 2.0}]
        batch_negative = [{**base_batch, "advantages": -2.0}]

        mock_model.zero_grad()
        output_pos = loss_fn.compute_loss(batch_positive)

        mock_model.zero_grad()
        output_neg = loss_fn.compute_loss(batch_negative)

        # With positive advantage, pg_loss should be negative (encouraging)
        # With negative advantage, pg_loss should be positive (discouraging)
        # Signs should be opposite
        assert output_pos.pg_loss * output_neg.pg_loss < 0

    def test_disable_adapter_called(self, mock_model):
        """Test that disable_adapter is called for reference model pass."""
        from src.training.loss import GRPOLoss

        loss_fn = GRPOLoss(mock_model, beta=0.04)

        batch = [
            {
                "input_ids": torch.randint(0, 100, (20,)),
                "attention_mask": torch.ones(20),
                "labels": torch.cat([torch.full((10,), -100), torch.randint(0, 100, (10,))]),
                "advantages": 0.5,
            },
        ]

        # Track if disable_adapter was called
        original_disable = mock_model.disable_adapter
        call_count = [0]

        def tracking_disable():
            call_count[0] += 1
            return original_disable()

        mock_model.disable_adapter = tracking_disable

        loss_fn.compute_loss(batch)

        assert call_count[0] == 1, "disable_adapter should be called once per loss computation"


# =============================================================================
# Tests for TrajectoryStats
# =============================================================================


class TestTrajectoryStats:
    """Tests for the TrajectoryStats dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        from src.training.trainer import TrajectoryStats

        stats = TrajectoryStats(
            total_trajectories=100,
            total_groups=10,
            reward_mean=3.5,
            reward_std=1.2,
            reward_min=1.0,
            reward_max=6.0,
            group_sizes=[10, 10, 10],
            group_reward_stds=[0.5, 0.8, 0.6],
            total_tokens=5000,
            avg_completion_tokens=50.0,
        )

        d = stats.to_dict()

        assert d["total_trajectories"] == 100
        assert d["total_groups"] == 10
        assert d["reward_mean"] == 3.5
        assert d["avg_group_size"] == 10.0
        assert d["avg_group_reward_std"] == pytest.approx(0.633, rel=0.01)

    def test_empty_stats(self):
        """Test stats with empty data."""
        from src.training.trainer import TrajectoryStats

        stats = TrajectoryStats()
        d = stats.to_dict()

        assert d["total_trajectories"] == 0
        assert d["avg_group_size"] == 0
        assert d["avg_group_reward_std"] == 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestTrainingIntegration:
    """Integration tests combining multiple components."""

    def test_full_pipeline(self, mock_tokenizer, sample_trajectories, mock_model):
        """Test the full trajectory processing to loss computation pipeline."""
        from src.training.loss import GRPOLoss
        from src.training.trainer import process_trajectories

        # Process trajectories
        batch, stats = process_trajectories(sample_trajectories, mock_tokenizer)

        assert len(batch) > 0

        # Compute loss
        loss_fn = GRPOLoss(mock_model, beta=0.04)

        # Take a small chunk
        chunk = batch[:2]
        output = loss_fn.compute_loss(chunk)

        # Should produce valid loss
        assert not torch.isnan(output.loss)
        assert not torch.isinf(output.loss)

        # Should be able to backprop
        output.loss.backward()

    def test_gradient_accumulation_simulation(self, mock_tokenizer, mock_model):
        """Test simulated gradient accumulation."""
        from src.training.loss import GRPOLoss
        from src.training.trainer import process_trajectories

        # Create larger dataset
        trajectories = []
        for i in range(8):
            trajectories.append(
                {
                    "prompt": f"Prompt {i}",
                    "completion": f"Completion {i}",
                    "reward": float(i),
                    "group_id": f"group_{i // 2}",  # 4 groups of 2
                }
            )

        batch, stats = process_trajectories(trajectories, mock_tokenizer)
        loss_fn = GRPOLoss(mock_model, beta=0.04)

        # Simulate gradient accumulation with chunk_size=2
        optimizer = torch.optim.SGD(mock_model.parameters(), lr=0.01)
        optimizer.zero_grad()

        chunk_size = 2
        num_chunks = 0

        for i in range(0, len(batch), chunk_size):
            chunk = batch[i : i + chunk_size]
            if not chunk:
                break

            output = loss_fn.compute_loss(chunk)
            scaled_loss = output.loss / max(1, len(batch) // chunk_size)
            scaled_loss.backward()
            num_chunks += 1

        # Should have processed multiple chunks
        assert num_chunks > 1

        # Gradients should exist
        for param in mock_model.parameters():
            assert param.grad is not None

        # Optimizer step should work
        optimizer.step()


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_very_long_completion(self, mock_tokenizer):
        """Test handling of very long completions."""
        from src.training.trainer import process_trajectories

        trajectories = [
            {
                "prompt": "Short",
                "completion": "A" * 10000,  # Very long
                "reward": 1.0,
                "group_id": "g",
            },
            {
                "prompt": "Short",
                "completion": "B" * 10000,
                "reward": 2.0,
                "group_id": "g",
            },
        ]

        # Should not crash, should truncate
        batch, stats = process_trajectories(trajectories, mock_tokenizer)
        assert len(batch) == 2

    def test_empty_completion(self, mock_tokenizer):
        """Test handling of empty completions."""
        from src.training.trainer import process_trajectories

        trajectories = [
            {"prompt": "Prompt", "completion": "", "reward": 1.0, "group_id": "g"},
            {"prompt": "Prompt", "completion": "", "reward": 2.0, "group_id": "g"},
        ]

        # Should handle gracefully
        batch, stats = process_trajectories(trajectories, mock_tokenizer)
        assert len(batch) == 2

    def test_extreme_rewards(self, mock_tokenizer):
        """Test handling of extreme reward values."""
        from src.training.trainer import process_trajectories

        trajectories = [
            {"prompt": "p", "completion": "c", "reward": 1e10, "group_id": "g"},
            {"prompt": "p", "completion": "c", "reward": -1e10, "group_id": "g"},
        ]

        batch, stats = process_trajectories(trajectories, mock_tokenizer)

        # Normalized advantages should still be finite
        for item in batch:
            assert np.isfinite(item["advantages"])

    def test_many_groups(self, mock_tokenizer):
        """Test handling of many groups."""
        from src.training.trainer import process_trajectories

        trajectories = []
        for g in range(100):
            for s in range(3):
                trajectories.append(
                    {
                        "prompt": f"p{g}_{s}",
                        "completion": f"c{g}_{s}",
                        "reward": float(s),
                        "group_id": f"group_{g}",
                    }
                )

        batch, stats = process_trajectories(trajectories, mock_tokenizer)

        assert stats.total_groups == 100
        assert stats.total_trajectories == 300
        assert len(batch) == 300


# =============================================================================
# Tests for DAPO/GTPO Improvements
# =============================================================================


class TestPPOClipping:
    """Tests for PPO clipping (DAPO-style Clip-Higher)."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock PEFT model for testing."""
        return MockPeftModel()

    def test_ppo_clipping_with_rollout_logprobs(self, mock_model):
        """Test that PPO clipping works when rollout_logprobs are provided."""
        from src.training.loss import GRPOLoss

        loss_fn = GRPOLoss(
            mock_model,
            beta=0.0,
            use_ppo_clipping=True,
            ppo_epsilon_low=0.2,
            ppo_epsilon_high=0.28,
        )

        batch = [
            {
                "input_ids": torch.randint(0, 100, (20,)),
                "attention_mask": torch.ones(20),
                "labels": torch.cat([torch.full((10,), -100), torch.randint(0, 100, (10,))]),
                "advantages": 1.0,
                "rollout_logprobs": -15.0,  # Approximate logprob for 10 tokens
            },
            {
                "input_ids": torch.randint(0, 100, (20,)),
                "attention_mask": torch.ones(20),
                "labels": torch.cat([torch.full((10,), -100), torch.randint(0, 100, (10,))]),
                "advantages": -1.0,
                "rollout_logprobs": -12.0,
            },
        ]

        output = loss_fn.compute_loss(batch)

        # Should have ratio metrics when clipping is active
        assert output.ratio_mean > 0  # Ratio should be positive
        assert output.ratio_clipped_fraction >= 0  # Can be 0 if no clipping needed

    def test_ppo_clipping_fallback_without_rollout_logprobs(self, mock_model):
        """Test that PPO clipping falls back to vanilla REINFORCE without rollout_logprobs."""
        from src.training.loss import GRPOLoss

        loss_fn = GRPOLoss(
            mock_model,
            beta=0.0,
            use_ppo_clipping=True,  # Enabled but won't be used
        )

        batch = [
            {
                "input_ids": torch.randint(0, 100, (20,)),
                "attention_mask": torch.ones(20),
                "labels": torch.cat([torch.full((10,), -100), torch.randint(0, 100, (10,))]),
                "advantages": 1.0,
                # No rollout_logprobs
            },
        ]

        # Should not raise, should fall back to vanilla REINFORCE
        output = loss_fn.compute_loss(batch)

        # Ratio should be default (1.0) since clipping wasn't used
        assert output.ratio_mean == 1.0
        assert output.ratio_clipped_fraction == 0.0

    def test_ppo_clipping_disabled(self, mock_model):
        """Test vanilla REINFORCE when PPO clipping is disabled."""
        from src.training.loss import GRPOLoss

        loss_fn = GRPOLoss(
            mock_model,
            beta=0.0,
            use_ppo_clipping=False,
        )

        batch = [
            {
                "input_ids": torch.randint(0, 100, (20,)),
                "attention_mask": torch.ones(20),
                "labels": torch.cat([torch.full((10,), -100), torch.randint(0, 100, (10,))]),
                "advantages": 1.0,
                "rollout_logprobs": -15.0,  # Present but shouldn't be used
            },
        ]

        output = loss_fn.compute_loss(batch)

        # Ratio should be default since clipping is disabled
        assert output.ratio_mean == 1.0


class TestTokenLevelLoss:
    """Tests for token-level loss weighting (DAPO-style)."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock PEFT model for testing."""
        return MockPeftModel()

    def test_token_level_loss_enabled(self, mock_model):
        """Test that token-level loss weights by token count."""
        from src.training.loss import GRPOLoss

        loss_fn = GRPOLoss(
            mock_model,
            beta=0.0,
            use_ppo_clipping=False,
            use_token_level_loss=True,
        )

        # Create batch with different completion lengths
        batch = [
            {
                "input_ids": torch.randint(0, 100, (15,)),  # 5 completion tokens
                "attention_mask": torch.ones(15),
                "labels": torch.cat([torch.full((10,), -100), torch.randint(0, 100, (5,))]),
                "advantages": 1.0,
            },
            {
                "input_ids": torch.randint(0, 100, (30,)),  # 20 completion tokens
                "attention_mask": torch.ones(30),
                "labels": torch.cat([torch.full((10,), -100), torch.randint(0, 100, (20,))]),
                "advantages": 1.0,
            },
        ]

        output = loss_fn.compute_loss(batch)
        assert isinstance(output.loss, torch.Tensor)

    def test_token_level_loss_disabled(self, mock_model):
        """Test sample-level loss when token-level is disabled."""
        from src.training.loss import GRPOLoss

        loss_fn = GRPOLoss(
            mock_model,
            beta=0.0,
            use_ppo_clipping=False,
            use_token_level_loss=False,
        )

        batch = [
            {
                "input_ids": torch.randint(0, 100, (20,)),
                "attention_mask": torch.ones(20),
                "labels": torch.cat([torch.full((10,), -100), torch.randint(0, 100, (10,))]),
                "advantages": 1.0,
            },
        ]

        output = loss_fn.compute_loss(batch)
        assert isinstance(output.loss, torch.Tensor)


class TestEntropyMonitoring:
    """Tests for entropy monitoring (GTPO-style collapse detection).

    NOTE: Entropy computation is currently disabled due to memory constraints
    (computing full softmax over 150k vocab causes OOM). These tests verify
    the disabled state - re-enable when memory-efficient entropy is implemented.
    """

    @pytest.fixture
    def mock_model(self):
        """Create a mock PEFT model for testing."""
        return MockPeftModel()

    def test_entropy_fields_exist(self, mock_model):
        """Test that entropy fields exist in output with reasonable values."""
        from src.training.loss import GRPOLoss

        loss_fn = GRPOLoss(mock_model, beta=0.0)

        batch = [
            {
                "input_ids": torch.randint(0, 100, (20,)),
                "attention_mask": torch.ones(20),
                "labels": torch.cat([torch.full((10,), -100), torch.randint(0, 100, (10,))]),
                "advantages": 1.0,
            },
        ]

        output = loss_fn.compute_loss(batch)

        # Entropy fields should exist with reasonable values
        assert hasattr(output, "entropy_mean")
        assert hasattr(output, "entropy_std")
        # Entropy should be positive (top-k approximation)
        assert output.entropy_mean >= 0.0
        # Single sample, std should be 0
        assert output.entropy_std == 0.0

    def test_entropy_statistics_computed(self, mock_model):
        """Test that entropy statistics are computed correctly for multiple samples."""
        from src.training.loss import GRPOLoss

        loss_fn = GRPOLoss(mock_model, beta=0.0)

        batch = [
            {
                "input_ids": torch.randint(0, 100, (20,)),
                "attention_mask": torch.ones(20),
                "labels": torch.cat([torch.full((10,), -100), torch.randint(0, 100, (10,))]),
                "advantages": 0.5,
            },
            {
                "input_ids": torch.randint(0, 100, (20,)),
                "attention_mask": torch.ones(20),
                "labels": torch.cat([torch.full((10,), -100), torch.randint(0, 100, (10,))]),
                "advantages": -0.5,
            },
        ]

        output = loss_fn.compute_loss(batch)

        # Entropy should be positive (computed via top-k approximation)
        assert output.entropy_mean > 0.0
        # With 2 samples, std might be small but can be computed
        assert output.entropy_std >= 0.0


# =============================================================================
# Tests for Discounted Cumulative Returns
# =============================================================================


class TestDiscountedReturns:
    """Tests for discounted cumulative returns in reward computation."""

    def test_discounted_returns_basic(self):
        """Test that discounted returns are computed correctly."""
        # Simulate the algorithm from build_trajectories
        gamma = 0.9

        # Step deltas: [1.0, 2.0, 3.0] at steps 0, 1, 2
        deltas = {0: 1.0, 1: 2.0, 2: 3.0}
        sorted_steps = sorted(deltas.keys())

        # Compute backwards
        cumulative = 0.0
        returns = {}
        for step in reversed(sorted_steps):
            cumulative = deltas[step] + gamma * cumulative
            returns[step] = cumulative

        # Expected:
        # R_2 = 3.0
        # R_1 = 2.0 + 0.9 * 3.0 = 4.7
        # R_0 = 1.0 + 0.9 * 4.7 = 5.23
        assert abs(returns[2] - 3.0) < 1e-6
        assert abs(returns[1] - 4.7) < 1e-6
        assert abs(returns[0] - 5.23) < 1e-6

    def test_discounted_returns_gamma_zero(self):
        """Test that gamma=0 gives immediate rewards only."""
        gamma = 0.0

        deltas = {0: 1.0, 1: 2.0, 2: 3.0}
        sorted_steps = sorted(deltas.keys())

        cumulative = 0.0
        returns = {}
        for step in reversed(sorted_steps):
            cumulative = deltas[step] + gamma * cumulative
            returns[step] = cumulative

        # With gamma=0, each step only sees its own delta
        assert returns[0] == 1.0
        assert returns[1] == 2.0
        assert returns[2] == 3.0

    def test_discounted_returns_gamma_one(self):
        """Test that gamma=1 gives undiscounted sum of future rewards."""
        gamma = 1.0

        deltas = {0: 1.0, 1: 2.0, 2: 3.0}
        sorted_steps = sorted(deltas.keys())

        cumulative = 0.0
        returns = {}
        for step in reversed(sorted_steps):
            cumulative = deltas[step] + gamma * cumulative
            returns[step] = cumulative

        # With gamma=1, full sum of future rewards
        # R_2 = 3.0
        # R_1 = 2.0 + 3.0 = 5.0
        # R_0 = 1.0 + 5.0 = 6.0
        assert returns[2] == 3.0
        assert returns[1] == 5.0
        assert returns[0] == 6.0

    def test_discounted_returns_negative_deltas(self):
        """Test with negative step deltas (losing territory)."""
        gamma = 0.95

        # A bad start followed by recovery
        deltas = {0: -2.0, 1: -1.0, 2: 5.0}
        sorted_steps = sorted(deltas.keys())

        cumulative = 0.0
        returns = {}
        for step in reversed(sorted_steps):
            cumulative = deltas[step] + gamma * cumulative
            returns[step] = cumulative

        # R_2 = 5.0
        # R_1 = -1.0 + 0.95 * 5.0 = 3.75
        # R_0 = -2.0 + 0.95 * 3.75 = 1.5625
        assert abs(returns[2] - 5.0) < 1e-6
        assert abs(returns[1] - 3.75) < 1e-6
        assert abs(returns[0] - 1.5625) < 1e-6

        # Early steps should still get positive credit from later success
        assert returns[0] > 0

    def test_discounted_returns_single_step(self):
        """Test with only one step."""
        gamma = 0.9

        deltas = {0: 5.0}
        sorted_steps = sorted(deltas.keys())

        cumulative = 0.0
        returns = {}
        for step in reversed(sorted_steps):
            cumulative = deltas[step] + gamma * cumulative
            returns[step] = cumulative

        assert returns[0] == 5.0

    def test_discount_factor_effect_on_horizon(self):
        """Test that different gammas weight future differently."""
        # With 10 steps all having delta=1.0
        deltas = dict.fromkeys(range(10), 1.0)
        sorted_steps = sorted(deltas.keys())

        # Compute for different gammas
        def compute_returns(gamma):
            cumulative = 0.0
            returns = {}
            for step in reversed(sorted_steps):
                cumulative = deltas[step] + gamma * cumulative
                returns[step] = cumulative
            return returns

        returns_low = compute_returns(0.5)
        returns_mid = compute_returns(0.9)
        returns_high = compute_returns(0.99)

        # First step should have more credit for future with higher gamma
        assert returns_low[0] < returns_mid[0] < returns_high[0]

        # Last step should be the same regardless of gamma
        assert returns_low[9] == returns_mid[9] == returns_high[9] == 1.0


# =============================================================================
# Tests for GSPO (Group Sequence Policy Optimization)
# =============================================================================


class TestGSPOLoss:
    """Tests for GSPO loss (sequence-level importance sampling)."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock PEFT model for testing."""
        return MockPeftModel()

    def test_gspo_sequence_ratio_computed(self, mock_model):
        """Test that GSPO computes sequence-level ratios correctly."""
        from src.training.loss import GRPOLoss

        loss_fn = GRPOLoss(
            mock_model,
            beta=0.0,
            use_ppo_clipping=True,
            loss_type="gspo",
        )

        batch = [
            {
                "input_ids": torch.randint(0, 100, (20,)),
                "attention_mask": torch.ones(20),
                "labels": torch.cat([torch.full((10,), -100), torch.randint(0, 100, (10,))]),
                "advantages": 1.0,
                "rollout_logprobs": -15.0,  # Sum of 10 token logprobs
            },
            {
                "input_ids": torch.randint(0, 100, (20,)),
                "attention_mask": torch.ones(20),
                "labels": torch.cat([torch.full((10,), -100), torch.randint(0, 100, (10,))]),
                "advantages": -1.0,
                "rollout_logprobs": -12.0,
            },
        ]

        output = loss_fn.compute_loss(batch)

        # GSPO should populate sequence-level metrics
        assert output.gspo_sequence_ratio_mean > 0
        assert hasattr(output, "gspo_log_ratio_mean")
        # Should still produce valid loss
        assert not torch.isnan(output.loss)
        assert not torch.isinf(output.loss)

    def test_gspo_vs_grpo_different_metrics(self, mock_model):
        """Test that GSPO and GRPO produce different metrics."""
        from src.training.loss import GRPOLoss

        batch = [
            {
                "input_ids": torch.randint(0, 100, (20,)),
                "attention_mask": torch.ones(20),
                "labels": torch.cat([torch.full((10,), -100), torch.randint(0, 100, (10,))]),
                "advantages": 1.0,
                "rollout_logprobs": -15.0,
            },
        ]

        loss_grpo = GRPOLoss(mock_model, beta=0.0, use_ppo_clipping=True, loss_type="grpo")
        loss_gspo = GRPOLoss(mock_model, beta=0.0, use_ppo_clipping=True, loss_type="gspo")

        mock_model.zero_grad()
        output_grpo = loss_grpo.compute_loss(batch)

        mock_model.zero_grad()
        output_gspo = loss_gspo.compute_loss(batch)

        # GSPO should have populated gspo metrics
        assert output_gspo.gspo_sequence_ratio_mean > 0
        # GRPO should have default gspo metrics
        assert output_grpo.gspo_sequence_ratio_mean == 1.0

    def test_gspo_loss_is_differentiable(self, mock_model):
        """Test that GSPO loss can be backpropagated."""
        from src.training.loss import GRPOLoss

        loss_fn = GRPOLoss(
            mock_model,
            beta=0.0,
            use_ppo_clipping=True,
            loss_type="gspo",
        )

        batch = [
            {
                "input_ids": torch.randint(0, 100, (20,)),
                "attention_mask": torch.ones(20),
                "labels": torch.cat([torch.full((10,), -100), torch.randint(0, 100, (10,))]),
                "advantages": 1.0,
                "rollout_logprobs": -15.0,
            },
        ]

        output = loss_fn.compute_loss(batch)

        # Should be able to backpropagate
        output.loss.backward()

        # Gradients should exist
        for param in mock_model.parameters():
            assert param.grad is not None

    def test_gspo_clipping_applied(self, mock_model):
        """Test that sequence-level clipping is applied in GSPO."""
        from src.training.loss import GRPOLoss

        loss_fn = GRPOLoss(
            mock_model,
            beta=0.0,
            use_ppo_clipping=True,
            ppo_epsilon_low=0.2,
            ppo_epsilon_high=0.28,
            loss_type="gspo",
        )

        # Create batch with extreme rollout logprobs to trigger clipping
        batch = [
            {
                "input_ids": torch.randint(0, 100, (20,)),
                "attention_mask": torch.ones(20),
                "labels": torch.cat([torch.full((10,), -100), torch.randint(0, 100, (10,))]),
                "advantages": 1.0,
                "rollout_logprobs": -50.0,  # Very different from expected
            },
        ]

        output = loss_fn.compute_loss(batch)

        # Should have clip fraction metric
        assert output.gspo_sequence_ratio_clipped_fraction >= 0

    def test_gspo_fallback_without_rollout_logprobs(self, mock_model):
        """Test that GSPO falls back to vanilla REINFORCE without rollout_logprobs."""
        from src.training.loss import GRPOLoss

        loss_fn = GRPOLoss(
            mock_model,
            beta=0.0,
            use_ppo_clipping=True,
            loss_type="gspo",
        )

        batch = [
            {
                "input_ids": torch.randint(0, 100, (20,)),
                "attention_mask": torch.ones(20),
                "labels": torch.cat([torch.full((10,), -100), torch.randint(0, 100, (10,))]),
                "advantages": 1.0,
                # No rollout_logprobs
            },
        ]

        # Should not raise, should fall back to vanilla REINFORCE
        output = loss_fn.compute_loss(batch)

        # GSPO metrics should be defaults since clipping wasn't used
        assert output.gspo_sequence_ratio_mean == 1.0


class TestDynamicSampling:
    """Tests for DAPO-style dynamic sampling (min_reward_variance)."""

    @pytest.fixture
    def mock_tokenizer(self):
        return MockTokenizer()

    def test_min_reward_variance_filters_low_variance_groups(self, mock_tokenizer):
        """Test that groups with variance below threshold are rejected."""
        from src.training.trainer import process_trajectories

        # Create group with very low variance
        trajectories = [
            {"prompt": "p", "completion": "c", "reward": 5.0, "group_id": "g"},
            {"prompt": "p", "completion": "c", "reward": 5.001, "group_id": "g"},
            {"prompt": "p", "completion": "c", "reward": 5.002, "group_id": "g"},
        ]

        # With high min_reward_variance, this group should be rejected
        batch, stats = process_trajectories(
            trajectories,
            mock_tokenizer,
            min_reward_variance=0.1,  # Variance threshold of 0.1
        )

        # Should be skipped (variance is ~0.000001, way below 0.1)
        assert len(batch) == 0
        assert stats.skipped_zero_variance_groups == 1

    def test_min_reward_variance_allows_high_variance_groups(self, mock_tokenizer):
        """Test that groups with variance above threshold are kept."""
        from src.training.trainer import process_trajectories

        # Create group with high variance
        trajectories = [
            {"prompt": "p", "completion": "c", "reward": 1.0, "group_id": "g"},
            {"prompt": "p", "completion": "c", "reward": 5.0, "group_id": "g"},
            {"prompt": "p", "completion": "c", "reward": 9.0, "group_id": "g"},
        ]

        # With low min_reward_variance, this group should be kept
        batch, stats = process_trajectories(
            trajectories,
            mock_tokenizer,
            min_reward_variance=0.01,  # Low threshold
        )

        # Should be kept (variance is ~10.67, way above 0.01)
        assert len(batch) == 3
        assert stats.skipped_zero_variance_groups == 0

    def test_min_reward_variance_zero_disables_feature(self, mock_tokenizer):
        """Test that min_reward_variance=0 disables the feature."""
        from src.training.trainer import process_trajectories

        # Create group with low variance that would normally be filtered
        trajectories = [
            {"prompt": "p", "completion": "c", "reward": 5.0, "group_id": "g"},
            {"prompt": "p", "completion": "c", "reward": 5.001, "group_id": "g"},
        ]

        # With min_reward_variance=0, only advantage_min_std applies
        batch, stats = process_trajectories(
            trajectories,
            mock_tokenizer,
            min_reward_variance=0.0,
            advantage_min_std=1e-8,  # Very low threshold
        )

        # Should be kept since min_reward_variance is disabled
        assert len(batch) == 2
