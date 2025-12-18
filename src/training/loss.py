from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

if TYPE_CHECKING:
    from peft import PeftModel


@dataclass
class KLControllerConfig:
    """Configuration for adaptive KL penalty controller."""

    initial_beta: float = 0.04  # Starting KL penalty coefficient
    warmup_steps: int = 0  # Steps to linearly warmup beta from 0
    target_kl: float | None = None  # Target KL for adaptive control (None = disabled)
    horizon: int = 10  # Steps to smooth KL for adaptive control
    beta_min: float = 0.001  # Minimum beta when using adaptive control
    beta_max: float = 0.5  # Maximum beta when using adaptive control


class AdaptiveKLController:
    """
    Manages KL penalty coefficient (beta) with warmup and adaptive adjustment.

    Features:
    1. Linear warmup: beta starts at 0 and increases to initial_beta over warmup_steps
    2. Adaptive control (optional): PPO-style adjustment based on observed KL
       - If KL > 1.5 * target: increase beta by 1.5x
       - If KL < 0.5 * target: decrease beta by 1.5x
    3. Exponential moving average (EMA) of KL for stability
    """

    def __init__(self, config: KLControllerConfig):
        self.config = config
        self.current_step = 0
        self._beta = config.initial_beta
        self._kl_ema: float | None = None  # Exponential moving average of KL

    def get_beta(self) -> float:
        """
        Get the current effective beta value.

        During warmup, this linearly interpolates from 0 to initial_beta.
        After warmup, returns the current (possibly adapted) beta.
        """
        if self.current_step < self.config.warmup_steps:
            # Linear warmup from 0 to initial_beta
            warmup_fraction = self.current_step / self.config.warmup_steps
            return self.config.initial_beta * warmup_fraction
        return self._beta

    def get_warmup_progress(self) -> float:
        """Returns warmup progress as a fraction [0, 1]."""
        if self.config.warmup_steps == 0:
            return 1.0
        return min(1.0, self.current_step / self.config.warmup_steps)

    def get_kl_ema(self) -> float | None:
        """Returns the current KL EMA value."""
        return self._kl_ema

    def step_update(self, observed_kl: float) -> dict[str, float]:
        """
        Update controller state after a training step.

        Args:
            observed_kl: The KL divergence observed in this step.

        Returns:
            Dict with diagnostic info for logging:
            - 'beta': current beta value
            - 'warmup_progress': fraction of warmup complete
            - 'kl_ema': exponential moving average of KL
            - 'beta_adjusted': 1 if beta was adjusted this step, else 0
        """
        self.current_step += 1
        beta_adjusted = 0

        # Update KL EMA
        alpha = 2.0 / (self.config.horizon + 1)  # EMA smoothing factor
        if self._kl_ema is None:
            self._kl_ema = observed_kl
        else:
            self._kl_ema = alpha * observed_kl + (1 - alpha) * self._kl_ema

        # Adaptive control (only after warmup and if target is set)
        if self.config.target_kl is not None and self.current_step >= self.config.warmup_steps:
            # PPO-style adaptive beta adjustment
            if self._kl_ema > 1.5 * self.config.target_kl:
                # KL too high - increase penalty
                self._beta = min(self._beta * 1.5, self.config.beta_max)
                beta_adjusted = 1
            elif self._kl_ema < 0.5 * self.config.target_kl:
                # KL too low - decrease penalty
                self._beta = max(self._beta / 1.5, self.config.beta_min)
                beta_adjusted = -1

        return {
            "beta": self.get_beta(),
            "warmup_progress": self.get_warmup_progress(),
            "kl_ema": self._kl_ema,
            "beta_adjusted": float(beta_adjusted),
        }


@dataclass
class GRPOLossOutput:
    """Structured output from GRPO loss computation for better observability."""

    loss: torch.Tensor
    pg_loss: float
    kl: float  # Alias for kl_mean for backwards compatibility

    # Additional metrics for logging
    mean_completion_logprob: float
    mean_ref_logprob: float
    mean_advantage: float
    advantage_std: float
    num_tokens: int

    # KL statistics (per-batch)
    kl_mean: float = 0.0
    kl_max: float = 0.0
    kl_min: float = 0.0
    kl_std: float = 0.0
    effective_beta: float = 0.04  # The beta value actually used for this batch

    # Optimization tracking
    used_cached_ref_logprobs: bool = False


class GRPOLoss:
    """
    Implements Group Relative Policy Optimization loss manually
    since we generate data externally.

    CRITICAL: This class expects a PeftModel and uses `disable_adapter()`
    context manager to compute reference logprobs from the base model,
    UNLESS pre-computed reference logprobs are provided in the batch.

    Optimization: When 'ref_logprobs' is present in batch items, skip the
    reference forward pass entirely. This saves ~50% of compute.
    """

    def __init__(
        self,
        model: "PeftModel",
        beta: float = 0.04,
        epsilon: float = 1e-4,
        kl_controller: AdaptiveKLController | None = None,
    ):
        """
        Args:
            model: A PeftModel (LoRA-wrapped model). Reference logprobs are computed
                   by disabling the adapter temporarily, unless pre-computed.
            beta: KL penalty coefficient (used when kl_controller is None).
            epsilon: Small constant for numerical stability.
            kl_controller: Optional adaptive KL controller. If provided, its get_beta()
                value is used instead of the fixed beta parameter.
        """
        self.model = model
        self.beta = beta
        self.epsilon = epsilon
        self.kl_controller = kl_controller

    def compute_loss(self, batch: list[dict]) -> GRPOLossOutput:
        """
        Compute GRPO loss for a batch of trajectories.

        Args:
            batch: List of dicts with keys:
                - 'input_ids': Tensor of shape (seq_len,)
                - 'attention_mask': Tensor of shape (seq_len,)
                - 'labels': Tensor of shape (seq_len,) with -100 for prompt tokens
                - 'advantages': Float, the normalized advantage
                - 'ref_logprobs': Float (optional) - pre-computed reference logprobs
                    If ALL items have this key, skip reference forward pass.

        Returns:
            GRPOLossOutput with loss tensor and detailed metrics.
        """
        # Get device from model parameters
        device = next(self.model.parameters()).device

        # 1. Stack Batch (handle variable lengths via padding)
        input_ids = pad_sequence(
            [b["input_ids"] for b in batch], batch_first=True, padding_value=0
        ).to(device)
        attention_mask = pad_sequence(
            [b["attention_mask"] for b in batch], batch_first=True, padding_value=0
        ).to(device)
        labels = pad_sequence(
            [b["labels"] for b in batch], batch_first=True, padding_value=-100
        ).to(device)
        advantages = torch.tensor(
            [b["advantages"] for b in batch], dtype=torch.float32, device=device
        )

        # 2. Forward Pass (Current Policy with LoRA enabled)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]  # Shift left for next-token prediction
        shifted_labels = labels[:, 1:]  # Shift right to align with predictions

        # 3. Compute Token LogProbs (negative cross-entropy)
        per_token_loss = F.cross_entropy(logits.transpose(1, 2), shifted_labels, reduction="none")

        # 4. Mask out prompt tokens (where labels == -100)
        token_mask = (shifted_labels != -100).float()
        token_log_probs = -per_token_loss * token_mask
        num_completion_tokens = token_mask.sum().int().item()

        # Sum log probs over the completion length per sequence
        completion_log_probs = token_log_probs.sum(dim=1)

        # 5. Reference LogProbs: Use cached or compute via forward pass
        # Check if ALL batch items have pre-computed reference logprobs
        # CRITICAL: Must be all-or-nothing to avoid mixing cached and computed logprobs
        has_ref_logprobs = ["ref_logprobs" in b for b in batch]
        all_have_ref_logprobs = all(has_ref_logprobs)
        none_have_ref_logprobs = not any(has_ref_logprobs)

        # Validate consistency
        if not all_have_ref_logprobs and not none_have_ref_logprobs:
            # Mixed batch - some have ref_logprobs, some don't
            # This is a bug in trajectory processing
            num_with = sum(has_ref_logprobs)
            num_without = len(batch) - num_with
            raise ValueError(
                f"Inconsistent reference logprobs in batch: {num_with} items have ref_logprobs, "
                f"{num_without} don't. This creates biased KL estimates. "
                "All trajectories in a batch must have the same ref_logprobs availability."
            )

        used_cached_ref = False

        if all_have_ref_logprobs:
            # OPTIMIZATION: Use pre-computed reference logprobs (skip forward pass!)
            ref_completion_log_probs = torch.tensor(
                [b["ref_logprobs"] for b in batch], dtype=torch.float32, device=device
            )
            used_cached_ref = True
        else:
            # Fallback: Forward Pass (Reference Policy) for KL
            # CRITICAL: Use disable_adapter() to get base model logprobs
            with torch.no_grad():
                with self.model.disable_adapter():
                    ref_outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                ref_logits = ref_outputs.logits[:, :-1, :]

                # Compute Reference LogProbs
                ref_per_token_loss = F.cross_entropy(
                    ref_logits.transpose(1, 2), shifted_labels, reduction="none"
                )
                ref_token_log_probs = -ref_per_token_loss * token_mask
                ref_completion_log_probs = ref_token_log_probs.sum(dim=1)

        # 6. KL Divergence Approximation (Schulman's estimator)
        # http://joschu.net/blog/kl-approx.html
        log_ratio = completion_log_probs - ref_completion_log_probs
        kl = torch.exp(log_ratio) - 1 - log_ratio

        # Compute KL statistics for diagnostics
        kl_mean = kl.mean().item()
        kl_max = kl.max().item()
        kl_min = kl.min().item()
        kl_std = kl.std().item() if len(kl) > 1 else 0.0

        # 7. GRPO Loss = -Advantage * LogProb + Beta * KL
        # Use adaptive beta from controller if available, otherwise fall back to fixed beta
        effective_beta = (
            self.kl_controller.get_beta() if self.kl_controller is not None else self.beta
        )
        # advantages are already normalized by the trainer
        pg_loss = -(advantages.detach() * completion_log_probs)
        total_loss = pg_loss + (effective_beta * kl)

        return GRPOLossOutput(
            loss=total_loss.mean(),
            pg_loss=pg_loss.mean().item(),
            kl=kl_mean,  # Keep for backwards compatibility
            mean_completion_logprob=completion_log_probs.mean().item(),
            mean_ref_logprob=ref_completion_log_probs.mean().item(),
            mean_advantage=advantages.mean().item(),
            advantage_std=advantages.std().item() if len(advantages) > 1 else 0.0,
            num_tokens=int(num_completion_tokens),
            kl_mean=kl_mean,
            kl_max=kl_max,
            kl_min=kl_min,
            kl_std=kl_std,
            effective_beta=effective_beta,
            used_cached_ref_logprobs=used_cached_ref,
        )
