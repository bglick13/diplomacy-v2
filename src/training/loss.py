from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

if TYPE_CHECKING:
    from peft import PeftModel


@dataclass
class GRPOLossOutput:
    """Structured output from GRPO loss computation for better observability."""

    loss: torch.Tensor
    pg_loss: float
    kl: float

    # Additional metrics for logging
    mean_completion_logprob: float
    mean_ref_logprob: float
    mean_advantage: float
    advantage_std: float
    num_tokens: int


class GRPOLoss:
    """
    Implements Group Relative Policy Optimization loss manually
    since we generate data externally.

    CRITICAL: This class expects a PeftModel and uses `disable_adapter()`
    context manager to compute reference logprobs from the base model.
    """

    def __init__(self, model: "PeftModel", beta: float = 0.04, epsilon: float = 1e-4):
        """
        Args:
            model: A PeftModel (LoRA-wrapped model). Reference logprobs are computed
                   by disabling the adapter temporarily.
            beta: KL penalty coefficient.
            epsilon: Small constant for numerical stability.
        """
        self.model = model
        self.beta = beta
        self.epsilon = epsilon

    def compute_loss(self, batch: list[dict]) -> GRPOLossOutput:
        """
        Compute GRPO loss for a batch of trajectories.

        Args:
            batch: List of dicts with keys:
                - 'input_ids': Tensor of shape (seq_len,)
                - 'attention_mask': Tensor of shape (seq_len,)
                - 'labels': Tensor of shape (seq_len,) with -100 for prompt tokens
                - 'advantages': Float, the normalized advantage

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
        per_token_loss = F.cross_entropy(
            logits.transpose(1, 2), shifted_labels, reduction="none"
        )

        # 4. Mask out prompt tokens (where labels == -100)
        token_mask = (shifted_labels != -100).float()
        token_log_probs = -per_token_loss * token_mask
        num_completion_tokens = token_mask.sum().int().item()

        # Sum log probs over the completion length per sequence
        completion_log_probs = token_log_probs.sum(dim=1)

        # 5. Forward Pass (Reference Policy) for KL
        # CRITICAL: Use disable_adapter() to get base model logprobs unless provided
        ref_completion_log_probs: torch.Tensor
        if all("reference_logprob" in b for b in batch):
            ref_completion_log_probs = torch.tensor(
                [b["reference_logprob"] for b in batch],
                dtype=torch.float32,
                device=device,
            )
        else:
            with torch.no_grad():
                with self.model.disable_adapter():
                    ref_outputs = self.model(
                        input_ids=input_ids, attention_mask=attention_mask
                    )
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

        # 7. GRPO Loss = -Advantage * LogProb + Beta * KL
        # advantages are already normalized by the trainer
        pg_loss = -(advantages.detach() * completion_log_probs)
        total_loss = pg_loss + (self.beta * kl)

        return GRPOLossOutput(
            loss=total_loss.mean(),
            pg_loss=pg_loss.mean().item(),
            kl=kl.mean().item(),
            mean_completion_logprob=completion_log_probs.mean().item(),
            mean_ref_logprob=ref_completion_log_probs.mean().item(),
            mean_advantage=advantages.mean().item(),
            advantage_std=advantages.std().item() if len(advantages) > 1 else 0.0,
            num_tokens=int(num_completion_tokens),
        )
