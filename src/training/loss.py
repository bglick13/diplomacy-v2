import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


class GRPOLoss:
    """
    Implements Group Relative Policy Optimization loss manually
    since we generate data externally.
    """

    def __init__(self, model, ref_model, beta=0.04, epsilon=1e-4):
        self.model = model
        self.ref_model = ref_model  # Usually the same model with LoRA disabled
        self.beta = beta
        self.epsilon = epsilon

    def compute_loss(self, batch):
        """
        batch: List of dicts {
            'input_ids': Tensor,
            'attention_mask': Tensor,
            'labels': Tensor (Masked prompt),
            'advantages': Float
        }
        """
        # 1. Stack Batch
        input_ids = pad_sequence(
            [b["input_ids"] for b in batch], batch_first=True, padding_value=0
        )
        attention_mask = pad_sequence(
            [b["attention_mask"] for b in batch], batch_first=True, padding_value=0
        )
        labels = pad_sequence(
            [b["labels"] for b in batch], batch_first=True, padding_value=-100
        )
        advantages = torch.tensor(
            [b["advantages"] for b in batch], device=self.model.device
        )

        # 2. Forward Pass (Current Policy)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]  # Shift left
        labels = labels[:, 1:]  # Shift right

        # 3. Compute Token LogProbs
        # CrossEntropy with reduction='none' gives us log-probs per token
        per_token_loss = F.cross_entropy(
            logits.transpose(1, 2), labels, reduction="none"
        )

        # 4. Mask out prompt tokens (where labels == -100)
        token_mask = (labels != -100).float()
        token_log_probs = -per_token_loss * token_mask

        # Sum log probs over the completion length
        completion_log_probs = token_log_probs.sum(dim=1)

        # 5. Forward Pass (Reference Policy) for KL
        # We do this in a no_grad block to save memory
        with torch.no_grad():
            ref_outputs = self.ref_model(
                input_ids=input_ids, attention_mask=attention_mask
            )
            ref_logits = ref_outputs.logits[:, :-1, :]

            # Compute Reference LogProbs
            ref_per_token_loss = F.cross_entropy(
                ref_logits.transpose(1, 2), labels, reduction="none"
            )
            ref_token_log_probs = -ref_per_token_loss * token_mask
            ref_completion_log_probs = ref_token_log_probs.sum(dim=1)

        # 6. KL Divergence Approximation (http://joschu.net/blog/kl-approx.html)
        # ratio = log(pi) - log(ref)
        log_ratio = completion_log_probs - ref_completion_log_probs
        kl = torch.exp(log_ratio) - 1 - log_ratio  # Schuman's estimator

        # 7. GRPO Loss
        # Loss = - (Advantage * LogProb) + (Beta * KL)
        # Note: We detach advantage to treat it as a scalar weight
        pg_loss = -(advantages.detach() * completion_log_probs)
        total_loss = pg_loss + (self.beta * kl)

        return total_loss.mean(), pg_loss.mean().item(), kl.mean().item()
