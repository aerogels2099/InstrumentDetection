import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss (Ridnik et al. 2021) for multi-label classification.

    Uses different focusing strengths for positives (gamma_pos) and negatives
    (gamma_neg), plus a probability margin that hard-zeros easy negatives.
    This directly addresses the extreme positive/negative imbalance in
    multi-instrument detection without needing a separate pos_weight tensor.
    """

    def __init__(self, gamma_pos=0, gamma_neg=4, margin=0.05, eps=1e-8):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.margin = margin
        self.eps = eps

    def forward(self, logits, targets):
        # Upcast to float32 so log/clamp are safe under AMP (float16 eps is ~6e-5)
        probs = torch.sigmoid(logits.float())
        targets = targets.float()
        probs_neg = (probs - self.margin).clamp(min=0)

        log_pos = torch.log(probs.clamp(min=self.eps))
        log_neg = torch.log((1 - probs_neg).clamp(min=self.eps))

        w_pos = (1 - probs) ** self.gamma_pos
        w_neg = probs_neg ** self.gamma_neg

        loss = targets * w_pos * log_pos + (1 - targets) * w_neg * log_neg
        return -loss.mean()
