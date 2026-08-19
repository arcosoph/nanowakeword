import torch

def BiasWeightedLoss(logits, labels, LOSS_BIAS, smoothing=0.05):
    """
    Asymmetric binary cross-entropy loss with class weighting and label smoothing.

    The masks are computed from the ORIGINAL (hard) labels before smoothing is
    applied, so the pos/neg split is always correct regardless of smoothing value.

    Args:
        logits:    Raw model outputs (before sigmoid), shape [B].
        labels:    Hard binary labels (0.0 or 1.0), shape [B].
        LOSS_BIAS: Weight given to the NEGATIVE class loss term.
                   0.9 means negatives contribute 90% of the total loss.
        smoothing: Label smoothing factor. Applied only to the soft targets
                   used in the cross-entropy calculation, not to the masks.
    """
    # Build masks from HARD labels BEFORE any smoothing
    pos_mask = (labels > 0.5)
    neg_mask = ~pos_mask

    # Apply label smoothing to soft targets only
    soft_labels = labels * (1.0 - smoothing) + 0.5 * smoothing

    yp = torch.sigmoid(logits)
    yt = soft_labels

    epsilon = 1e-7
    pos_term = -yt * torch.log(torch.clamp(yp, min=epsilon))
    neg_term = -(1.0 - yt) * torch.log(torch.clamp(1.0 - yp, min=epsilon))

    pos_loss_mean = pos_term[pos_mask].mean() if pos_mask.sum() > 0 else torch.tensor(0.0, device=logits.device)
    neg_loss_mean = neg_term[neg_mask].mean() if neg_mask.sum() > 0 else torch.tensor(0.0, device=logits.device)

    total_loss = (LOSS_BIAS * neg_loss_mean) + ((1.0 - LOSS_BIAS) * pos_loss_mean)

    # Per-example loss for curriculum hardness tracking (uses hard masks for correct weighting)
    per_example_loss = torch.where(pos_mask, (1.0 - LOSS_BIAS) * pos_term, LOSS_BIAS * neg_term)

    return total_loss, per_example_loss.detach()
