import torch

def BiasWeightedLoss(logits, labels, LOSS_BIAS, smoothing=0.1):
    
    labels = labels * (1 - smoothing) + 0.5 * smoothing

    yp = torch.sigmoid(logits)
    yt = labels
    
    epsilon = 1e-7

    pos_term = -(yt) * torch.log(torch.clamp(yp, min=epsilon))
    neg_term = -(1 - yt) * torch.log(torch.clamp(1 - yp, min=epsilon))

    pos_mask = (labels > 0.5) 
    neg_mask = (labels <= 0.5)

    pos_loss_mean = pos_term[pos_mask].mean() if pos_mask.sum() > 0 else torch.tensor(0.0, device=logits.device)
    neg_loss_mean = neg_term[neg_mask].mean() if neg_mask.sum() > 0 else torch.tensor(0.0, device=logits.device)
    
    total_loss = (LOSS_BIAS * neg_loss_mean) + ((1.0 - LOSS_BIAS) * pos_loss_mean)

    per_example_loss = (LOSS_BIAS * neg_term) + ((1.0 - LOSS_BIAS) * pos_term)

    return total_loss, per_example_loss.detach()
