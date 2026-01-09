import torch

def BiasWeightedLoss(self, data, step_ndx, logger):
    
    LOSS_BIAS = self.config.get("LOSS_BIAS", 0.8)

    # Data Setup 
    x, y = data
    x = x.to(self.device)
    
    y = y.to(self.device).float().view(-1) 

    self.optimizer.zero_grad()
    
    # Forward Pass 
    embeddings = self.model(x)

    logits = self.classifier(embeddings).view(-1)
    
    yp = torch.sigmoid(logits)
    yt = y
    
    epsilon = 1e-7
    
    # Calculate term for Positives
    # If yt=0, then this entire line will become 0.
    pos_term = -(yt) * torch.log(yp + epsilon)
    
    # Calculate term for Negatives
    # If yt=1, then (1-yt) = 0, so the entire line becomes 0.
    neg_term = -(1 - yt) * torch.log(1 - yp + epsilon)
    
    neg_loss_mean = neg_term.mean()
    pos_loss_mean = pos_term.mean()
    
    total = (LOSS_BIAS * neg_loss_mean) + ((1.0 - LOSS_BIAS) * pos_loss_mean)
    
    # Optimization 
    # History tracking (Optional, for your graph)
    if not hasattr(self, 'state'):
        self.state = {'loss_hist': [], 'fa_ema': 0.0, 'miss_ema': 0.0}
    self.state['loss_hist'].append(total.item())
    if len(self.state['loss_hist']) > 100: self.state['loss_hist'] = self.state['loss_hist'][-100:]
    
    total.backward()
    
    # Gradient Clipping (Standard Safe Practice, not strictly part of logic but recommended)
    grad_norm = torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.classifier.parameters()),
                max_norm=1.0
               )
    
    self.optimizer.step()
    self.scheduler.step()
    
    # Logging 
    with torch.no_grad():
        if logger and step_ndx % 100 == 0:
            is_pos = (yt == 1)
            is_neg = (yt == 0)

            current_lr = self.optimizer.param_groups[0]['lr']

            # Average predictions
            pos_avg = yp[is_pos].mean().item() if is_pos.sum() > 0 else 0.0
            neg_avg = yp[is_neg].mean().item() if is_neg.sum() > 0 else 0.0

            # Calculate FA (False Alarm) and Misses
            FA = (yp[is_neg] > 0.5).float().mean().item() if is_neg.sum() > 0 else 0.0
            Ms = (yp[is_pos] < 0.5).float().mean().item() if is_pos.sum() > 0 else 0.0

            # Positive & Negative Loss
            PosL = pos_term.mean().item()
            NegL = neg_term.mean().item()

            logger.info(
                f"[{step_ndx:5d}] L:{total.item():.6f} "
                f"PL:{PosL:.6f} NL:{NegL:.6f} |PA:{pos_avg:.3f} NA:{neg_avg:.3f} "
                f"|FA:{FA:.3f} Ms:{Ms:.3f} |η:{current_lr:.2e} gNorm:{grad_norm:.3f}"
            )

    per_example_loss = (LOSS_BIAS * neg_term) + ((1.0 - LOSS_BIAS) * pos_term)

    # return total.detach().cpu().item()
    return total, per_example_loss.detach()

