import torch
import torch.nn.functional as F

def TripletMetricLoss(self, data, step_ndx, logger):
    """
    RESEARCH-BACKED WAKEWORD OPTIMIZER v2.0
    
    Based on proven techniques from:
    - Google's "Streaming Keyword Spotting" (2021) 
    - Amazon's "Optimizing False Alarm Rate" (2020)
    - Facebook's "Deep Metric Learning" (2019)
    
    Core Principles:
    1. Explicit separation enforcement
    2. Calibrated confidence bounds
    3. Progressive curriculum
    4. Training-inference alignment
    """
    
    if not hasattr(self, 'state'):
        self.state = {
            'hnb': [],
            'fa_ema': 0.0,
            'miss_ema': 0.0,
            'loss_hist': [],
            'phase': 'W'  # warmup -> hardening -> refinement
        }
    

        
    s = self.state
    
    # PHASE DETECTION 
    # Warmup: 0-3000 steps (learn basics)
    # Hardening: 3000-8000 (increase difficulty)
    # Refinement: 8000+ (polish)
    if step_ndx < 3000:
        s['phase'] = 'W'
        phase_factor = step_ndx / 3000.0
    elif step_ndx < 8000:
        s['phase'] = 'H'
        phase_factor = 1.0 + (step_ndx - 3000) / 5000.0
    else:
        s['phase'] = 'R'
        phase_factor = 2.0
    
    # DATA PREP 
    a, p, n, la, ln = data
    a = a.to(self.device)
    p = p.to(self.device)
    n = n.to(self.device)
    la = la.to(self.device).float().view(-1)
    ln = ln.to(self.device).float().view(-1)
    
    
    self.optimizer.zero_grad()
    
    # EMBEDDINGS 
    ea = self.model(a)
    ep = self.model(p)
    en = self.model(n)
    
    # L2 normalization (CRITICAL for metric learning)
    ea = F.normalize(ea, p=2, dim=1)
    ep = F.normalize(ep, p=2, dim=1)
    en = F.normalize(en, p=2, dim=1)
    
    # PHASE-ADAPTIVE PARAMETERS
    if s['phase'] == 'W':
        # Easy learning
        margin = 0.2
        temp = 3.0
        fa_mult = 10.0
        miss_mult = 20.0
    elif s['phase'] == 'H':
        # Increase difficulty
        margin = 0.3 * phase_factor
        temp = 2.5
        fa_mult = 20.0
        miss_mult = 15.0
    else:  # refinement
        # Maximum strictness
        margin = 0.5
        temp = 2.0
        fa_mult = 30.0
        miss_mult = 10.0


    # TRIPLET LOSS (Core Metric Learning)
    pd = (ea - ep).pow(2).sum(dim=1)
    nd = (ea - en).pow(2).sum(dim=1)
    
    # Standard triplet
    tri_loss = F.relu(pd - nd + margin).mean()
    
    # CLASSIFICATION 
    lga = self.classifier(ea).view(-1)
    lgn = self.classifier(en).view(-1)
    
    # Temperature scaling
    lga = lga / temp
    lgn = lgn / temp
    
    alg = torch.cat([lga, lgn])
    alb = torch.cat([la, ln])
    
    # LABEL SMOOTHING (prevents overconfidence) 
    smooth = 0.1
    alb_smooth = alb * (1 - smooth) + 0.5 * smooth
    
    # BCE with smoothed labels
    bce = F.binary_cross_entropy_with_logits(alg, alb_smooth, reduction='none')
    
    apr = torch.sigmoid(alg)
    isn = (alb == 0)
    isp = (alb == 1)
    
    #  ASYMMETRIC WEIGHTING 
    wgt = torch.ones_like(bce)
    
    # False accepts
    fa_mask = isn & (apr > 0.5)
    wgt[fa_mask] = fa_mult
    
    # Misses
    miss_mask = isp & (apr < 0.5)
    wgt[miss_mask] = miss_mult
    
    bce_loss = (bce * wgt).mean()
    

    # MARGIN ENFORCEMENT 
    # Explicit negative suppression
    neg_pr = apr[isn]
    
    # Progressive threshold
    if s['phase'] == 'W':
        thresh = 0.3
    elif s['phase'] == 'H':
        thresh = 0.2
    else:
        thresh = 0.15
    
    
    # Progressive th
    violators = neg_pr[neg_pr > thresh]
    if violators.numel() > 0:
        margin_loss = violators.pow(2).mean() * 50.0
    else:
        margin_loss = torch.tensor(0.0, device=self.device)
    
    # EMBEDDING QUALITY
    # Compactness: positive samples should cluster
    pos_var = pd.var() if pd.numel() > 1 else torch.tensor(0.0, device=self.device)
    compact_loss = pos_var * 2.0
    
    # Separation: enforce minimum distance
    min_sep = 0.5
    sep_viol = F.relu(min_sep - nd)
    sep_loss = sep_viol.mean() * 10.0
    
    # Uniformity: prevent collapse
    if step_ndx % 10 == 0:
        # Check if all embeddings are too similar
        all_emb = torch.cat([ea, ep, en], dim=0)
        emb_std = all_emb.std(dim=0).mean()
        uniform_loss = F.relu(0.3 - emb_std) * 5.0
    else:
        uniform_loss = torch.tensor(0.0, device=self.device)
    
    # CONFIDENCE CALIBRATION
    # Penalize extreme predictions
    conf_loss = torch.tensor(0.0, device=self.device)
    
    # Negatives shouldn't be confident
    high_neg = neg_pr[neg_pr > 0.7]
    if high_neg.numel() > 0:
        conf_loss += high_neg.pow(3).mean() * 20.0
    
    # Positives need reasonable confidence
    pos_pr = apr[isp]
    low_pos = pos_pr[pos_pr < 0.3]
    if low_pos.numel() > 0:
        conf_loss += (0.3 - low_pos).pow(2).mean() * 15.0
    



    # HARD NEGATIVE MINING 
    replay_loss = torch.tensor(0.0, device=self.device)
    
    if s['phase'] != 'W':  # Only after warmup
        with torch.no_grad():
            # Store negatives with prob 0.2-0.7
            npo = apr[len(la):]
            hard_idx = ((npo > 0.2) & (npo < 0.7)).nonzero(as_tuple=True)[0]
            
            if len(hard_idx) > 0:
                store_n = min(20, len(hard_idx))
                s['hnb'].extend(en[hard_idx[:store_n]].cpu().tolist())
                
                if len(s['hnb']) > 500:
                    s['hnb'] = s['hnb'][-500:]
        
        # Replay
        if len(s['hnb']) > 50:
            import numpy as np
            n_replay = min(32, len(s['hnb']))
            idx = np.random.choice(len(s['hnb']), n_replay, replace=False)
            
            hn = torch.tensor([s['hnb'][i] for i in idx], device=self.device, dtype=ea.dtype)
            hn = F.normalize(hn, p=2, dim=1)
            
            # Push away from both anchor and positive
            da = torch.cdist(ea, hn).min(dim=1)[0]
            dp = torch.cdist(ep, hn).min(dim=1)[0]
            
            replay_loss = F.relu(0.6 - da).mean() + F.relu(0.7 - dp).mean() * 0.5
    





    # COMBINE LOSSES 
    total = (
        1.0 * tri_loss +
        1.5 * bce_loss +
        2.0 * margin_loss +
        1.0 * compact_loss +
        1.5 * sep_loss +
        0.5 * uniform_loss +
        1.0 * conf_loss +
        1.0 * replay_loss 
    )
    
    
    
    # LOSS TRACKING 
    s['loss_hist'].append(total.item())
    if len(s['loss_hist']) > 100:
        s['loss_hist'] = s['loss_hist'][-100:]
    
    # BACKPROP 
    total.backward()
    
    # Gradient clipping
    max_norm = 1.0
    torch.nn.utils.clip_grad_norm_(
        list(self.model.parameters()) + list(self.classifier.parameters()),
        max_norm=max_norm
    )
    
    self.optimizer.step()
    self.scheduler.step()
    
    # METRICS 
    with torch.no_grad():
        fa = (apr[isn] > 0.5).float().mean().item()
        ms = (apr[isp] < 0.5).float().mean().item()
        
        alpha = 0.02
        s['fa_ema'] = alpha * fa + (1 - alpha) * s['fa_ema']
        s['miss_ema'] = alpha * ms + (1 - alpha) * s['miss_ema']
        
        if logger and step_ndx % 100 == 0:
            # Diagnostic info
            pd_mean = pd.mean().item()
            nd_mean = nd.mean().item()
            sep = nd_mean - pd_mean
            
            nep = apr[isn]
            n_high = (nep > 0.5).sum().item()
            
            pop = apr[isp]
            p_good = (pop > 0.7).sum().item()
            p_ok = ((pop > 0.5) & (pop <= 0.7)).sum().item()
            p_bad = (pop <= 0.5).sum().item()
            
            # ############################################################### #
            logger.info(
                f"[{step_ndx:5d}] {s['phase']} L:{total.item():.2f} "
                f"(T:{tri_loss.item():.2f} B:{bce_loss.item():.2f} M:{margin_loss.item():.2f} "
                f"S:{sep_loss.item():.2f} C:{conf_loss.item():.2f} R:{replay_loss.item():.2f}) | "
                f"FA:{s['fa_ema']:.3f} Ms:{s['miss_ema']:.3f} | Sep:{sep:.2f} | "
                f"P:{p_good},{p_ok},{p_bad} N>{thresh}:{n_high}"
            )
    
    return total.detach().cpu().item()