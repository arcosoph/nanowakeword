import torch
            
def TripletMetricLoss(self, data, step_ndx, logger):
    """
    Production-grade wakeword optimization with Enhanced Negative Separation.
    
    Key Features:
    - Contrastive Margin Enforcement (Push negatives FAR away)
    - Temperature Calibration (Prevent overconfident predictions)
    - Dual-Stage Classification (Embedding + Logit verification)
    
    Args:
        data: Tuple of (anchor, positive, negative, labels_anchor, labels_negative)
        step_ndx: Current training step
        logger: Logger object
    
    Returns:
        float: Total loss value
    """

    # Initialize
    if not hasattr(self, 'hard_negative_bank'):
        self.hard_negative_bank = []
        self.fa_rate_ema = 0.0
        self.miss_rate_ema = 0.0
        self.high_confidence_negatives = []  # Track overconfident negatives
    

    strategy_cfg = self.config.get("training_refinement", {})

    # =========================
    # TRIPLET SETTINGS
    # =========================
    triplet_cfg = strategy_cfg.get("triplet_settings", {})

    base_margin          = triplet_cfg.get("base_margin", 0.5)
    curriculum_steps     = triplet_cfg.get("curriculum_steps", 15000)
    hard_mining_ratio    = triplet_cfg.get("hard_mining_ratio", 0.6)
    negative_push_margin = triplet_cfg.get("negative_push_margin", 0.8)

    # =========================
    # LOSS SETTINGS
    # =========================
    loss_cfg = strategy_cfg.get("loss_settings", {})

    fa_penalty   = loss_cfg.get("false_accept_penalty", 20.0)
    miss_penalty = loss_cfg.get("miss_penalty", 5.0)
    focal_gamma  = loss_cfg.get("focal_gamma", 2.5)
    temperature  = loss_cfg.get("temperature", 2.0)

    # =========================
    # MARGIN SETTINGS
    # =========================
    margin_cfg = strategy_cfg.get("margin_settings", {})

    safety_threshold    = margin_cfg.get("safety_threshold", 0.05)
    exp_scale           = margin_cfg.get("exponential_scale", 4.0)
    conf_penalty_weight = margin_cfg.get("confidence_penalty_weight", 50.0)

    # =========================
    # REGULARIZATION
    # =========================
    reg_cfg = strategy_cfg.get("regularization", {})

    spectral_weight = reg_cfg.get("spectral_weight", 0.01)
    enable_spectral = reg_cfg.get("enable_spectral", True)
    conf_reg_weight = reg_cfg.get("confidence_regularization", 0.1)

    # =========================
    # LOSS WEIGHTS
    # =========================
    weight_cfg = strategy_cfg.get("loss_weights", {})

    w_triplet       = weight_cfg.get("triplet", 0.5)
    w_bce           = weight_cfg.get("bce", 0.3)
    w_focal         = weight_cfg.get("focal", 0.2)
    w_margin        = weight_cfg.get("margin", 100.0)
    w_replay        = weight_cfg.get("replay", 20.0)
    w_negative_push = weight_cfg.get("negative_push", 30.0)
    w_conf_reg      = weight_cfg.get("confidence_reg", 10.0)

    # =========================
    # OPTIMIZATION
    # =========================
    opt_cfg = strategy_cfg.get("optimization", {})

    base_grad_norm    = opt_cfg.get("max_grad_norm", 1.5)
    adaptive_clip     = opt_cfg.get("adaptive_clipping", True)
    aggressive_factor = opt_cfg.get("aggressive_clip_factor", 0.5)

    # =========================
    # HARD NEGATIVE MINING
    # =========================
    hnm_cfg = strategy_cfg.get("hard_negative_mining", {})

    hnm_enable        = hnm_cfg.get("enable", True)
    max_bank_size     = hnm_cfg.get("bank_size", 500)
    mining_threshold  = hnm_cfg.get("mining_threshold", 0.3)
    replay_batch_size = hnm_cfg.get("replay_batch_size", 24)
    min_bank_size     = hnm_cfg.get("min_bank_size", 50)
    focus_confusing   = hnm_cfg.get("focus_on_confusing", True)

    # =========================================================
    # 2. DATA PREPARATION
    # =========================================================
    anchor, positive, negative, labels_anchor, labels_negative = data
    
    anchor = anchor.to(self.device)
    positive = positive.to(self.device)
    negative = negative.to(self.device)
    labels_anchor = labels_anchor.to(self.device).float().view(-1)
    labels_negative = labels_negative.to(self.device).float().view(-1)
    
    self.optimizer.zero_grad()
    
    # =========================================================
    # 3. EMBEDDING EXTRACTION with Normalization
    # =========================================================
    emb_anchor = self.model(anchor)
    emb_positive = self.model(positive)
    emb_negative = self.model(negative)
    
    emb_anchor = torch.nn.functional.normalize(emb_anchor, p=2, dim=1)
    emb_positive = torch.nn.functional.normalize(emb_positive, p=2, dim=1)
    emb_negative = torch.nn.functional.normalize(emb_negative, p=2, dim=1)
    
    # =========================================================
    # 4. ENHANCED TRIPLET LOSS with Negative Push
    # =========================================================
    curriculum_factor = min(1.0, step_ndx / curriculum_steps)
    adaptive_margin = base_margin * (0.5 + 0.5 * curriculum_factor)
    
    pos_dist = torch.sum((emb_anchor - emb_positive) ** 2, dim=1)
    neg_dist = torch.sum((emb_anchor - emb_negative) ** 2, dim=1)
    
    triplet_loss_raw = torch.nn.functional.relu(pos_dist - neg_dist + adaptive_margin)
    
    k = max(1, int(len(triplet_loss_raw) * hard_mining_ratio))
    loss_triplet = torch.topk(triplet_loss_raw, k)[0].mean()
    
    # Extra Negative Push Loss
    # Force negatives to be VERY far from positives
    negative_push_loss = torch.nn.functional.relu(negative_push_margin - neg_dist).mean()
    
    # =========================================================
    # 5. CLASSIFICATION with Temperature Calibration
    # =========================================================
    logits_anchor = self.classifier(emb_anchor).view(-1)
    logits_negative = self.classifier(emb_negative).view(-1)
    
    # Apply temperature scaling (prevents overconfidence)
    logits_anchor_calibrated = logits_anchor / temperature
    logits_negative_calibrated = logits_negative / temperature
    
    all_logits = torch.cat([logits_anchor, logits_negative])
    all_logits_calibrated = torch.cat([logits_anchor_calibrated, logits_negative_calibrated])
    all_labels = torch.cat([labels_anchor, labels_negative])
    all_probs = torch.sigmoid(all_logits_calibrated)
    
    # =========================================================
    # 6. ASYMMETRIC WEIGHTED BCE LOSS
    # =========================================================
    bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        all_logits_calibrated, 
        all_labels, 
        reduction='none'
    )
    
    is_negative = (all_labels == 0)
    is_positive = (all_labels == 1)
    
    false_accept_mask = is_negative & (all_probs > 0.5)
    miss_mask = is_positive & (all_probs < 0.5)
    
    loss_weights = torch.ones_like(bce_loss)
    loss_weights[false_accept_mask] *= fa_penalty
    loss_weights[miss_mask] *= miss_penalty
    
    weighted_bce = (bce_loss * loss_weights).mean()
    
    # =========================================================
    # 7. FOCAL LOSS
    # =========================================================
    pt = torch.exp(-bce_loss)
    focal_loss = ((1 - pt) ** focal_gamma * bce_loss * loss_weights).mean()
    
    # =========================================================
    # 8. ENHANCED MARGIN PENALTY with Confidence Control
    # =========================================================
    neg_probs = all_probs[is_negative]
    violators = neg_probs[neg_probs > safety_threshold]
    
    if violators.numel() > 0:
        margin_penalty = (torch.exp(violators * exp_scale) - 1).mean()
        
        # Extra penalty for VERY confident negatives (>0.8)
        overconfident = violators[violators > 0.8]
        if overconfident.numel() > 0:
            confidence_penalty = (overconfident ** 3).mean() * conf_penalty_weight
            margin_penalty = margin_penalty + confidence_penalty
    else:
        margin_penalty = torch.tensor(0.0, device=self.device)
    
    # =========================================================
    # 9. CONFIDENCE REGULARIZATION (Prevent Extreme Scores)
    # =========================================================
    # Penalize predictions too close to 0 or 1
    confidence_reg = torch.tensor(0.0, device=self.device)
    
    # For negatives: penalize if prob > 0.7
    high_conf_negs = neg_probs[neg_probs > 0.7]
    if high_conf_negs.numel() > 0:
        confidence_reg += (high_conf_negs - 0.7).pow(2).mean()
    
    # For positives: encourage confidence but not extreme
    pos_probs = all_probs[is_positive]
    extreme_pos = pos_probs[pos_probs > 0.95]
    if extreme_pos.numel() > 0:
        confidence_reg += (extreme_pos - 0.95).pow(2).mean() * 0.5
    
    # =========================================================
    # 10. ENHANCED HARD NEGATIVE MINING
    # =========================================================
    replay_loss = torch.tensor(0.0, device=self.device)
    
    if hnm_enable:
        with torch.no_grad():
            neg_start_idx = len(labels_anchor)
            neg_probs_only = all_probs[neg_start_idx:]
            
            if focus_confusing:
                # Focus on negatives with probability 0.3-0.7 (most confusing)
                confusing_mask = (neg_probs_only > 0.3) & (neg_probs_only < 0.7)
                hard_neg_indices = confusing_mask.nonzero(as_tuple=True)[0]
            else:
                hard_mask = neg_probs_only > mining_threshold
                hard_neg_indices = hard_mask.nonzero(as_tuple=True)[0]
            
            if len(hard_neg_indices) > 0:
                num_to_store = min(10, len(hard_neg_indices))
                hard_embeddings = emb_negative[hard_neg_indices[:num_to_store]]
                self.hard_negative_bank.extend(hard_embeddings.cpu().tolist())
                
                if len(self.hard_negative_bank) > max_bank_size:
                    self.hard_negative_bank = self.hard_negative_bank[-max_bank_size:]
        
        if len(self.hard_negative_bank) >= min_bank_size:
            import numpy as np
            actual_batch = min(replay_batch_size, len(self.hard_negative_bank))
            bank_sample = np.random.choice(len(self.hard_negative_bank), actual_batch, replace=False)
            
            hard_negs = torch.tensor(
                [self.hard_negative_bank[i] for i in bank_sample],
                device=self.device,
                dtype=emb_anchor.dtype
            )
            hard_negs = torch.nn.functional.normalize(hard_negs, p=2, dim=1)
            
            # Push with stronger margin
            replay_dist = torch.cdist(emb_anchor, hard_negs).min(dim=1)[0]
            replay_loss = torch.nn.functional.relu(0.7 - replay_dist).mean()
    
    # =========================================================
    # 11. SPECTRAL REGULARIZATION
    # =========================================================
    spectral_reg = torch.tensor(0.0, device=self.device)
    
    if enable_spectral:
        try:
            emb_all = torch.cat([emb_anchor, emb_positive, emb_negative], dim=0)
            cov_matrix = torch.mm(emb_all.T, emb_all) / len(emb_all)
            eye_matrix = torch.eye(cov_matrix.size(0), device=self.device)
            spectral_reg = -torch.logdet(cov_matrix + 1e-6 * eye_matrix)
        except:
            spectral_reg = torch.tensor(0.0, device=self.device)
    
    # =========================================================
    # 12. TOTAL LOSS COMBINATION
    # =========================================================
    total_loss = (
        w_triplet * loss_triplet +
        w_bce * weighted_bce +
        w_focal * focal_loss +
        w_margin * margin_penalty +
        w_replay * replay_loss +
        w_negative_push * negative_push_loss +
        w_conf_reg * confidence_reg * conf_reg_weight +
        spectral_weight * spectral_reg
    )
    
    # =========================================================
    # 13. BACKPROPAGATION
    # =========================================================
    total_loss.backward()
    
    if adaptive_clip and self.fa_rate_ema > 0.01:
        clip_value = base_grad_norm * aggressive_factor
    else:
        clip_value = base_grad_norm
    
    torch.nn.utils.clip_grad_norm_(
        list(self.model.parameters()) + list(self.classifier.parameters()),
        max_norm=clip_value
    )
    
    self.optimizer.step()
    self.scheduler.step()
    
    # =========================================================
    # 14. METRICS TRACKING
    # =========================================================
    with torch.no_grad():
        fa_rate = (all_probs[is_negative] > 0.5).float().mean().item()
        miss_rate = (all_probs[is_positive] < 0.5).float().mean().item()
        
        alpha = 0.01
        self.fa_rate_ema = alpha * fa_rate + (1 - alpha) * self.fa_rate_ema
        self.miss_rate_ema = alpha * miss_rate + (1 - alpha) * self.miss_rate_ema
        
        # Track high confidence negatives
        high_conf_neg_count = (neg_probs > 0.8).sum().item()
        
        if logger is not None and step_ndx % 100 == 0:
            logger.info(
                f"Step {step_ndx:5d} | Loss: {total_loss.item():.4f} | "
                f"FA: {fa_rate:.4f} | Miss: {miss_rate:.4f} | "
                f"FA_EMA: {self.fa_rate_ema:.4f} | HighConfNeg: {high_conf_neg_count} | "
                f"Bank: {len(self.hard_negative_bank)}"
            )
    
    return total_loss.detach().cpu().item()

