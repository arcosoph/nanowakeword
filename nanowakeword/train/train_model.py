# ==============================================================================
#  NanoWakeWord: Lightweight, Intelligent Wake Word Detection
#  Copyright 2025 Arcosoph. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#  Project: https://github.com/arcosoph/nanowakeword
# ==============================================================================

# (✿◕‿◕✿)
import os
import re 
import copy
import torch
import random
import logging
import warnings
import numpy as np
import collections
from tqdm import tqdm
from logging.handlers import RotatingFileHandler
from collections import deque

from nanowakeword.modules.loss import BiasWeightedLoss
from nanowakeword.modules.model import Model
from nanowakeword.utils.logger import print_info, print_key_value, print_final_report_header


# To make the terminal look clean
warnings.filterwarnings("ignore")
logging.getLogger("torchaudio").setLevel(logging.ERROR)

SEED=10
def set_seed(seed):
    """
    This function sets the seed to make the training results reliable.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(SEED)


class Trainer:
    def __init__(self, model: Model, config: dict):
        """
        The Trainer takes a model object and the config to initialize.
        """
        self.model = model
        self.config = config
        
        # All state-related attributes now belong to the Trainer
        self.optimizer = None
        self.scheduler = None
        self.history = {} # Or defaultdict
        self.best_training_checkpoints = []
        
        # Setup optimizer and scheduler here
        self._setup_optimizer_and_scheduler()




    def _setup_optimizer_and_scheduler(self):
            """
            Sets up the optimizer and the learning rate scheduler based on the
            provided configuration. Supports multiple scheduler types.
            """
            from itertools import chain

            all_params = chain(self.model.parameters(), self.model.classifier.parameters())

            optimizer_type = self.config.get("optimizer_type", "adamw").lower() 
            learning_rate = self.config.get('learning_rate_max', 1e-4)
            weight_decay = self.config.get("weight_decay", 1e-2)
            momentum = self.config.get("momentum", 0.9)

            if optimizer_type == "adam":
                self.optimizer = torch.optim.Adam(all_params, lr=learning_rate, weight_decay=weight_decay)
            elif optimizer_type == "sgd":
                self.optimizer = torch.optim.SGD(all_params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
            else: # Default to AdamW
                self.optimizer = torch.optim.AdamW(all_params, lr=learning_rate, weight_decay=weight_decay)
            
            print_info(f"Using optimizer: {optimizer_type.upper()}")


            #  Scheduler Setup 
            # Get the scheduler type from config, defaulting to 'onecycle' for backward compatibility.
            scheduler_type = self.config.get('lr_scheduler_type', 'onecycle').lower()
            
            print_info(f"Setting up learning rate scheduler: {scheduler_type.upper()}")

            if scheduler_type == 'cyclic':
                # This is your original, powerful CyclicLR setup
                self.scheduler = torch.optim.lr_scheduler.CyclicLR(
                    self.optimizer,
                    base_lr=self.config['learning_rate_base'],
                    max_lr=self.config['learning_rate_max'],
                    step_size_up=self.config['clr_step_size_up'],
                    step_size_down=self.config.get("clr_step_size_down", self.config['clr_step_size_up']),
                    mode='triangular2', # or 'triangular' 
                    cycle_momentum=False
                )
        
            elif scheduler_type == 'onecycle':
                # OneCycleLR is another very powerful scheduler, great for fast convergence.
                # It requires the maximum learning rate and total training steps.
                self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    self.optimizer,
                    max_lr=self.config['learning_rate_max'],
                    total_steps=self.config['steps']
                )

            elif scheduler_type == 'cosine':
                # CosineAnnealingLR smoothly decreases the learning rate in a cosine curve.
                # It only requires the total number of training steps (T_max).
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.config['steps'],
                    eta_min=self.config.get('learning_rate_base', 1e-6) # Use base_lr as the minimum
                )

            else:
                raise ValueError(
                    f"Unsupported lr_scheduler_type: '{scheduler_type}'. "
                    "Supported types are: 'cyclic', 'onecycle', 'cosine'."
                )


    def validate(self, val_loader):
        """
        Runs a validation cycle and returns key performance metrics.

        Uses a threshold sweep to find the operating point that minimises the
        weighted error score (miss_weight * FN + fp_weight * FP).

        For speed, only a random subsample of the validation set is used per
        call (controlled by val_subsample_batches). The full set is used for
        the final evaluation at the end of training.
        """
        import collections
        import torch.nn.functional as F

        self.model.eval()
        self.model.classifier.eval()

        all_logits = []
        all_labels = []

        # Subsample validation for speed: only evaluate on N random batches
        # per validation call during training. Set to 0 to use the full set.
        max_val_batches = self.config.get("val_subsample_batches", 0)

        with torch.no_grad():
            for batch_idx, (features, labels, _) in enumerate(val_loader):
                if max_val_batches > 0 and batch_idx >= max_val_batches:
                    break
                features = features.to(self.model.device)
                labels = labels.to(self.model.device)

                logits = self.model(features).squeeze()

                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())

        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels).float()

        val_loss = F.binary_cross_entropy_with_logits(all_logits, all_labels).item()

        # Threshold sweep to find the best operating point 
        # We sweep over sigmoid probabilities from 0.1 to 0.9 and pick the
        # threshold that minimises the weighted error score.
        # miss_weight > fp_weight means we penalise missed wake words more.
        miss_weight = float(self.config.get("val_miss_weight", 4.0))
        fp_weight   = float(self.config.get("val_fp_weight",   1.0))

        probs = torch.sigmoid(all_logits)
        # Sweep from 0.2 to 0.8 -- avoid extremes that indicate a degenerate model.
        # A threshold of 0.10 means the model barely separates classes; real-world
        # inference typically uses 0.5+. Keeping the floor at 0.2 prevents the
        # sweep from selecting a threshold that only works on easy validation data.
        thresholds = torch.linspace(0.2, 0.8, steps=13)

        best_error  = float('inf')
        best_thresh = 0.5
        best_tp = best_tn = best_fp = best_fn = 0

        for thresh in thresholds:
            preds = (probs >= thresh).float()
            tp = ((preds == 1) & (all_labels == 1)).sum().item()
            tn = ((preds == 0) & (all_labels == 0)).sum().item()
            fp = ((preds == 1) & (all_labels == 0)).sum().item()
            fn = ((preds == 0) & (all_labels == 1)).sum().item()
            weighted_err = miss_weight * fn + fp_weight * fp
            if weighted_err < best_error:
                best_error  = weighted_err
                best_thresh = thresh.item()
                best_tp, best_tn, best_fp, best_fn = tp, tn, fp, fn

        val_recall = best_tp / (best_tp + best_fn) if (best_tp + best_fn) > 0 else 0.0
        val_fpr    = best_fp / (best_fp + best_tn) if (best_fp + best_tn) > 0 else 0.0

        # Raw (unweighted) error count for logging
        raw_error_score = best_fp + best_fn

        metrics = collections.OrderedDict()
        metrics['val_loss']           = val_loss
        metrics['val_recall']         = val_recall
        metrics['val_fpr']            = val_fpr
        metrics['total_false_alarms'] = best_fp
        metrics['total_misses']       = best_fn
        metrics['error_score']        = best_error          # weighted -- used for checkpointing
        metrics['raw_error_score']    = raw_error_score     # unweighted -- for display
        metrics['best_threshold']     = best_thresh

        self.model.train()
        self.model.classifier.train()

        return metrics


    def auto_train(self, X_train, X_val, steps, table_updater, debug_path, resume_from_dir=None):
            """
            A modern, single-sequence training process that combines the best checkpoints to
                create a final and robust model.
            """

            self.train_model(
                X=X_train,
                X_val=X_val,
                max_steps=steps,
                log_path=debug_path,
                table_updater=table_updater,
                resume_from_dir=resume_from_dir        
            )

            print_info("Training finished. Building final model...")


            final_model = copy.deepcopy(self.model)

            # Prefer the model that achieved the best WEIGHTED validation error score.
            # IMPORTANT: If validation data overlaps with training data (a common
            # misconfiguration), the validation score is meaningless -- the model will
            # always achieve 0 errors on data it was trained on. In that case, the
            # training-loss checkpoint pool (which tracks EMA loss over time) is a
            # better signal because it captures the model BEFORE it fully memorised
            # the training set.
            #
            # We detect this by checking if best_error_score == 0.0 AND the best
            # checkpoint was found very late in training (after 80% of steps). A
            # legitimate validation set should show improvement early, not only at
            # the very end when the model has memorised everything.
            val_checkpoint_is_suspicious = (
                self.best_error_score == 0.0
                and self.best_model_on_error_score is not None
            )

            if self.best_model_on_error_score is not None and not val_checkpoint_is_suspicious:
                print_info("Using best validation-error-score checkpoint as the final model.")
                final_model.load_state_dict(self.best_model_on_error_score)
            elif self.best_training_checkpoints:
                if val_checkpoint_is_suspicious:
                    print_info(
                        "WARNING: Validation achieved 0 errors - this likely means your "
                        "validation set overlaps with training data. "
                        "Using training-loss checkpoint averaging instead, which is more "
                        "reliable when val data = train data."
                    )
                else:
                    print_info("No validation data used. Averaging top training-loss checkpoints.")
                avg_state = final_model.average_models(self.best_training_checkpoints)
                final_model.load_state_dict(avg_state)
            else:
                print_info("No checkpoints available. Using the model at the end of training.")

            final_model.eval()

            print_info("Calculating performance metrics for the final model...")
                        
            final_results = collections.OrderedDict()

            if self.best_training_scores:
                avg_stable_loss = np.mean([score['stable_loss'] for score in self.best_training_scores])
                final_results["Average Stable Loss"] = f"{avg_stable_loss:.4f}"
            else:
                final_results["Average Stable Loss"] = "N/A"
            
            try:
                with torch.no_grad():
                    # Get a sample batch, which now contains features, labels, and indices
                    x_batch, y_batch, _ = next(iter(X_train)) # Unpack all three, but ignore the index
                    
                    confidence_batch_x = x_batch.to(self.model.device)
                    confidence_batch_y = y_batch.to(self.model.device)

                    predictions = final_model(confidence_batch_x).squeeze()
                    
                    pos_preds = predictions[confidence_batch_y.squeeze() == 1]
                    neg_preds = predictions[confidence_batch_y.squeeze() == 0]

                    # Ensure we have predictions to average to avoid NaN errors
                    if pos_preds.numel() > 0:
                        final_results["Avg. Positive Score (Logit)"] = f"{pos_preds.mean().item():.3f}"
                    else:
                        final_results["Avg. Positive Score (Logit)"] = "N/A (No positives in batch)"

                    if neg_preds.numel() > 0:
                        final_results["Avg. Negative Score (Logit)"] = f"{neg_preds.mean().item():.3f}"
                    else:
                        final_results["Avg. Negative Score (Logit)"] = "N/A (No negatives in batch)"

            except (StopIteration, RuntimeError) as e:
                final_results["Confidence Score"] = f"N/A (Error: {e})"

            print_final_report_header()
            print_info("NOTE: These metrics are indicators of model health, not real-world performance.")

            for key, value in final_results.items():
                print_key_value(key, value)

            self.model.history["final_report"] = final_results

            # Returning the completed and averaged model
            return final_model


    def train_model(self, X, X_val, max_steps, log_path, table_updater, resume_from_dir=None):
        
        import itertools 

        debug_mode = self.config.get("debug_mode", False)
        log_dir = os.path.join(log_path, "training_debug")
        os.makedirs(log_dir, exist_ok=True)
        debug_log_file = os.path.join(log_dir, "training_debug.log")
        if debug_mode:
            logger = logging.getLogger("NanoTrainerDebug")
            logger.setLevel(logging.INFO)
            if not logger.handlers:
                handler = RotatingFileHandler(
                    debug_log_file, 
                    maxBytes=5_000_000, 
                    backupCount=30,
                    encoding='utf-8'  
                )
                formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
                handler.setFormatter(formatter)
                logger.addHandler(handler)
            logger.propagate = False
            print_info(f"Debug mode ON. Logs will be saved to:\n{debug_log_file}")
        else:
            logger = logging.getLogger("NanoTrainerDebug")
            logger.disabled = True


        checkpoint_cfg = self.config.get("checkpointing", {})
        checkpointing_enabled = checkpoint_cfg.get("enabled", False)
        checkpoint_interval = checkpoint_cfg.get("interval_steps", 1000)
        checkpoint_limit = checkpoint_cfg.get("limit", 3)
        checkpoint_dir = os.path.join(log_path, "checkpoints")
        if checkpointing_enabled:
            os.makedirs(checkpoint_dir, exist_ok=True)
            print_info(f"Checkpointing is ENABLED. A checkpoint will be saved every {checkpoint_interval} steps.")

        ema_loss = None
        self.best_training_checkpoints = [] 
        self.best_training_scores = []
        checkpoint_averaging_top_k= self.config.get("checkpoint_averaging_top_k", 5)

        smoothing_window_size = self.config.get("validation_smoothing_window", 3)
        self.recent_val_scores = deque(maxlen=smoothing_window_size)
        
        self.best_smoothed_val_loss = float('inf')
        self.best_model_checkpoint = None 

        self.best_error_score = float('inf')
        self.best_model_on_error_score = None

        # Stabilization: use a smaller fraction so validation starts earlier.
        # The old default of 15% meant validation never ran until step ~7500 on a
        # 50k run -- by then the model had already memorised the training set.
        default_stb_steps = int(max_steps * 0.05)
        stabilization_steps = self.config.get("stabilization_steps", default_stb_steps)
        min_delta = self.config.get("min_delta", 0.0001)
        best_ema_loss_for_stopping = float('inf')
        steps_without_improvement = 0
        ema_alpha = self.config.get("ema_alpha", 0.01)

        # Early stopping patience: default to 10% of steps (was 15%).
        # Also track the VALIDATION error score for stopping, not just training EMA loss,
        # when a val loader is available.
        default_patience_steps = int(max_steps * 0.10)
        user_patience = self.config.get("early_stopping_patience", None)
        if user_patience is not None:
            patience = user_patience
        elif self.config.get("steps", max_steps) < 3000:
            patience = 0
        else:
            patience = default_patience_steps

        # Validation-based early stopping state
        val_patience_steps = self.config.get("val_early_stopping_patience", int(max_steps * 0.15))
        val_steps_without_improvement = 0

        if patience == 0:
            print_info("Early stopping is DISABLED. Training will run for the full duration of 'steps'.")
        else:
            print_info(f"Training for {max_steps} steps. Early stopping will activate after {stabilization_steps} stabilization steps.")

        self.model.to(self.model.device)
        self.model.train() 
        self.model.classifier.train() 

        start_step = 0
        data_iterator = iter(itertools.cycle(X))

        if resume_from_dir:
                    resume_checkpoint_dir = os.path.join(resume_from_dir, "training_artifacts", "checkpoints")
                    print_info(f"Attempting to resume training from: {resume_checkpoint_dir}")
                    if os.path.exists(resume_checkpoint_dir):
                        checkpoints = [f for f in os.listdir(resume_checkpoint_dir) if f.startswith("checkpoint_step_") and f.endswith(".pth")]
                        if checkpoints:
                            latest_step = -1
                            latest_checkpoint_file = None
                            for cp_file in checkpoints:
                                match = re.search(r"checkpoint_step_(\d+).pth", cp_file)
                                if match:
                                    step = int(match.group(1))
                                    if step > latest_step:
                                        latest_step = step
                                        latest_checkpoint_file = cp_file
                            
                            if latest_checkpoint_file:
                                checkpoint_path = os.path.join(resume_checkpoint_dir, latest_checkpoint_file)
                                print_info(f"Loading latest checkpoint: {checkpoint_path}")
                                checkpoint = torch.load(checkpoint_path, map_location=self.model.device, weights_only=False)
                                
                                # Model, Optimizer & Scheduler State
                                self.model.load_state_dict(checkpoint['model_state_dict'])
                                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                                
                                # Training States
                                start_step = checkpoint.get('step', 0) + 1 # +1 to start from next step
                                ema_loss = checkpoint.get('ema_loss', None)
                                steps_without_improvement = checkpoint.get('steps_without_improvement', 0)
                                best_ema_loss_for_stopping = checkpoint.get('best_ema_loss_for_stopping', float('inf'))
                                
                                # Validation & History States (Crucial for auto_train final model)
                                self.model.history = checkpoint.get('model_history', self.model.history)
                                self.best_error_score = checkpoint.get('best_error_score', float('inf'))
                                self.best_model_on_error_score = checkpoint.get('best_model_on_error_score', None)
                                self.best_training_checkpoints = checkpoint.get('best_training_checkpoints', [])
                                self.best_training_scores = checkpoint.get('best_training_scores', [])
                                
                                # RNG States (for deterministic resume)
                                if 'torch_rng_state' in checkpoint:
                                    torch.set_rng_state(checkpoint['torch_rng_state'])
                                if 'torch_cuda_rng_state' in checkpoint and torch.cuda.is_available():
                                    torch.cuda.set_rng_state_all(checkpoint['torch_cuda_rng_state'])
                                if 'np_rng_state' in checkpoint:
                                    np.random.set_state(checkpoint['np_rng_state'])
                                if 'random_rng_state' in checkpoint:
                                    random.setstate(checkpoint['random_rng_state'])

                                print_info(f"Successfully restored state. Resuming training from step {start_step}.")
                                
                            else:
                                print_info("WARNING: Checkpoint files found, but names are invalid. Starting fresh.")
                        else:
                            print_info("WARNING: No valid checkpoint files found. Starting fresh.")
                    else:
                        print_info(f"WARNING: Checkpoint directory not found at '{resume_checkpoint_dir}'. Starting fresh.")

        table_updater.update(force_print=True)
    
        training_loop = tqdm(data_iterator, total=max_steps, desc="Training", initial=start_step)
        for step_ndx, data in enumerate(training_loop, start=start_step):
            
            self.model.train()
            features, labels, indices = data
            features = features.to(self.model.device)
            labels = labels.to(self.model.device).float().view(-1)

            self.optimizer.zero_grad()
            
            embeddings = self.model.model(features)
            logits = self.model.classifier(embeddings).view(-1)

            LOSS_BIAS = float(self.config.get("LOSS_BIAS", 0.75))
            loss_fn_type = self.config.get("loss_function", "bias_weighted").lower()

            if loss_fn_type == "asymmetric_focal":
                gamma_pos = self.config.get("afl_gamma_pos", 0.0)
                gamma_neg = self.config.get("afl_gamma_neg", 4.0)
                # total_loss, per_example_loss = AsymmetricFocalLoss(
                #     logits, labels, LOSS_BIAS,
                #     gamma_pos=gamma_pos, gamma_neg=gamma_neg
                # )
            else:  # default: bias_weighted
                total_loss, per_example_loss = BiasWeightedLoss(logits, labels, LOSS_BIAS)

            # Logit regularisation: penalise extreme logit magnitudes.
            # ASYMMETRIC: only penalise negative logits that go too negative.
            # Positive logits are left free -- the model needs high confidence on
            # positives to survive real-world noise. Pulling them down causes misses.
            # Negative logits going to -25 is the problem: it means the model has
            # zero uncertainty on negatives and will fail on any out-of-distribution
            # audio (like partial wake words). Keeping them above -logit_reg_margin
            # forces the model to maintain a calibrated decision boundary.
            logit_reg_weight = float(self.config.get("logit_reg_weight", 2e-4))
            logit_reg_margin = float(self.config.get("logit_reg_margin", 6.0))
            if logit_reg_weight > 0:
                # Penalise BOTH positive logits that are too high AND negative logits
                # that are too negative. Both extremes cause real-world failure:
                # - Positive logits too high (+10): model has no margin for noisy audio
                # - Negative logits too low (-10): model has zero uncertainty on negatives,
                #   so out-of-distribution audio (partial wake words) gets high scores
                # Target range: positives in [+3, +margin], negatives in [-margin, -3]
                pos_mask_reg = (labels >= 0.5)
                neg_mask_reg = ~pos_mask_reg

                reg_loss = torch.tensor(0.0, device=logits.device)
                if pos_mask_reg.sum() > 0:
                    pos_logits = logits[pos_mask_reg]
                    excess_pos = torch.clamp(pos_logits - logit_reg_margin, min=0.0)
                    reg_loss = reg_loss + (excess_pos ** 2).mean()
                if neg_mask_reg.sum() > 0:
                    neg_logits = logits[neg_mask_reg]
                    excess_neg = torch.clamp(-neg_logits - logit_reg_margin, min=0.0)
                    reg_loss = reg_loss + (excess_neg ** 2).mean()

                total_loss = total_loss + logit_reg_weight * reg_loss

            total_loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                        list(self.model.parameters()) + list(self.model.classifier.parameters()),
                        max_norm=1.0
                    )

            self.optimizer.step()
            self.scheduler.step()

            # Live Feedback Loop, Update Hardness Scores
            # Using Exponential Moving Average (EMA) for stable updates.
            # A smaller alpha (0.05) gives smoother, more stable hardness estimates
            # compared to the previous 0.1, reducing noise-driven over-sampling.
            hardness_alpha = self.config.get("hardness_ema_alpha", 0.05)

            # Get the old hardness scores for the samples in this batch
            old_hardness = X.dataset.sample_hardness[indices].to(self.model.device)

            # Use raw per-example BCE (not the class-weighted version) for hardness,
            # so that positive and negative samples are compared on the same scale.
            with torch.no_grad():
                raw_bce = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits.detach(), labels, reduction='none'
                )

            new_hardness = hardness_alpha * raw_bce + (1.0 - hardness_alpha) * old_hardness

            # Hardness floor: prevent scores from collapsing to zero.
            # When the model memorises training data, all scores approach 0 and the
            # curriculum sampler loses its diversity signal entirely. A floor of 0.05
            # ensures every sample retains a minimum chance of being selected.
            hardness_floor = self.config.get("hardness_floor", 0.05)
            new_hardness = torch.clamp(new_hardness, min=hardness_floor)

            X.dataset.sample_hardness[indices] = new_hardness.cpu()

            # Periodic hardness reset: every N steps, partially reset hardness scores
            # toward 1.0 to re-inject diversity and prevent the curriculum from
            # permanently ignoring samples the model learned early.
            hardness_reset_interval = self.config.get("hardness_reset_interval", 5000)
            if hardness_reset_interval > 0 and step_ndx > 0 and step_ndx % hardness_reset_interval == 0:
                decay = self.config.get("hardness_reset_decay", 0.5)
                X.dataset.sample_hardness.mul_(decay).add_(1.0 - decay)
                if debug_mode:
                    logger.info(f"[{step_ndx:5d}] Hardness scores partially reset (decay={decay}).")

            # Use the 'total_loss' for history and EMA calculation
            current_loss = total_loss.detach().cpu().item()

            if current_loss is not None:
                self.model.history["loss"].append(current_loss)
                
                if ema_loss is None: 
                    ema_loss = current_loss
                ema_loss = ema_alpha * current_loss + (1 - ema_alpha) * ema_loss

                # Training-loss checkpoint pool: only update every N steps to avoid
                # the cost of copy.deepcopy on every single training step.
                # The pool is only used as a fallback when no validation data exists.
                checkpoint_pool_interval = self.config.get("checkpoint_pool_interval", 500)
                if (step_ndx > stabilization_steps
                        and step_ndx % checkpoint_pool_interval == 0):
                    current_score = ema_loss
                    if len(self.best_training_checkpoints) < checkpoint_averaging_top_k:
                        self.best_training_checkpoints.append(copy.deepcopy(self.model.state_dict()))
                        self.best_training_scores.append({"step": step_ndx, "stable_loss": current_score})
                    else:
                        worst_score = max(s['stable_loss'] for s in self.best_training_scores)
                        if current_score < worst_score:
                            worst_idx = [i for i, s in enumerate(self.best_training_scores) if s['stable_loss'] == worst_score][0]
                            self.best_training_checkpoints[worst_idx] = copy.deepcopy(self.model.state_dict())
                            self.best_training_scores[worst_idx] = {"step": step_ndx, "stable_loss": current_score}

            # info
            with torch.no_grad():
                if step_ndx % 100 == 0:
                    yp = torch.sigmoid(logits)
                    yt = labels
                    
                    is_pos = (yt == 1)
                    is_neg = (yt == 0)

                    # Training recall (TP / (TP + FN)) - cheap, reuses current batch
                    tp_train = int((yp[is_pos] >= 0.5).sum().item()) if is_pos.sum() > 0 else 0
                    fn_train = int((yp[is_pos] < 0.5).sum().item()) if is_pos.sum() > 0 else 0
                    train_recall = tp_train / (tp_train + fn_train) if (tp_train + fn_train) > 0 else 0.0
                    self.model.history['train_recall_steps'].append(step_ndx)
                    self.model.history['train_recall'].append(train_recall)

                    if logger:
                        current_lr = self.optimizer.param_groups[0]['lr']
                        pos_avg = yp[is_pos].mean().item() if is_pos.sum() > 0 else 0.0
                        neg_avg = yp[is_neg].mean().item() if is_neg.sum() > 0 else 0.0
                   
                        FA = int((yp[is_neg] > 0.5).sum().item()) if is_neg.sum() > 0 else 0
                        Ms = fn_train

                        N_total = is_neg.sum().item()
                        P_total = is_pos.sum().item()      

                        pos_term = per_example_loss[is_pos] / (1.0 - LOSS_BIAS) if (1.0 - LOSS_BIAS) > 0 else per_example_loss[is_pos]
                        neg_term = per_example_loss[is_neg] / LOSS_BIAS if LOSS_BIAS > 0 else per_example_loss[is_neg]
                        
                        PosL = pos_term.mean().item() if is_pos.sum() > 0 else 0.0
                        NegL = neg_term.mean().item() if is_neg.sum() > 0 else 0.0

                        logger.info(
                            f"[{step_ndx:5d}] L:{total_loss.item():.6f} "
                            f"PL:{PosL:.6f} NL:{NegL:.6f} |PA:{pos_avg:.3f} NA:{neg_avg:.3f} "
                            f"|FA:{FA}/{N_total} Ms:{Ms}/{P_total} |Recall:{train_recall:.3f} |η:{current_lr:.2e} gNorm:{grad_norm:.8f}"
                        )


            if patience > 0 and ema_loss is not None:
                if ema_loss < best_ema_loss_for_stopping - min_delta:
                    best_ema_loss_for_stopping = ema_loss
                    steps_without_improvement = 0
                else:
                    steps_without_improvement += 1
                
                # Only use training-loss early stopping when NO validation data is
                # available. When val data exists, validation-based stopping (above)
                # is the primary signal and this becomes a safety fallback.
                use_train_stopping = (X_val is None) or (not X_val)
                if use_train_stopping and step_ndx > stabilization_steps and steps_without_improvement >= patience:
                    print_info(f"\nEarly stopping triggered at step {step_ndx}. No improvement in stable loss for {patience} steps.")
                    break

            if checkpointing_enabled and step_ndx > 0 and step_ndx % checkpoint_interval == 0:
                checkpoint_data = {
                    'step': step_ndx,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'ema_loss': ema_loss,
                    'best_ema_loss_for_stopping': best_ema_loss_for_stopping,
                    'steps_without_improvement': steps_without_improvement,
                    
                    'model_history': self.model.history,
                    'best_error_score': self.best_error_score,
                    'best_model_on_error_score': self.best_model_on_error_score,
                    'best_training_checkpoints': self.best_training_checkpoints,
                    'best_training_scores': self.best_training_scores,
                    
                    # RNG States for perfect resume:
                    'torch_rng_state': torch.get_rng_state(),
                    'torch_cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    'np_rng_state': np.random.get_state(),
                    'random_rng_state': random.getstate()
                }
                checkpoint_name = f"checkpoint_step_{step_ndx}.pth"
                torch.save(checkpoint_data, os.path.join(checkpoint_dir, checkpoint_name))
                
                all_checkpoints = sorted(
                    [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_step_")],
                    key=lambda f: int(re.search(r"(\d+)", f).group(1))
                )
                if len(all_checkpoints) > checkpoint_limit:
                    os.remove(os.path.join(checkpoint_dir, all_checkpoints[0]))
                    
            validation_interval = self.config.get("val_interval", 500)
            validation_stabilization_steps = self.config.get("val_stabilization_steps", stabilization_steps)

            if X_val and step_ndx > validation_stabilization_steps and step_ndx % validation_interval == 0:
  
                val_metrics = self.validate(X_val)
                current_val_loss = val_metrics['val_loss']
                current_error_score = val_metrics['error_score']

                self.model.history['val_loss_steps'].append(step_ndx)
                self.model.history['val_loss'].append(current_val_loss)
                self.model.history['val_recall_steps'].append(step_ndx)
                self.model.history['val_recall'].append(val_metrics['val_recall'])
                self.model.history['val_fpr'].append(val_metrics['val_fpr'])
            
                if current_error_score < self.best_error_score:
                    self.best_error_score = current_error_score
                    self.best_model_on_error_score = copy.deepcopy(self.model.state_dict())
                    val_steps_without_improvement = 0

                    if debug_mode:
                        logger.info(
                            f"[VAL {step_ndx:5d}] New best! "
                            f"err={current_error_score:.1f} "
                            f"FA={val_metrics['total_false_alarms']} "
                            f"Miss={val_metrics['total_misses']} "
                            f"thresh={val_metrics['best_threshold']:.2f}"
                        )
                else:
                    val_steps_without_improvement += validation_interval

                # Validation-based early stopping: stop if val error hasn't improved
                # for val_patience_steps steps. This is the primary stopping signal
                # when validation data is available -- it directly measures real-world
                # generalisation rather than training loss.
                if (X_val and val_patience_steps > 0
                        and step_ndx > stabilization_steps
                        and val_steps_without_improvement >= val_patience_steps):
                    print_info(
                        f"\nValidation early stopping at step {step_ndx}. "
                        f"No improvement in val error score for {val_patience_steps} steps."
                    )
                    break

            if step_ndx >= max_steps - 1:
                break