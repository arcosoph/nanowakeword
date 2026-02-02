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
        Runs a full validation cycle on the provided dataloader and returns a
        dictionary of key performance metrics, including an error-based score.
        """
        import collections 
        import torch.nn.functional as F

        self.model.eval()
        self.model.classifier.eval()

        all_logits = []
        all_labels = []

        with torch.no_grad():
            for features, labels, _ in val_loader:
                features = features.to(self.model.device)
                labels = labels.to(self.model.device)
                
                logits = self.model(features).squeeze()
                
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())

        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels).float()
        
        val_loss = F.binary_cross_entropy_with_logits(all_logits, all_labels).item()

        preds_binary = (all_logits >= 0.0).float() 

        tp = ((preds_binary == 1) & (all_labels == 1)).sum().item() # True Positives
        tn = ((preds_binary == 0) & (all_labels == 0)).sum().item() # True Negatives
        fp = ((preds_binary == 1) & (all_labels == 0)).sum().item() # False Positives (False Alarms)
        fn = ((preds_binary == 0) & (all_labels == 1)).sum().item() # False Negatives (Misses)

        val_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        val_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        error_score = fp + fn

        metrics = collections.OrderedDict()
        metrics['val_loss'] = val_loss
        metrics['val_recall'] = val_recall
        metrics['val_fpr'] = val_fpr
        metrics['total_false_alarms'] = fp
        metrics['total_misses'] = fn
        metrics['error_score'] = error_score 

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

            # print_info("Training finished. Merging best checkpoints to create final model...")
            
            # if not self.best_training_checkpoints:
            #     print_info("No stable models were saved based on training loss stability. Returning the final model state.")

            #     final_model = self
            #     final_model.eval()
            # else:
            #     print_info(f"Averaging the top {len(self.best_training_checkpoints)} most stable models found during training...")
                
            #     averaged_state_dict = self.model.average_models(state_dicts=self.best_training_checkpoints)

            #     final_model = copy.deepcopy(self.model)
            #     final_model.load_state_dict(averaged_state_dict)
            #     final_model.eval() 



            print_info("Training finished. Selecting the best stable model based on smoothed validation score...")

            print_info(f"Loading the champion model with the best error score of {self.best_error_score}")
            final_model = copy.deepcopy(self.model)
            final_model.load_state_dict(self.best_model_on_error_score)

            final_model.eval()

            print_info("Calculating performance metrics for the final model...")
                        
            final_results = collections.OrderedDict()

            if self.best_training_scores:
                avg_stable_loss = np.mean([score['stable_loss'] for score in self.best_training_scores])
                final_results["Average Stable Loss"] = f"{avg_stable_loss:.4f}"
            else:
                final_results["Average Stable Loss"] = "N/A"
            
            # try:
            #     final_classifier_weights = final_model.classifier.weight.detach().cpu().numpy()
            #     weight_std_dev = np.std(final_classifier_weights)
            #     final_results["Weight Diversity (Std Dev)"] = f"{weight_std_dev:.4f}"
            # except Exception as e:
            #     final_results["Weight Diversity (Std Dev)"] = f"N/A (Error: {e})"


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

        default_warmup_steps = int(max_steps * 0.15)
        WARMUP_STEPS = self.config.get("WARMUP_STEPS", default_warmup_steps)
        min_delta = self.config.get("min_delta", 0.0001)
        best_ema_loss_for_stopping = float('inf')
        steps_without_improvement = 0
        ema_alpha = self.config.get("ema_alpha", 0.01)

        # steps_without_improvement = 0
        validation_interval = self.config.get("validation_interval", 500) 

        default_patience_steps = int(max_steps * 0.15)
        user_patience = self.config.get("early_stopping_patience", None)
        if user_patience is not None:
            patience = user_patience
        elif self.config.get("steps", max_steps) < 3000:
            patience = 0
        else:
            patience = default_patience_steps

        if patience == 0:
            print_info("Early stopping is DISABLED. Training will run for the full duration of 'steps'.")
        else:
            print_info(f"Training for {max_steps} steps. Model checkpointing and early stopping will activate after {WARMUP_STEPS} warm-up steps.")

        self.model.to(self.model.device)
        self.model.train() 
        self.model.classifier.train() 

        start_step = 0
        data_iterator = iter(itertools.cycle(X))

        if resume_from_dir:
            resume_checkpoint_dir = os.path.join(resume_from_dir, "2_training_artifacts", "checkpoints")
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
                        checkpoint = torch.load(checkpoint_path, map_location=self.device)
                        
                        self.load_state_dict(checkpoint['model_state_dict'])
                        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                        
                        start_step = checkpoint.get('step', 0)
                        ema_loss = checkpoint.get('ema_loss', None)
                        steps_without_improvement = checkpoint.get('steps_without_improvement', 0)
                        best_ema_loss_for_stopping = checkpoint.get('best_ema_loss_for_stopping', float('inf'))
                        self.history['loss'] = checkpoint.get('loss_history', [])
                        print_info(f"Successfully restored state. Resuming training from step {start_step + 1}.")
                        
                        print_info("Synchronizing data stream to the restored step...")
                        for _ in tqdm(range(start_step + 1), desc="Fast-forwarding data", unit="steps", leave=False):
                            next(data_iterator, None)
                    else:
                        print_info("WARNING: Checkpoint files found, but their names are not in the expected format. Starting fresh.")
                else:
                    print_info("WARNING: No valid checkpoint files found in the directory. Starting fresh.")
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

            LOSS_BIAS = self.config.get("LOSS_BIAS", 0.9)
            total_loss, per_example_loss = BiasWeightedLoss(logits, labels, LOSS_BIAS)
            
            total_loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                        list(self.model.parameters()) + list(self.model.classifier.parameters()),
                        max_norm=1.0
                    )

            self.optimizer.step()
            self.scheduler.step()

            # Live Feedback Loop, Update Hardness Scores
            # Using Exponential Moving Average (EMA) for stable updates
            alpha = 0.1  # Smoothing factor

            # Get the old hardness scores for the samples in this batch
            # We access the dataset via the DataLoader object 'X'
            old_hardness = X.dataset.sample_hardness[indices].to(self.model.device)

            # Calculate the new hardness score based on the current loss
            new_hardness = (alpha * per_example_loss) + ((1.0 - alpha) * old_hardness)

            # Update the scores in the dataset's memory (move back to CPU)
            X.dataset.sample_hardness[indices] = new_hardness.cpu()

            # Use the 'total_loss' for history and EMA calculation
            current_loss = total_loss.detach().cpu().item()

            if current_loss is not None:
                self.model.history["loss"].append(current_loss)
                
                if ema_loss is None: 
                    ema_loss = current_loss
                ema_loss = ema_alpha * current_loss + (1 - ema_alpha) * ema_loss

                if step_ndx > WARMUP_STEPS:
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
                if logger and step_ndx % 100 == 0:
                    yp = torch.sigmoid(logits)
                    yt = labels
                    
                    is_pos = (yt == 1)
                    is_neg = (yt == 0)
                    current_lr = self.optimizer.param_groups[0]['lr']
                    pos_avg = yp[is_pos].mean().item() if is_pos.sum() > 0 else 0.0
                    neg_avg = yp[is_neg].mean().item() if is_neg.sum() > 0 else 0.0
                    FA = (yp[is_neg] > 0.5).float().mean().item() if is_neg.sum() > 0 else 0.0
                    Ms = (yp[is_pos] < 0.5).float().mean().item() if is_pos.sum() > 0 else 0.0
                    
                    pos_term = per_example_loss[is_pos] / (1.0 - LOSS_BIAS) if (1.0 - LOSS_BIAS) > 0 else per_example_loss[is_pos]
                    neg_term = per_example_loss[is_neg] / LOSS_BIAS if LOSS_BIAS > 0 else per_example_loss[is_neg]

                    PosL = pos_term.mean().item() if is_pos.sum() > 0 else 0.0
                    NegL = neg_term.mean().item() if is_neg.sum() > 0 else 0.0

                    logger.info(
                        f"[{step_ndx:5d}] L:{total_loss.item():.6f} "
                        f"PL:{PosL:.6f} NL:{NegL:.6f} |PA:{pos_avg:.3f} NA:{neg_avg:.3f} "
                        f"|FA:{FA:.3f} Ms:{Ms:.3f} |η:{current_lr:.2e} gNorm:{grad_norm:.3f}"
                    )


            if patience > 0 and ema_loss is not None:
                if ema_loss < best_ema_loss_for_stopping - min_delta:
                    best_ema_loss_for_stopping = ema_loss
                    steps_without_improvement = 0
                else:
                    steps_without_improvement += 1
                
                if step_ndx > WARMUP_STEPS and steps_without_improvement >= patience:
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
                    'loss_history': self.model.history['loss']
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
            validation_warmup_steps = self.config.get("val_warmup_steps", WARMUP_STEPS)

            if X_val and step_ndx > validation_warmup_steps and step_ndx % validation_interval == 0:
  
                val_metrics = self.validate(X_val)
                current_val_loss = val_metrics['val_loss']
                current_error_score = val_metrics['error_score']

                self.model.history['val_loss_steps'].append(step_ndx)
                self.model.history['val_loss'].append(current_val_loss)
            
                if current_error_score < self.best_error_score:
                    self.best_error_score = current_error_score
                    self.best_model_on_error_score = copy.deepcopy(self.model.state_dict())
                    
                    # print_info(f"New best model found! Error Score: {current_error_score} (FA: {val_metrics['total_false_alarms']}, Misses: {val_metrics['total_misses']}) at step {step_ndx}. Checkpoint saved.")

            if step_ndx >= max_steps - 1:
                break
