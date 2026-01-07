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
import sys
import time
import yaml
import json
import copy
import scipy
import torch
import random
import hashlib
import logging
import warnings
import argparse
import torchinfo
import matplotlib
import numpy as np
import collections
from torch import nn
from tqdm import tqdm
import collections.abc
from pathlib import Path
import matplotlib.pyplot as plt
from logging.handlers import RotatingFileHandler
from nanowakeword.config_generator import ConfigGenerator

from .data import HardSampleFilterSampler
from torch.utils.data import DataLoader
from nanowakeword.data import augment_clips, WakeWordDataset, generate_adversarial_texts

from nanowakeword.modules import BiasWeightedLoss
from nanowakeword.modules.audio_processing import AudioFeatures
from nanowakeword.modules.preprocess import verify_and_process_directory

from nanowakeword.utils.GNMV import GNMV
from nanowakeword.utils.ConfigProxy import ConfigProxy
from nanowakeword.utils.analyzer import DatasetAnalyzer
from nanowakeword.utils.DynamicTable import DynamicTable
from nanowakeword.utils.journal import update_training_journal
from nanowakeword.utils.logger import print_banner, print_step_header, print_info, print_key_value, print_final_report_header, print_table

from .modules.architectures import (
    CNNModel, LSTMModel, Net, GRUModel, RNNModel, TransformerModel, 
    CRNNModel, TCNModel, QuartzNetModel, ConformerModel, EBranchformerModel
)

matplotlib.use('Agg')

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

def deep_merge(d1, d2):
    """
    Recursively merges d2 into d1. If a key exists in both and the values are
    dictionaries, it merges them recursively. Otherwise, the value from d2
    overwrites the value from d1.
    """
    for k, v in d2.items():
        if k in d1 and isinstance(d1[k], dict) and isinstance(v, collections.abc.Mapping):
            d1[k] = deep_merge(d1[k], v)
        else:
            d1[k] = v
    return d1

class Model(nn.Module):
    def __init__(self, config, model_name: str, n_classes=1, input_shape=(16, 96), model_type="dnn",
                layer_dim=128, n_blocks=1, seconds_per_example=None, dropout_prob=0.5):
        super().__init__()

        # Store inputs as attributes
        self.n_classes = n_classes
        self.input_shape = input_shape
        self.seconds_per_example = seconds_per_example
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.history = collections.defaultdict(list)
        self.model_name = model_name

        act_fn_type = config.get("activation_function", "relu").lower()
        if act_fn_type == "gelu":
            self.activation_fn = nn.GELU()
        elif act_fn_type == "silu":
            self.activation_fn = nn.SiLU()
        else: # Default to ReLU
            self.activation_fn = nn.ReLU()        

        embedding_dim = config.get("embedding_dim", 64)

        self.debug_save_dir = None

        if model_type.lower() in {"conformer", "e_branchformer", "crnn"}:
            print_info(f"\n[WARNING] The '{model_type.upper()}' architecture is highly sensitive to hyperparameters and may exhibit convergence instability.\n")

        if model_type == "cnn":
            self.model = CNNModel(input_shape, embedding_dim, dropout_prob=dropout_prob, activation_fn=self.activation_fn)
        elif model_type == "lstm":
            self.model = LSTMModel(input_shape[1], layer_dim, n_blocks, embedding_dim, bidirectional=True, dropout_prob=dropout_prob)
        elif model_type == "dnn":
            self.model = Net(input_shape, layer_dim, n_blocks, embedding_dim, dropout_prob=dropout_prob, activation_fn=self.activation_fn)
        elif model_type == "gru":
            self.model = GRUModel(input_shape[1], layer_dim, n_blocks, embedding_dim, bidirectional=True, dropout_prob=dropout_prob)
        elif model_type == "rnn":
            self.model = RNNModel(input_shape, embedding_dim, n_blocks, dropout_prob=dropout_prob)
        elif model_type == "transformer":
            d_model = config.get("transformer_d_model", 128)
            n_head = config.get("transformer_n_head", 4)
            self.model = TransformerModel(
                input_dim=input_shape[1], d_model=d_model, n_head=n_head, 
                n_layers=n_blocks, embedding_dim=embedding_dim, dropout_prob=dropout_prob
            )
        elif model_type == "crnn":
            cnn_channels = config.get("crnn_cnn_channels", [16, 32, 32])
            rnn_type = config.get("crnn_rnn_type", "lstm")
            self.model = CRNNModel(
                input_shape=input_shape, rnn_type=rnn_type, rnn_hidden_size=layer_dim, 
                n_rnn_layers=n_blocks, cnn_channels=cnn_channels, embedding_dim=embedding_dim,
                dropout_prob=dropout_prob, activation_fn=self.activation_fn
            )
        elif model_type == "tcn":
            tcn_channels = config.get("tcn_channels", [64, 64, 128])
            tcn_kernel_size = config.get("tcn_kernel_size", 3)
            self.model = TCNModel(
                input_dim=input_shape[1], num_channels=tcn_channels, embedding_dim=embedding_dim,
                kernel_size=tcn_kernel_size, dropout_prob=dropout_prob
            )
        elif model_type == "quartznet":
            default_quartznet_config = [[256, 33, 1], [256, 33, 1], [512, 39, 1]]
            quartznet_config = config.get("quartznet_config", default_quartznet_config)
            self.model = QuartzNetModel(
                input_dim=input_shape[1], quartznet_config=quartznet_config,
                embedding_dim=embedding_dim, dropout_prob=dropout_prob
            )
        elif model_type == "conformer":
            conformer_d_model = config.get("conformer_d_model", 144)
            conformer_n_head = config.get("conformer_n_head", 4)
            self.model = ConformerModel(
                input_dim=input_shape[1], d_model=conformer_d_model, n_head=conformer_n_head,
                n_layers=n_blocks, embedding_dim=embedding_dim, dropout_prob=dropout_prob
            )
        elif model_type == "e_branchformer":
            branchformer_d_model = config.get("branchformer_d_model", 144)
            branchformer_n_head = config.get("branchformer_n_head", 4)
            self.model = EBranchformerModel(
                input_dim=input_shape[1], d_model=branchformer_d_model, n_head=branchformer_n_head,
                n_layers=n_blocks, embedding_dim=embedding_dim, dropout_prob=dropout_prob
            )
        else:
            raise ValueError(f"Unsupported model_type: '{model_type}'.")

        self.classifier = nn.Linear(embedding_dim, n_classes)

        # Define logging dict (in-memory)
        self.history = collections.defaultdict(list)


    def setup_optimizer_and_scheduler(self, config):
            """
            Sets up the optimizer and the learning rate scheduler based on the
            provided configuration. Supports multiple scheduler types.
            """
            from itertools import chain

            all_params = chain(self.model.parameters(), self.classifier.parameters())
            
            optimizer_type = config.get("optimizer_type", "adamw").lower() 
            learning_rate = config.get('learning_rate_max', 1e-4)
            weight_decay = config.get("weight_decay", 1e-2)
            momentum = config.get("momentum", 0.9)

            if optimizer_type == "adam":
                self.optimizer = torch.optim.Adam(all_params, lr=learning_rate, weight_decay=weight_decay)
            elif optimizer_type == "sgd":
                self.optimizer = torch.optim.SGD(all_params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
            else: # Default to AdamW
                self.optimizer = torch.optim.AdamW(all_params, lr=learning_rate, weight_decay=weight_decay)
            
            print_info(f"Using optimizer: {optimizer_type.upper()}")


            #  Scheduler Setup (New Dynamic Logic) 
            # Get the scheduler type from config, defaulting to 'onecycle' for backward compatibility.
            scheduler_type = config.get('lr_scheduler_type', 'onecycle').lower()
            
            print_info(f"Setting up learning rate scheduler: {scheduler_type.upper()}")

            if scheduler_type == 'cyclic':
                # This is your original, powerful CyclicLR setup
                self.scheduler = torch.optim.lr_scheduler.CyclicLR(
                    self.optimizer,
                    base_lr=config['learning_rate_base'],
                    max_lr=config['learning_rate_max'],
                    step_size_up=config['clr_step_size_up'],
                    step_size_down=config.get("clr_step_size_down", config['clr_step_size_up']),
                    mode='triangular2', # or 'triangular' 
                    cycle_momentum=False
                )
            
            elif scheduler_type == 'onecycle':
                # OneCycleLR is another very powerful scheduler, great for fast convergence.
                # It requires the maximum learning rate and total training steps.
                self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    self.optimizer,
                    max_lr=config['learning_rate_max'],
                    total_steps=config['steps']
                )

            elif scheduler_type == 'cosine':
                # CosineAnnealingLR smoothly decreases the learning rate in a cosine curve.
                # It only requires the total number of training steps (T_max).
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=config['steps'],
                    eta_min=config.get('learning_rate_base', 1e-6) # Use base_lr as the minimum
                )

            else:
                raise ValueError(
                    f"Unsupported lr_scheduler_type: '{scheduler_type}'. "
                    "Supported types are: 'cyclic', 'onecycle', 'cosine'."
                )

    def plot_history(self, output_dir):
            """
            Creates a meaningful graph of training loss and its stable form (EMA).
            """
            print_info("Generating training performance graph...")
            graph_output_dir = os.path.join(output_dir, "graphs")
            os.makedirs(graph_output_dir, exist_ok=True)

            loss_history = np.array(self.history['loss'])
            
            ema_loss_history = []
            ema_loss = None
            # alpha = 0.01  # Match this value with the train_model function.
                        # alpha = 0.01  # Match this value with the train_model function.
            alpha = self.config.get("ema_alpha", 0.01)

            for loss_val in loss_history:
                if ema_loss is None:
                    ema_loss = loss_val
                else:
                    ema_loss = alpha * loss_val + (1 - alpha) * ema_loss

                ema_loss_history.append(ema_loss)

            plt.figure(figsize=(12, 6))
            
            plt.plot(loss_history, label='Training Loss (Raw)', color='skyblue', alpha=0.6)
            
            plt.plot(ema_loss_history, label='Training Loss (Stable/EMA)', color='navy', linewidth=2)
            
            plt.title('Training Loss Stability Analysis', fontsize=16)
            plt.xlabel('Training Steps', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.ylim(bottom=0) # Loss will never go below 0

            save_path = os.path.join(graph_output_dir, "training_performance_graph.png")
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()

            print_info(f"Performance graph saved to: {save_path}")

    def forward(self, x):
            """
            Takes input features and returns the final classification logits
            in a standardized tensor shape [batch, sequence, classes].
            """
            embeddings = self.model(x)
            logits = self.classifier(embeddings) # Shape: [batch_size, 1]

            # Ensure standardized output shape [B, 1, 1] 
            # Add a new dimension for the 'sequence' length, which is 1 in our case.
            return logits.unsqueeze(1)

    def summary(self):
        return torchinfo.summary(self.model, input_size=(1,) + self.input_shape, device='cpu')


    def average_models(self, state_dicts: list):
                """The given model averages the weights of the state_dicts."""
                if not state_dicts:
                    raise ValueError("Cannot average an empty list of state dicts.")

                avg_state_dict = copy.deepcopy(state_dicts[0])
                
                # Zero out all floating-point parameters to prepare for summation.
                # We will skip non-floating point parameters like 'num_batches_tracked'.
                for key in avg_state_dict:
                    if avg_state_dict[key].is_floating_point():
                        avg_state_dict[key].fill_(0)

                # Sum up the parameters from all checkpoints.
                for state_dict in state_dicts:
                    for key in avg_state_dict:
                        if avg_state_dict[key].is_floating_point():
                            avg_state_dict[key] += state_dict[key]

                # Divide the summed parameters by the number of checkpoints to get the average.
                for key in avg_state_dict:
                    if avg_state_dict[key].is_floating_point():
                        avg_state_dict[key] /= len(state_dicts)
                # Non-floating point parameters (like counters) will retain the value from the first checkpoint.

                return avg_state_dict


    def auto_train(self, X_train, dataset, steps, table_updater, debug_path, resume_from_dir=None):
            """
            A modern, single-sequence training process that combines the best checkpoints to
              create a final and robust model.
            """

            self.train_model(
                X=X_train,
                dataset=dataset,
                max_steps=steps,
                log_path=debug_path,
                table_updater=table_updater,
                resume_from_dir=resume_from_dir        
            )

            print_info("Training finished. Merging best checkpoints to create final model...")
            
            if not self.best_training_checkpoints:
                print_info("No stable models were saved based on training loss stability. Returning the final model state.")
    
                final_model = self
                final_model.eval()
            else:
                print_info(f"Averaging the top {len(self.best_training_checkpoints)} most stable models found during training...")
                
                averaged_state_dict = self.average_models(state_dicts=self.best_training_checkpoints)

                final_model = copy.deepcopy(self)
                final_model.load_state_dict(averaged_state_dict)
                final_model.eval() 

            print_info("Calculating performance metrics for the final averaged model...")
            
            final_results = collections.OrderedDict()

            if self.best_training_scores:
                avg_stable_loss = np.mean([score['stable_loss'] for score in self.best_training_scores])
                final_results["Average Stable Loss"] = f"{avg_stable_loss:.4f}"
            else:
                final_results["Average Stable Loss"] = "N/A"
            
            try:
                final_classifier_weights = final_model.classifier.weight.detach().cpu().numpy()
                weight_std_dev = np.std(final_classifier_weights)
                final_results["Weight Diversity (Std Dev)"] = f"{weight_std_dev:.4f}"
            except Exception as e:
                final_results["Weight Diversity (Std Dev)"] = f"N/A (Error: {e})"

            try:
                with torch.no_grad():
                    _, x_batch, y_batch = next(iter(X_train))
                    confidence_batch_x = x_batch.to(self.device)
                    confidence_batch_y = y_batch.to(self.device)

                    predictions = final_model(confidence_batch_x).squeeze()
                    
                    pos_preds = predictions[confidence_batch_y.squeeze() == 1]
                    neg_preds = predictions[confidence_batch_y.squeeze() == 0]

                    final_results["Avg. Positive Score (Logit)"] = f"{pos_preds.mean().item():.3f}"
                    final_results["Avg. Negative Score (Logit)"] = f"{neg_preds.mean().item():.3f}"

            except (StopIteration, RuntimeError) as e:
                final_results["Confidence Score"] = f"N/A (Error: {e})"

            print_final_report_header()
            print_info("NOTE: These metrics are indicators of model health, not real-world performance.")

            for key, value in final_results.items():
                print_key_value(key, value)
    
            self.history["final_report"] = final_results

            # Returning the completed and averaged model
            return final_model
    
    def export_model(self, model, model_name, output_dir):
        """
        Exports the final trained model to a standard, inference-ready ONNX format.

        This function ensures hardware independence by moving both the model and a
        dummy input to the CPU before export. It also guarantees a standardized
        output shape of [batch_size, 1, 1] for maximum compatibility.
        """
        # A robust wrapper to apply sigmoid and ensure the final output shape
        class InferenceWrapper(nn.Module):
            def __init__(self, trained_model):
                super().__init__()
                self.trained_model = trained_model

            def forward(self, x):
                logits = self.trained_model(x)
                probabilities = torch.sigmoid(logits)
                # Forcefully reshape the output to a standard 3D tensor
                return probabilities.view(-1, 1, 1)

        exportable_model = InferenceWrapper(model)
        exportable_model.eval()

        # Define a dummy input for tracing the model graph
        # dummy_input = torch.rand(1, *self.input_shape)
        # New line (Ensures float32 and correct device)
        dummy_input = torch.randn(1, *self.input_shape, device='cpu', dtype=torch.float32)
        
        onnx_path = os.path.join(output_dir, model_name + '.onnx')
        
        print_info(f"Saving inference-ready ONNX model to '{onnx_path}'")
        
        opset_version = self.config.get("onnx_opset_version", 17)
        print_info(f"Using ONNX opset version: {opset_version}")

        try:
            # For maximum compatibility and to prevent device errors, always move both
            # the model and the dummy input to the CPU before exporting to ONNX.
            
            model_cpu = exportable_model.cpu()
            dummy_input_cpu = dummy_input.cpu()
            
            torch.onnx.export(
                model_cpu,
                dummy_input_cpu,
                onnx_path,
                opset_version=opset_version,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )
            print_info("ONNX model saved successfully.")

        except Exception as e:
            # Provide a more detailed error message
            print_info("ERROR: ONNX export failed. Fix the issue and run again with --resume if a checkpoint exists.")
            print_info(f"   Details: {e}")


    def export_pytorch_model(self, model, model_name, output_dir):
        """
        Saves the final trained PyTorch model's state_dict to a .pt file.
        This is the recommended way to save PyTorch models for robustness and portability.

        Args:
            model (torch.nn.Module): The final, trained model object (e.g., best_model).
            model_name (str): The base name for the output model file.
            output_dir (str): The directory where the model file will be saved.
        """
        model.eval()
        pytorch_path = os.path.join(output_dir, model_name + '.pt')
        
        print_info(f"Saving final PyTorch model (state_dict) to '{pytorch_path}'")
        
        try:
            torch.save(model.state_dict(), pytorch_path)
            print_info("PyTorch model saved successfully.")
            
        except Exception as e:
            print_info(f"ERROR: PyTorch model save failed. Details: {e}")


    def train_model(self, X, dataset, max_steps, log_path, table_updater, resume_from_dir=None):
        
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
        default_warmup_steps = int(max_steps * 0.15)
        WARMUP_STEPS = self.config.get("WARMUP_STEPS", default_warmup_steps)
        min_delta = self.config.get("min_delta", 0.0001)
        best_ema_loss_for_stopping = float('inf')
        steps_without_improvement = 0
        ema_alpha = self.config.get("ema_alpha", 0.01)

        FILTER_INTERVAL_STEPS = self.config.get("filter_interval_steps", 500)
        LOSS_THRESHOLD = self.config.get("loss_threshold_for_easy", 0.01)

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

        self.to(self.device)
        self.model.train() 
        self.classifier.train() 

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

            current_loss, original_indices, per_sample_losses = BiasWeightedLoss(self, data, step_ndx, logger)

            if current_loss is not None:
                self.history["loss"].append(current_loss)
                
                if ema_loss is None: 
                    ema_loss = current_loss
                ema_loss = ema_alpha * current_loss + (1 - ema_alpha) * ema_loss

                if step_ndx > WARMUP_STEPS:
                    current_score = ema_loss
                    if len(self.best_training_checkpoints) < checkpoint_averaging_top_k:
                        self.best_training_checkpoints.append(copy.deepcopy(self.state_dict()))
                        self.best_training_scores.append({"step": step_ndx, "stable_loss": current_score})
                    else:
                        worst_score = max(s['stable_loss'] for s in self.best_training_scores)
                        if current_score < worst_score:
                            worst_idx = [i for i, s in enumerate(self.best_training_scores) if s['stable_loss'] == worst_score][0]
                            self.best_training_checkpoints[worst_idx] = copy.deepcopy(self.state_dict())
                            self.best_training_scores[worst_idx] = {"step": step_ndx, "stable_loss": current_score}

            if (step_ndx + 1) % FILTER_INTERVAL_STEPS == 0 and step_ndx > 0:   

                current_device = per_sample_losses.device                
                indices_on_correct_device = original_indices.to(current_device)                
                easy_mask = per_sample_losses < LOSS_THRESHOLD                
                easy_indices_in_batch = indices_on_correct_device[easy_mask].cpu().tolist()
                
                if easy_indices_in_batch:
                    dataset.mark_as_easy(easy_indices_in_batch)
                    
                    training_loop.set_description(
                        f"Training (Hard samples: {len(dataset)}/{dataset.total_samples})"
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
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'ema_loss': ema_loss,
                    'best_ema_loss_for_stopping': best_ema_loss_for_stopping,
                    'steps_without_improvement': steps_without_improvement,
                    'loss_history': self.history['loss']
                }
                checkpoint_name = f"checkpoint_step_{step_ndx}.pth"
                torch.save(checkpoint_data, os.path.join(checkpoint_dir, checkpoint_name))
                
                all_checkpoints = sorted(
                    [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_step_")],
                    key=lambda f: int(re.search(r"(\d+)", f).group(1))
                )
                if len(all_checkpoints) > checkpoint_limit:
                    os.remove(os.path.join(checkpoint_dir, all_checkpoints[0]))
                    
            if step_ndx >= max_steps - 1:
                break


def train(cli_args=None):
    
    parser = argparse.ArgumentParser(
        description="NanoWakeWord: The Intelligent Wake Word Training Framework.",
        formatter_class=argparse.RawTextHelpFormatter # For better help message formatting
    )

    # Configuration 
    parser.add_argument(
            "-c", "--config_path",
            help="Path to the training configuration YAML file. (Required)",
            type=str,
            required=True,
            metavar="PATH"
        )

    # Pipeline Stages (Primary Actions)
    parser.add_argument(
        "-G", "--generate_clips",
        help="Activates the 'Generation' stage to synthesize audio clips.",
        action="store_true"
    )
    parser.add_argument(
        "-t", "--transform_clips",
        help="Activates the preparatory 'transform' stage (augmentation and feature extraction).",
        action="store_true"
    )
    parser.add_argument(
        "-T", "--train_model",
        help="Activates the final 'Training' stage to build the model.",
        action="store_true"
    )

    # Modifiers 
    parser.add_argument(
        "-f", "--force-verify",
        help="Forces re-verification of all data directories, ignoring the cache.",
        action="store_true"
    )
    parser.add_argument(
        "--overwrite", # NO SHORTHAND BY DESIGN FOR SAFETY
        help="Forces regeneration of feature files, overwriting any existing ones. Use with caution.",
        action="store_true"
    )

    parser.add_argument(
        "--resume",
        help="Path to the project directory to resume training from. (e.g., --resume ./trained_models/my_wakeword_v1)",
        type=str,
        default=None,
        metavar="PATH"
    )
    
    args = parser.parse_args(cli_args)


#=====
    print_banner()

    user_config = yaml.load(open(args.config_path, 'r', encoding='utf-8').read(), yaml.Loader)
# #=====

    # Define a stable cache directory based on the user's output_dir
    #    This ensures the path is known and available from the very beginning.
    output_dir_from_config = user_config.get("output_dir", "./trained_models")
    VERIFICATION_CACHE_DIR = os.path.join(output_dir_from_config, ".cache", "verification_receipts")
    os.makedirs(VERIFICATION_CACHE_DIR, exist_ok=True)

    # Define these file names here to avoid magic strings
    VERIFICATION_RECEIPT_FILENAME_TEMPLATE = "{hash}.json" # We'll use a hash now

    def get_directory_state(path):
        """Returns the current state of a directory (number of files and total size)."""
        file_count = 0
        total_size = 0
        # More robustly check for various audio extensions
        audio_extensions = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
        
        try:
            for entry in os.scandir(path):
                if entry.is_file() and os.path.splitext(entry.name)[1].lower() in audio_extensions:
                    file_count += 1
                    total_size += entry.stat().st_size
        except FileNotFoundError:
            # If the directory doesn't exist, its state is empty
            return {"file_count": 0, "total_size": 0}
                
        return {"file_count": file_count, "total_size": total_size}

    def smart_verify(path, force=False):
        """
        Smartly verifies a directory using a centralized cache in the project's output directory.
        This version is robust and will work correctly.
        """
        if not path: # Handle cases where a path might be None or empty in the config
            return

        # 1. Create a stable and unique hash for the directory path
        path_hash = hashlib.md5(path.encode('utf-8')).hexdigest()
        receipt_filename = VERIFICATION_RECEIPT_FILENAME_TEMPLATE.format(hash=path_hash)
        receipt_path = os.path.join(VERIFICATION_CACHE_DIR, receipt_filename)

        # 2. Check the verification receipt
        if not force and os.path.exists(receipt_path):
            try:
                with open(receipt_path, 'r') as f:
                    saved_state = json.load(f)
                
                current_state = get_directory_state(path)
                
                # If the state is identical, we can safely skip
                if saved_state == current_state:
                    print_info(f"'{os.path.basename(path)}' already verified. Skipping.")
                    return 
                else:
                    print_info(f"Data in '{os.path.basename(path)}' has changed. Re-verifying...")

            except (json.JSONDecodeError, KeyError) as e:
                print_info(f"Could not read or parse receipt for '{os.path.basename(path)}'. Re-verifying... Error: {e}")
        
        # 3. Perform the actual verification and processing
        # This part runs if verification is forced, receipt doesn't exist, or state has changed.
        try:
            verify_and_process_directory(path)
        
            # 4. Write the new state to the centralized cache
            current_state = get_directory_state(path)
            with open(receipt_path, 'w') as f:
                json.dump(current_state, f, indent=4)

        except FileNotFoundError:
            logging.warning(f"Directory not found, skipping preprocessing: {path}")
        except Exception as e:
            # Catch other potential errors during verification or writing the receipt
            print_info(f"Warning: An unexpected error occurred for '{os.path.basename(path)}'. Error: {e}")


    print_step_header("Verifying and Preprocessing Data Directories")
    
    data_paths_to_process = [
        user_config.get("positive_data_path"),
        user_config.get("negative_data_path")
    ]

    data_paths_to_process.extend(user_config.get("background_paths", []))
    data_paths_to_process.extend(user_config.get("rir_paths", []))


    unique_paths = set(p for p in data_paths_to_process if p)
    
    ISforce_verify= user_config.get("force_verify", False)
    if args.force_verify or ISforce_verify:
        print_info("User has forced re-verification of all data directories.")
    
    for path in unique_paths:
        smart_verify(path, force=args.force_verify or ISforce_verify)
        
    print_info("Data verification and preprocessing complete.\n")
   
    # Hardware-Only Configuration (Express Pass) 
    print_info("Determining hardware-specific configurations...")
    try:
        generator1 = ConfigGenerator() 
        intelligent_config1 = generator1.generate()

        final_config1 = intelligent_config1.copy()
        final_config1.update(user_config)

        base_config = final_config1

    except Exception as e:
        print_info(f"Could not generate intelligent hardware config due to an error: {e}. Proceeding with user config only.")
        base_config = user_config.copy() 

    ISgenaret_data = base_config.get("generate_clips", False)
    if args.generate_clips or ISgenaret_data:
        from nanowakeword.generate_samples import generate_samples
        print_step_header("Activating Synthetic Data Generation Engine")

        # Acquire the Target Phrase
        target_phrase = base_config.get("target_phrase")
        if not target_phrase:
            print_info("\n[CONFIGURATION NOTICE]: 'target_phrase' is not set in your config file. This is required to generate audio samples.")
            try:
                user_input = input(">>> Please enter the target phrase to proceed: ").strip()
                if not user_input:
                    print_info("\n[ABORT] A target phrase is mandatory for generation. Exiting.")
                    sys.exit(1)
                target_phrase = [user_input]
                print_info(f"Using runtime target phrase: '{user_input}'")
            except (KeyboardInterrupt, EOFError):
                print_info("\n\nOperation cancelled by user.")
                sys.exit()

        # 1. Retrieve Sample Counts (Handle missing values safely)
        raw_pos_samples = base_config.get('generate_positive_samples')
        raw_neg_samples = base_config.get('generate_negative_samples')

        tts_settings = base_config.get("tts_settings", {})

        # Convert to integers if present, else default to 0
        n_pos_train = int(raw_pos_samples) if raw_pos_samples is not None else 0

        # 2. Configure Negative Data Generation Strategy
        enable_auto_adversarial = base_config.get("adversarial_text_generation", True)
        custom_negatives = base_config.get("custom_negative_phrases", [])
        repeats_per_phrase = int(base_config.get("custom_negative_per_phrase", 50))

        include_partial_phrase= base_config.get("include_partial_phrase", 0.5)
        include_input_words= base_config.get("include_input_words", 0.1)
        multi_word_prob = base_config.get("multi_word_prob", 0.8)
        max_multi_word_len= base_config.get("max_multi_word_len", 2)

        final_negative_texts = []

        # Custom Negative Phrases 
        if custom_negatives:
            print_info(f"Processing {len(custom_negatives)} custom negative phrases.")
            print_info(f"Generating {repeats_per_phrase} copies for EACH custom phrase.")

            # Expand the list: Each custom phrase is repeated 'repeats_per_phrase' times
            for phrase in custom_negatives:
                final_negative_texts.extend([phrase] * repeats_per_phrase)
            
            print_info(f"Total custom samples prepared: {len(final_negative_texts)}")

        # Gap Filling with Auto-Adversarial Data 
        if raw_neg_samples is not None:
            # Scenario: User provided a specific target total (e.g., 600)
            target_total_neg = int(raw_neg_samples)
            current_count = len(final_negative_texts)
            gap = max(0, target_total_neg - current_count)

            if gap > 0:
                if enable_auto_adversarial:
                    print_info(f"Target negative samples: {target_total_neg}. Current custom samples: {current_count}.")
                    print_info(f"Generating {gap} auto-adversarial phrases to fill the gap.")
                    
                    # Generate phonetically similar words to fill the remaining count
                    auto_adversarial = generate_adversarial_texts(
                                                                  target_phrase[0], 
                                                                  N=gap, 
                                                                  include_input_words=include_input_words, 
                                                                  include_partial_phrase=include_partial_phrase, 
                                                                  multi_word_prob=multi_word_prob,
                                                                  max_multi_word_len=max_multi_word_len)
                    
                    final_negative_texts.extend(auto_adversarial)
                else:
                    print_info(f"Target is {target_total_neg}, but auto-adversarial generation is DISABLED.")
                    print_info(f"Proceeding with only {current_count} custom samples.")
            else:
                if current_count > target_total_neg:
                    print_info(f"Note: Custom samples ({current_count}) exceed the target ({target_total_neg}). Keeping all custom samples.")

        # Update final count
        final_neg_count = len(final_negative_texts)

        # Construct Unified Generation Plan
        generation_plan = {}

        if n_pos_train > 0:
            generation_plan["Positive_Train"] = {
                "count": n_pos_train,
                "texts": target_phrase,
                "output_dir": base_config["positive_data_path"],
                "prefix": "pos" 
            }
        
        if final_neg_count > 0:
            generation_plan["Adversarial_Train"] = {
                "count": final_neg_count,
                "texts": final_negative_texts,
                "output_dir": base_config["negative_data_path"],
                "prefix": "neg" 
            }

        # Execute Generation Engine
        if generation_plan:
            print_info(f"Initiating data generation pipeline for phrase: '{target_phrase[0]}'")
            
            for task_name, params in generation_plan.items():
                if params["count"] > 0 and params["texts"]:
                    print_info(f"Executing task '{task_name}': {params['count']} clips -> '{params['output_dir']}'")
                    os.makedirs(params["output_dir"], exist_ok=True)
                    
                    generate_samples(
                        text=params["texts"],
                        max_samples=params["count"],
                        output_dir=params["output_dir"],
                        file_prefix=params.get("prefix", "sample"),
                        **tts_settings
                    )
                    
                    # Clear GPU cache after each heavy task to prevent fragmentation
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        print_info("Synthetic data generation process finished successfully.\n")


    print_step_header("Activating Intelligent Configuration Engine")
    try:
        analyzer = DatasetAnalyzer(
            positive_path=user_config["positive_data_path"],
            negative_path=user_config["negative_data_path"],
            noise_path=user_config.get("background_paths", []), 
            rir_path=user_config["rir_paths"][0]
        ) 
        dataset_stats = analyzer.analyze()
        print_table(dataset_stats, "Dataset Statistics")

        generator = ConfigGenerator(dataset_stats)
        intelligent_config = generator.generate()

    except KeyError as e:
        print_info(f"ERROR: Missing essential path in config file for auto-config: {e}")
        exit()
    except Exception as e:
        print_info(f"Could not generate intelligent config due to an error: {e}. Proceeding with user config only.")
        intelligent_config = {} 

    final_config = intelligent_config.copy()
    
    # Deep merge the user's configuration on top of it.
    # This will correctly merge nested dictionaries like 'augmentation_settings'.
    final_config = deep_merge(final_config, user_config)
    
    # Now, `final_config` contains the correct, merged values.
    config_proxy = ConfigProxy(final_config)
    config = config_proxy

    show_table_flag = config.get("show_training_summary", True)
    dynamic_table = DynamicTable(config_proxy, title="Effective Training Configuration", enabled=show_table_flag)

    # Define and Create Professional Output Directory Structure  
    # project_dir = os.path.join(os.path.abspath(base_config["output_dir"]), base_config.get("model_name", AGMINTVP))
    project_dir = os.path.join(
        os.path.abspath(base_config["output_dir"]),
        base_config.get(
            "model_name",
            GNMV(model_type=config.get("model_type", "dnn"))
        )
    )

    feature_save_dir = os.path.join(project_dir, "features")
    artifacts_dir = os.path.join(project_dir, "training_artifacts")
    model_save_dir = os.path.join(project_dir, "model")

    for path in [project_dir, feature_save_dir, artifacts_dir, model_save_dir]:
        os.makedirs(path, exist_ok=True)

    print_info(f"Project assets will be saved in: {project_dir}")

    # Get paths for impulse response and background audio files
    rir_paths = [i.path for j in config["rir_paths"] for i in os.scandir(j)]
    background_paths = []
    if len(config["background_paths_duplication_rate"]) != len(config["background_paths"]):
        config["background_paths_duplication_rate"] = [1]*len(config["background_paths"])
    for background_path, duplication_rate in zip(config["background_paths"], config["background_paths_duplication_rate"]):
        background_paths.extend([i.path for i in os.scandir(background_path)]*duplication_rate)

    # Determine the optimal training clip length 
    # Get the audio config section using the proxy. This returns another ConfigProxy.
    audio_cfg = config.get("audio_processing", {})

    # Priority 1: Check if the user has provided a fixed clip length to override everything.
    # The access to 'clip_length_samples' is now automatically tracked by the proxy.
    fixed_clip_length = audio_cfg.get("clip_length_samples", None)

    if fixed_clip_length is not None:
        # If a fixed length is specified, use it directly and skip the autotune process.
        config["total_length"] = fixed_clip_length
        print_info(f"Using user-defined clip duration: {fixed_clip_length} samples.")

    else:
        # Priority 2: Proceed with the autotune process.
        # Get the autotune section. If it doesn't exist, use an empty dict as default.
        # This also returns a ConfigProxy.
        autotune_cfg = audio_cfg.get("autotune_length", {})
        
        # Autotune is enabled by default. Each .get() call from here is automatically
        # tracked with the full nested path (e.g., "audio.autotune_length.enabled").
        if autotune_cfg.get("enabled", True):
            print_info("Autotuning optimal clip duration...")

            # Get autotune parameters. The proxy handles defaults gracefully.
            num_to_inspect = autotune_cfg.get("num_samples_to_inspect", 50)
            buffer_ms = autotune_cfg.get("duration_buffer_ms", 750)
            min_length = autotune_cfg.get("min_allowable_length", 32000)
            snap_tolerance = autotune_cfg.get("snap_to_min_tolerance", 4000)
            
            # Sample clips and calculate median duration 
            positive_clips_path = Path(config["positive_data_path"])
            positive_clips = [str(p) for p in positive_clips_path.glob("*.wav")]
            
            if not positive_clips:
                raise FileNotFoundError(f"No .wav files found for autotuning in: {positive_clips_path}")
            
            num_to_sample = min(num_to_inspect, len(positive_clips))
            sampled_clips = np.random.choice(positive_clips, num_to_sample, replace=False)

            duration_in_samples = []
            for clip_path in sampled_clips:
                try:
                    sample_rate, data = scipy.io.wavfile.read(clip_path)
                    if sample_rate != 16000:
                        print_info(f"[WARNING] Clip '{os.path.basename(clip_path)}' has sample rate {sample_rate}Hz, not 16kHz. This may affect duration calculation.")
                    duration_in_samples.append(len(data))
                except Exception as e:
                    print_info(f"[WARNING] Could not read and process clip '{os.path.basename(clip_path)}': {e}")
            
            # Calculate the final length based on the sampled durations
            if not duration_in_samples:
                print_info("[WARNING] Could not determine median duration. Using minimum allowable length as fallback.")
                final_length = min_length
            else:
                median_duration_samples = np.median(duration_in_samples)
                buffer_samples = int((buffer_ms / 1000) * 16000)
                
                base_length = round(median_duration_samples / 1000) * 1000
                calculated_length = int(base_length + buffer_samples)

                # Apply constraints
                if calculated_length < min_length:
                    final_length = min_length
                elif abs(calculated_length - min_length) <= snap_tolerance:
                    final_length = min_length
                else:
                    final_length = calculated_length
            
            config["total_length"] = final_length
            print_info(f"Optimal clip duration autotuned to: {final_length} samples ({final_length/16000:.2f} seconds).")
        
        else:
            # Priority 3: Autotune is explicitly disabled, and no fixed length was given.
            fallback_length = autotune_cfg.get("min_allowable_length", 32000)
            config["total_length"] = fallback_length
            print_info(f"Autotuning is disabled. Using fallback clip duration: {fallback_length} samples.")

    ISoverwrite = config.get("overwrite", False)
    transform_data = config.get("transform_clips", False)

    if args.transform_clips is True or transform_data:

        generation_manifest = config.get("feature_generation_manifest")

        if not generation_manifest:
            print_info("[INFO] 'feature_generation_manifest' not found in config.yaml. Skipping custom feature generation.")
        else:
            # print_step_header("Activating Flexible Feature Generation Engine")
            print_step_header("Computing Acoustic Features from Audio Sources")

            for job_name, recipe in generation_manifest.items():
                print_info(f"Running Generation: {job_name}")

                output_filename = recipe.get("output_filename")
                if not output_filename:
                    print_info(f"[WARNING] Skipping job '{job_name}' because 'output_filename' is missing.")
                    continue

                output_filepath = os.path.join(feature_save_dir, output_filename)

                if os.path.exists(output_filepath) and not (args.overwrite or ISoverwrite):
                    print_info(f"[INFO] Feature file '{output_filename}' already exists. Skipping generation. (Use --overwrite to force regeneration)")
                    continue

                input_audio_dirs = recipe.get("input_audio_dirs", [])
                if not input_audio_dirs:
                    print_info(f"[WARNING] Skipping job '{job_name}' because 'input_audio_dirs' is empty or missing.")
                    continue

                input_clips = []
                for d in input_audio_dirs:
                    input_clips.extend([str(p) for p in Path(d).rglob("*.wav")])
                
                if not input_clips:
                    print_info(f"[WARNING] Skipping job '{job_name}' as no .wav files were found in the specified directories.")
                    continue
                
                print_info(f"Found {len(input_clips)} source audio files.")

                
                global_aug_proxy = config.get("augmentation_settings", {})
                recipe_aug_proxy = recipe.get("augmentation_settings", {})

                # Converting ConfigProxy objects to plain dictionaries
                # The .to_dict attribute holds the actual dictionary inside the ConfigProxy.
                global_aug_dict = global_aug_proxy.to_dict() if global_aug_proxy else {}
                recipe_aug_dict = recipe_aug_proxy.to_dict() if recipe_aug_proxy else {}

                final_aug_settings = {**global_aug_dict, **recipe_aug_dict}          
                
                use_bg = recipe.get("use_background_noise", True)
                use_rir = recipe.get("use_rir", True)

                bg_paths_for_job = background_paths if use_bg else []
                rir_paths_for_job = rir_paths if use_rir else []

                aug_rounds = recipe.get("augmentation_rounds", 1)
                clips_to_generate = input_clips * aug_rounds
                total_clips_to_generate = len(clips_to_generate)
                
                print_info(f"Augmentation rounds: {aug_rounds}. Total clips to generate: {total_clips_to_generate}")

                audio_generator = augment_clips(
                    clip_paths=clips_to_generate,
                    total_length=config["total_length"],
                    batch_size=config["augmentation_batch_size"],
                    background_clip_paths=bg_paths_for_job,
                    RIR_paths=rir_paths_for_job,
                    augmentation_settings=final_aug_settings
                )
                
                # print(f"Computing features'{output_filename}'...")
                n_cpus = os.cpu_count()
                cpu_usage_ratio = config.get("feature_gen_cpu_ratio", 0.6)
                n_cpus = max(1, int(n_cpus * cpu_usage_ratio))
                 
                feature_extractor = AudioFeatures(device="gpu" if torch.cuda.is_available() else "cpu")
                sample_embedding_shape = feature_extractor.get_embedding_shape(config["total_length"] / 16000)
                output_shape = (total_clips_to_generate, sample_embedding_shape[0], sample_embedding_shape[1])
                
                fp = np.lib.format.open_memmap(output_filepath, mode='w+', dtype=np.float32, shape=output_shape)
                
                row_counter = 0
                batch_size = config.get('augmentation_batch_size', 128)
                pbar = tqdm(audio_generator, total=-(total_clips_to_generate // -batch_size), desc=f"{job_name}")

                for audio_batch in pbar:
                    if row_counter >= total_clips_to_generate: break
                    features = feature_extractor.embed_clips(audio_batch, batch_size=len(audio_batch), ncpu=n_cpus)
                    end_index = min(row_counter + features.shape[0], total_clips_to_generate)
                    fp[row_counter:end_index, :, :] = features[:end_index - row_counter]
                    row_counter = end_index
                    fp.flush()
                
                del fp
                from nanowakeword.data import trim_mmap
                trim_mmap(output_filepath)
                
                print_info(f"{job_name} Completed Successfully!")
            
            print_info("Flexible Feature Generation Finished")

    else:
        print_info("Feature generation is disabled as 'transform_clips' is false and '--transform_clips' flag is not set.")


    should_train = config.get("train_model", False)
    if args.train_model is True or should_train:
        training_start_time = time.time()

        # Get all positive and negative file paths from .yaml
        manifest = config.get("feature_manifest", {})
        positive_paths = list(manifest.get("targets", {}).values())

        negative_paths = list(manifest.get("negatives", {}).values())
        negative_paths.extend(list(manifest.get("backgrounds", {}).values()))

        positive_paths = [p for p in positive_paths if p]
        negative_paths = [p for p in negative_paths if p]

        if not positive_paths or not negative_paths:
            raise ValueError("CRITICAL: 'targets' and at least one of 'negatives' or 'backgrounds' must be defined in the feature_manifest.")

        dataset = WakeWordDataset(
            positive_feature_paths=positive_paths,
            negative_feature_paths=negative_paths
        )

        if len(dataset) == 0:
            raise ValueError("CRITICAL: Dataset is empty. Check your feature file paths in the manifest.")

        sampler = HardSampleFilterSampler(dataset)

        # DataLoader
        num_workers = config.get("num_workers", 2)
        X_train = DataLoader(
            dataset,
            batch_size=config.get('batch_size', 128),
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True if num_workers > 0 else False,
            persistent_workers=True if num_workers > 0 else False,
            # drop_last=True 
        )

        # Shape deteced
        try:
            _, sample_feature, _ = dataset[0]
            input_shape = sample_feature.shape
            seconds_per_example = (1280 * input_shape[0]) / 16000 
            print_info(f"Input Shape Detected: {input_shape} ({seconds_per_example:.2f}s context)")
        except Exception as e:
            print_info(f"[ERROR] Data integrity check failed: {e}")
            sys.exit(1)
        

        # MODEL INITIALIZATION
        print_info("Initializing Neural Architecture...")
        
        nww = Model(
            n_classes=1, 
            input_shape=input_shape,
            config=config,
            model_name=config.get("model_name", GNMV(config.get("model_type", "dnn"))),
            model_type=config.get("model_type", "dnn"),
            layer_dim=config["layer_size"],
            n_blocks=config["n_blocks"],
            dropout_prob=config.get("dropout_prob", 0.5),
            seconds_per_example=seconds_per_example
        )
      
        nww.setup_optimizer_and_scheduler(config=config)
        
        # EXECUTE TRAINING
        print_step_header("Training is progress")
        
        best_model = nww.auto_train(
            X_train=X_train,
            dataset=dataset,
            steps=config.get("steps", 15000),
            debug_path=artifacts_dir,
            table_updater=dynamic_table,
            resume_from_dir=args.resume 
        )

        nww.plot_history(artifacts_dir)
        
        training_end_time = time.time()
        training_duration_minutes = (training_end_time - training_start_time) / 60

        nww.export_model(
            model=best_model, 
            model_name=config.get("model_name", GNMV(config.get("model_type", "dnn"))), 
            output_dir=model_save_dir
        )

        nww.export_pytorch_model(
            model=best_model,
            model_name=config.get("model_name", GNMV(config.get("model_type", "dnn"))),
            output_dir=model_save_dir
        )

        if config.get("enable_journaling", True):
            final_metrics = {}
            if nww.history.get("final_report"):
                report = nww.history["final_report"]
                final_metrics["Stable Loss"] = report.get("Average Stable Loss", "N/A")
                final_metrics["Avg. Pos Conf"] = report.get("Avg. Positive Score (Logit)", "N/A")
                final_metrics["Avg. Neg Conf"] = report.get("Avg. Negative Score (Logit)", "N/A")

            final_metrics["Train Time"] = f"{training_duration_minutes:.1f}"
            base_output_directory = os.path.abspath(config["output_dir"])

            update_training_journal(
                base_output_dir=base_output_directory,
                model_name=config.get("model_name", GNMV(config.get("model_type", "dnn"))),
                metrics=final_metrics,
                current_config=config.report()
            )

if __name__ == '__main__':
    train()