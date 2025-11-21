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
import sys
import math
import yaml
import copy
import scipy
import torch
import random
import logging
import warnings
import argparse
import torchinfo
import matplotlib
import collections
import numpy as np
matplotlib.use('Agg')
from tqdm import tqdm
from pathlib import Path
from torch import optim, nn
import matplotlib.pyplot as plt
from nanowakeword.utils.audio_processing import compute_features_from_generator
from nanowakeword.data import augment_clips, mmap_batch_generator, generate_adversarial_texts
from nanowakeword.utils.logger import print_banner, print_step_header, print_info, print_key_value, print_final_report_header, print_table
# All Architectures
from .architectures import (
    CNNModel, LSTMModel, Net, GRUModel, RNNModel, TransformerModel, 
    CRNNModel, TCNModel, QuartzNetModel, ConformerModel, EBranchformerModel
)

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


import collections.abc

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


class TripletLoss(nn.Module):
    """
    Triplet loss function.
    """
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss, as described in https://arxiv.org/abs/1708.02002.

    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha (float): Weighting factor for the rare class.
            gamma (float): Focusing parameter to down-weight easy examples.
            reduction (str): 'mean', 'sum' or 'none'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Forward pass.

        Args:
            inputs (torch.Tensor): The model's raw logits.
            targets (torch.Tensor): The ground truth labels.
        """
        # Ensure inputs and targets are flattened to handle any shape
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Calculate BCE Loss without reduction
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Calculate pt (the probability of the correct class)
        pt = torch.exp(-BCE_loss)
        
        # Calculate Focal Loss
        F_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss

        # Apply reduction
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else: # 'none'
            return F_loss


class LabelSmoothingBCELoss(nn.Module):
    """
    Binary Cross Entropy with Label Smoothing.
    Handles potential shape mismatches between inputs and targets.
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingBCELoss, self).__init__()
        self.smoothing = smoothing
    
    def forward(self, inputs, targets):

        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        smoothed_targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        
        return nn.functional.binary_cross_entropy_with_logits(inputs, smoothed_targets)


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

        # Training progress tracking attributes 
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

        # ================= STABILITY WARNING =================
        if model_type.lower() in {"quartznet", "conformer", "e_branchformer", "crnn"}:
            print(f"\n[WARNING] The '{model_type.upper()}' architecture is highly sensitive to hyperparameters and may exhibit convergence instability.\n")
        # =====================================================

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
            raise ValueError(
                f"Unsupported model_type: '{model_type}'. "
                "Supported types are: dnn, lstm, gru, rnn, cnn, transformer, crnn, tcn, quartznet, conformer, e_branchformer."
            )


        triplet_margin = config.get("triplet_loss_margin", 0.2)
        label_smooth = config.get("label_smoothing", 0.1)
    
        self.classifier = nn.Linear(embedding_dim, n_classes)
        self.triplet_loss = TripletLoss(margin=triplet_margin)

        # Define logging dict (in-memory)
        self.history = collections.defaultdict(list)

        # Define optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=1e-5) # <-- weight_decay 

        #  Select the classification loss function based on configuration 

        # Get the desired loss type from config.
        # If not specified at all, it will be None.
        classification_loss_type = self.config.get("classification_loss", None)

        # Logic to select the loss function 
        if classification_loss_type == "focalloss":
            # Option 1: User explicitly chooses 'focalloss'
            # print_info("Using FocalLoss for the classification task.")
            print("WARNING: Using FocalLoss for the classification task may impact performance")
            
            focal_alpha = self.config.get("focal_loss_alpha", 0.25)
            focal_gamma = self.config.get("focal_loss_gamma", 2.0)
            
            self.classification_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

        elif classification_loss_type == "bce":
            # Option 2: User explicitly chooses 'bce' (standard Binary Cross Entropy)
            # print_info("Using standard BCEWithLogitsLoss for the classification task.")
            print("WARNING: Using standard BCEWithLogitsLoss for the classification task may impact performance")
            
            self.classification_loss = nn.BCEWithLogitsLoss()

        elif classification_loss_type == "labelsmoothing" or classification_loss_type is None:
            # Option 3 (DEFAULT):
            # This block runs if the user explicitly chooses 'labelsmoothing'
            # OR if the user does not specify `classification_loss` at all (it will be None).
            
            # We check for None to make the info message more accurate.
            if classification_loss_type is None:
                print_info("Using LabelSmoothingBCELoss for the classification task.")
                # Ensure the default is tracked for the table display
                self.config['classification_loss'] = "labelsmoothing"
            else:
                print_info("Using LabelSmoothingBCELoss for the classification task.")

            label_smooth = self.config.get("label_smoothing", 0.1)
            
            self.classification_loss = LabelSmoothingBCELoss(smoothing=label_smooth)

        else:
            # Handle invalid user input
            raise ValueError(
                f"Unsupported classification_loss: '{classification_loss_type}'. "
                "Supported types are: 'focalloss', 'bce', 'labelsmoothing'."
            )


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
            print("\nGenerating training performance graph...")
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
            
            print(f"Performance graph saved to: {save_path}")



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


    def auto_train(self, X_train, steps, table_updater, debug_path, resume_from_dir=None):
            """
            A modern, single-sequence training process that combines the best checkpoints to
              create a final and robust model.
            """

            self.train_model(
                X=X_train,
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
                    # Unpacking data from a triplet batch
                    anchor_batch, _, negative_batch, anchor_labels, negative_labels = next(iter(X_train))
                    
                    # Setting up the device properly
                    final_model.to(self.device)
                    confidence_batch_x = torch.cat([anchor_batch, negative_batch]).to(self.device)
                    confidence_batch_y = torch.cat([anchor_labels, negative_labels]).to(self.device)

                    predictions = final_model(confidence_batch_x).squeeze()
                    
                    pos_preds = predictions[confidence_batch_y.squeeze() == 1]
                    neg_preds = predictions[confidence_batch_y.squeeze() == 0]

                    avg_pos_confidence = pos_preds.mean().item() if pos_preds.numel() > 0 else -99.0
                    avg_neg_confidence = neg_preds.mean().item() if neg_preds.numel() > 0 else 99.0

                    final_results["Avg. Positive Confidence (Logit)"] = f"{avg_pos_confidence:.3f}"
                    final_results["Avg. Negative Confidence (Logit)"] = f"{avg_neg_confidence:.3f}"
            except (StopIteration, RuntimeError) as e:
                final_results["Confidence Score"] = f"N/A (Error: {e})"

            print_final_report_header()
            print_info("NOTE: These metrics are indicators of model health, not real-world performance.")

            for key, value in final_results.items():
                print_key_value(key, value)
    
            # Returning the completed and averaged model
            return final_model
    
    def export_model(self, model, model_name, output_dir):
        """
        Exports the final trained model to a standard, inference-ready ONNX format.

        This function ensures hardware independence by moving both the model and a
        dummy input to the CPU before export. It also guarantees a standardized
        output shape of [batch_size, 1, 1] for maximum compatibility.
        """
        # Ensure the model is in a sequential format for export
        if not isinstance(model, nn.Sequential):
            print_info("Reconstructing model into a sequential format for export.")
            model = nn.Sequential(self.model, self.classifier)

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
        dummy_input = torch.rand(1, *self.input_shape)
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


    def _perform_train_step(self, data, step_ndx, logger): 
                """
                Advanced Training Step with Hard Negative Mining & Full Debugging.
                """
                # Preparing data
                anchor, positive, negative, labels_anchor, labels_negative = data

                # ===================== DEBUG BLOCK 1: DATA & LABELS ========================
                log_interval = int(self.config.get("log_interval", 1000))
                debug_mode = self.config.get("debug_mode", False)
                
                if debug_mode and step_ndx % log_interval == 0:
                    logger.info(f"\n\n[DEBUG] Step {step_ndx}: Data and Labels Check") 
                    logger.info(f"Anchor Batch Shape: {anchor.shape}")  
                    unique_anchors, anchor_counts = torch.unique(labels_anchor, return_counts=True)
                    unique_negatives, negative_counts = torch.unique(labels_negative, return_counts=True)
                    logger.info(f"Unique Anchor Labels: {unique_anchors.cpu().numpy()} (Count: {anchor_counts.cpu().numpy()})")  
                    logger.info(f"Unique Negative Labels: {unique_negatives.cpu().numpy()} (Count: {negative_counts.cpu().numpy()})")  
                # ========================== END DEBUG BLOCK 1 =====================================

                # Move to Device
                anchor, positive, negative = anchor.to(self.device), positive.to(self.device), negative.to(self.device)
                labels_anchor, labels_negative = labels_anchor.to(self.device), labels_negative.to(self.device)

                # Forward Pass
                self.optimizer.zero_grad()
                
                # Get Embeddings
                emb_anchor = self.model(anchor)
                emb_positive = self.model(positive)
                emb_negative = self.model(negative)

                # 1. Triplet Loss (Distance Based)
                loss_triplet = self.triplet_loss(emb_anchor, emb_positive, emb_negative)
                
                # 2. Classification (Logits)
                logits_anchor = self.classifier(emb_anchor)
                logits_negative = self.classifier(emb_negative)
 
                penalty_loss = torch.tensor(0.0, device=self.device)
                num_hard_negatives = 0

                with torch.no_grad():
                    probs_neg = torch.sigmoid(logits_negative).view(-1)
                    hard_indices = probs_neg > 0.5
                    num_hard_negatives = hard_indices.sum().item()

                if num_hard_negatives > 0:
                    hard_logits = logits_negative.view(-1)[hard_indices]
                    hard_labels = labels_negative.view(-1)[hard_indices]
                    
                    bce_penalty_fn = nn.BCEWithLogitsLoss()
                    hard_loss = bce_penalty_fn(hard_logits, hard_labels)
                    
                    penalty_loss = hard_loss * 2.0

                # Standard Classification Loss
                all_logits = torch.cat([logits_anchor, logits_negative])
                all_labels = torch.cat([labels_anchor, labels_negative])
                
                loss_class = self.classification_loss(all_logits, all_labels)

                # Add the Penalty to the classification loss
                final_class_loss = loss_class + penalty_loss

                if debug_mode and step_ndx % log_interval == 0:
                    logger.info(f"\n[DEBUG] Step {step_ndx}: Model Confidence & Hard Negatives")
                    
                    # Calculate average confidences
                    avg_pos_conf = torch.sigmoid(logits_anchor).mean().item()
                    avg_neg_conf = torch.sigmoid(logits_negative).mean().item()
                    
                    logger.info(f"Avg. Positive Confidence: {avg_pos_conf:.4f}")  
                    logger.info(f"Avg. Negative Confidence: {avg_neg_conf:.4f}")
                    
                    if num_hard_negatives > 0:
                        logger.info(f"ALERT: Found {num_hard_negatives} Hard Negatives in this batch!")
                        logger.info(f"  -> Penalty Loss added: {penalty_loss.item():.6f}")
                    else:
                        logger.info("No Hard Negatives found (Good job!)")
                # ========================== END DEBUG BLOCK 2 ==================================

                # Total Loss Weights (Adjusted for better stability)
                loss_weight_triplet = self.config.get("loss_weight_triplet", 0.4) 
                loss_weight_class = self.config.get("loss_weight_class", 1.0)
                
                total_loss = (loss_triplet * loss_weight_triplet) + (final_class_loss * loss_weight_class)

                # ========================== DEBUG BLOCK 3: LOSS VALUES ==========================
                if debug_mode and step_ndx % log_interval == 0:
                    logger.info(f"\n[DEBUG] Step {step_ndx}: Final Loss Components")  
                    logger.info(f"Triplet Loss (raw):      {loss_triplet.item():.6f}")  
                    logger.info(f"Class Loss (raw):        {loss_class.item():.6f}")
                    logger.info(f"Penalty Loss (added):    {penalty_loss.item():.6f}")
                    logger.info(f"TOTAL WEIGHTED LOSS:     {total_loss.item():.6f}")  
                # ========================== END DEBUG BLOCK 3 ===================================

                # Backward Pass
                total_loss.backward()

                # ========================== DEBUG BLOCK 4: GRADIENTS ===========================
                if debug_mode and  step_ndx % log_interval == 0:
                    first_param_grad = next(self.model.parameters()).grad
                    first_layer_grad = first_param_grad.mean().item() if first_param_grad is not None else 0.0
                    classifier_grad = self.classifier.weight.grad.mean().item() if self.classifier.weight.grad is not None else 0.0

                    logger.info(f"\n[DEBUG] Step {step_ndx}: Gradient Health")  
                    logger.info(f"First Layer Grad Mean: {first_layer_grad:.8f}")  
                    logger.info(f"Classifier Grad Mean:  {classifier_grad:.8f}")                
                # ========================== END DEBUG BLOCK 4 ==================================

                default_max_norm = self.config.get("max_norm", 1.0)
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=default_max_norm)
                
                self.optimizer.step()            
                self.scheduler.step()
                
                return total_loss.detach().cpu().item()

    def train_model(self, X, max_steps, log_path, table_updater, resume_from_dir=None):
                
                import logging
                from logging.handlers import RotatingFileHandler
                import os
                import re # Regular expressions for parsing filenames

                # 1. INITIAL SETUP 
                # This section remains the same, preparing logging.
                debug_mode = self.config.get("debug_mode", False)
                log_dir = os.path.join(log_path, "training_debug")
                os.makedirs(log_dir, exist_ok=True)
                debug_log_file = os.path.join(log_dir, "training_debug.log")
                if debug_mode:
                    logger = logging.getLogger("NanoTrainerDebug")
                    logger.setLevel(logging.INFO)
                    if not logger.handlers:
                        handler = RotatingFileHandler(debug_log_file, maxBytes=5_000_000, backupCount=30)
                        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
                        handler.setFormatter(formatter)
                        logger.addHandler(handler)
                    logger.propagate = False
                    print_info(f"Debug mode ON. Logs will be saved to:\n{debug_log_file}")
                else:
                    logger = logging.getLogger("NanoTrainerDebug")
                    logger.disabled = True
                    # print_info("Debug mode OFF. Logging disabled.")

                # CHECKPOINTING CONFIGURATION 
                # Reads the 'checkpointing' section from config.yaml.
                # If the section doesn't exist, it defaults to being disabled.
                checkpoint_cfg = self.config.get("checkpointing", {})
                checkpointing_enabled = checkpoint_cfg.get("enabled", False)
                checkpoint_interval = checkpoint_cfg.get("interval_steps", 1000)
                checkpoint_limit = checkpoint_cfg.get("limit", 3)
                checkpoint_dir = os.path.join(log_path, "checkpoints")
                if checkpointing_enabled:
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    print_info(f"Checkpointing is ENABLED. A checkpoint will be saved every {checkpoint_interval} steps.")
                

                # Training parameter setup -
                # This section also remains the same.
                ema_loss = None
                self.best_training_checkpoints = [] 
                self.best_training_scores = []
                checkpoint_averaging_top_k= self.config.get("checkpoint_averaging_top_k", 5)
                default_warmup_steps = int(max_steps * 0.15)
                WARMUP_STEPS = self.config.get("WARMUP_STEPS", default_warmup_steps)
                min_delta = self.config.get("main_delta", 0.0001)
                best_ema_loss_for_stopping = float('inf')
                steps_without_improvement = 0
                ema_alpha = self.config.get("ema_alpha", 0.01)

                # Default patience value
                default_patience_steps = int(max_steps * 0.15)

                # Check if user has explicitly provided early_stopping_patience
                user_patience = self.config.get("early_stopping_patience", None)

                if user_patience is not None:
                    patience = user_patience  # user override
                elif self.config["steps"] < 3000:
                    patience = 0  # auto disable for short trainings
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
                data_iterator = iter(X)

                if resume_from_dir:
                    # Construct the path to the checkpoints folder within the specified resume directory
                    resume_checkpoint_dir = os.path.join(resume_from_dir, "2_training_artifacts", "checkpoints")
                    print_info(f"Attempting to resume training from: {resume_checkpoint_dir}")
                    
                    if os.path.exists(resume_checkpoint_dir):
                        # Find all checkpoint files in the directory
                        checkpoints = [f for f in os.listdir(resume_checkpoint_dir) if f.startswith("checkpoint_step_") and f.endswith(".pth")]
                        if checkpoints:
                            # Robustly find the latest checkpoint by parsing the step number from the filename
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
                                
                                # Restore the complete state
                                self.load_state_dict(checkpoint['model_state_dict'])
                                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                                
                                start_step = checkpoint.get('step', 0)
                                ema_loss = checkpoint.get('ema_loss', None)
                                steps_without_improvement = checkpoint.get('steps_without_improvement', 0)
                                best_ema_loss_for_stopping = checkpoint.get('best_ema_loss_for_stopping', float('inf'))
                                self.history['loss'] = checkpoint.get('loss_history', [])

                                print_info(f"Successfully restored state. Resuming training from step {start_step + 1}.")
                                
                                # CRITICAL: Fast-forward the data iterator to the correct position
                                print_info("Synchronizing data stream to the restored step...")
                                for _ in tqdm(range(start_step + 1), desc="Fast-forwarding data", unit="steps", leave=False):
                                    next(data_iterator, None)
                            else:
                                print_info("WARNING: Checkpoint files found, but their names are not in the expected format. Starting fresh.")
                        else:
                            print_info("WARNING: No valid checkpoint files found in the directory. Starting fresh.")
                    else:
                        print_info(f"WARNING: Checkpoint directory not found at '{resume_checkpoint_dir}'. Starting fresh.")
                
                # The "dry run" for the first step ONLY happens if we are NOT resuming.
                if start_step == 0:
                    try:
                        first_batch = next(data_iterator)
                        initial_loss = self._perform_train_step(first_batch, step_ndx=0, logger=logger)
                        self.history["loss"].append(initial_loss)
                        ema_loss = initial_loss
                        start_step = 1 # The main loop will now correctly start from step 1
                    except StopIteration:
                        print("\n[ERROR] Data source is empty. Cannot start training.")
                        return
                
                table_updater.update(force_print=True)

                # The main training loop now correctly starts from `start_step`.
                training_loop = tqdm(data_iterator, total=max_steps, desc="Training", initial=start_step)
                for step_ndx, data in enumerate(training_loop, start=start_step):
                    
                    current_loss = self._perform_train_step(data, step_ndx=step_ndx, logger=logger)
                    self.history["loss"].append(current_loss)
                    
                    if ema_loss is None: ema_loss = current_loss
                    ema_loss = ema_alpha * current_loss + (1 - ema_alpha) * ema_loss

                    # The logic for checkpoint averaging (best models) remains the same
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

                    # Early stopping logic remains the same
                    if patience > 0:
                        if ema_loss < best_ema_loss_for_stopping - min_delta:
                            best_ema_loss_for_stopping = ema_loss
                            steps_without_improvement = 0
                        else:
                            steps_without_improvement += 1
                        
                        if step_ndx > WARMUP_STEPS and steps_without_improvement >= patience:
                            print(f"\nINFO: Early stopping triggered at step {step_ndx}. No improvement in stable loss for {patience} steps.")
                            break

                    if checkpointing_enabled and step_ndx > 0 and step_ndx % checkpoint_interval == 0:
                        # Gather all necessary data to save a complete checkpoint
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
                        
                        # Manage checkpoint limit: Keep only the latest N checkpoints
                        all_checkpoints = sorted(
                            [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_step_")],
                            # Sort numerically, not alphabetically (so step_10000 comes after step_2000)
                            key=lambda f: int(re.search(r"(\d+)", f).group(1))
                        )
                        if len(all_checkpoints) > checkpoint_limit:
                            os.remove(os.path.join(checkpoint_dir, all_checkpoints[0]))
                        
                    if step_ndx >= max_steps - 1:
                        break


class ConfigProxy:
    def __init__(self, data: dict, root_proxy=None, prefix=""):
        self._data = data
        self._root = root_proxy if root_proxy is not None else self
        self._prefix = prefix
        
        if self._root is self:
            self.used_params = {}
            self._accessed_keys = set()

    def _track_access(self, key, value):
        full_key = self._prefix + key
        # Do not track whole dictionaries, only their final scalar values
        if not isinstance(value, (dict, ConfigProxy)):
            if full_key not in self._root._accessed_keys:
                self._root.used_params[full_key] = value
                self._root._accessed_keys.add(full_key)

    def get(self, key: str, default=None):
        """
        Gets a value. 
        - If the value is a dictionary, returns a new ConfigProxy for it.
        - If a default dictionary is provided and the key is not found,
          it uses the default dictionary and tracks all its items.
        """
        # The Ultimate Solution is here 
        if key in self._data:
            value = self._data[key]
            # Track the access, even for dicts, to know the section was requested
            self._track_access(key, value)
            
            if isinstance(value, dict):
                new_prefix = f"{self._prefix}{key}."
                return ConfigProxy(value, root_proxy=self._root, prefix=new_prefix)
            else:
                return value
        
        # If key is not in the data, handle defaults
        elif isinstance(default, dict):
            # This is the magic for merging defaults automatically
            new_prefix = f"{self._prefix}{key}."
            # Create a proxy for the default dict and track all its items
            default_proxy = ConfigProxy(default, root_proxy=self._root, prefix=new_prefix)
            for k, v in default_proxy.items(): # .items() will trigger tracking
                pass
            return default_proxy
        else:
            self._track_access(key, default)
            return default

    def get_merged_dict(self, key: str, defaults: dict) -> dict:
        """
        Gets a dictionary section, merges it over a default dictionary,
        and tracks every key from the final result.
        This is the definitive way to handle partial user overrides.
        """
        final_dict = defaults.copy()
        
        user_proxy = self.get(key, {}) 
        
        if user_proxy and isinstance(user_proxy, ConfigProxy):
            final_dict.update(user_proxy._data)
        
        for sub_key in final_dict.keys():
            self.get(key, {}).get(sub_key, final_dict[sub_key])
            
        return final_dict

    def __getitem__(self, key):
        if key not in self._data:
            raise KeyError(f"Key '{self._prefix}{key}' not found in configuration.")
        value = self._data[key]
        if isinstance(value, dict):
            new_prefix = f"{self._prefix}{key}."
            return ConfigProxy(value, root_proxy=self._root, prefix=new_prefix)
        else:
            self._track_access(key, value)
            return value

    def __setitem__(self, key, value):
        self._data[key] = value
        self._track_access(key, value)

    def report(self) -> dict:
        return self._root.used_params

    def __contains__(self, key):
        return key in self._data

    def items(self):
        for key in self._data.keys():
            yield key, self[key]



import sys
from io import StringIO

class DynamicTable:
    def __init__(self, config_proxy, title="Training Configuration", enabled: bool = True):
        self._proxy = config_proxy
        self._title = title
        self._print_func = print_table 
        self._last_state = {}
        self._last_table_height = 0

        self._is_enabled = enabled


    def _move_cursor_up(self, lines):
        """Moves the terminal cursor up by a specified number of lines."""
        if lines > 0:
            sys.stdout.write(f'\x1b[{lines}A')
            sys.stdout.flush()

    def _clear_from_cursor(self):
        """Clears the screen from the current cursor position to the end."""
        sys.stdout.write('\x1b[J')
        sys.stdout.flush()

    def update(self, force_print=False):
        """
        Updates the table in-place, but only if it's enabled.
        """
        if not self._is_enabled:
            return 

        current_state = self._proxy.report()
        
        if current_state != self._last_state or force_print:
            self._move_cursor_up(self._last_table_height)
            self._clear_from_cursor()

            display_config = {}
            keys_to_exclude = {
                "positive_data_path", "negative_data_path", "background_paths",
                "rir_paths", "output_dir", "force_verify"
            }
            sorted_keys = sorted(current_state.keys())
            for key in sorted_keys:
                if key not in keys_to_exclude and current_state[key] is not None:
                    display_config[key] = str(current_state[key])
            
            old_stdout = sys.stdout
            sys.stdout = string_io = StringIO()
            self._print_func(display_config, self._title)
            table_string = string_io.getvalue()
            sys.stdout = old_stdout 
            
            self._last_table_height = table_string.count('\n') + 1

            print(table_string, end='')
            
            self._last_state = current_state.copy()



def train(cli_args=None):
    
    parser = argparse.ArgumentParser(
        description="NanoWakeWord: The Intelligent Wake Word Training Framework.",
        formatter_class=argparse.RawTextHelpFormatter # For better help message formatting
    )

    # --- Configuration ---
    parser.add_argument(
            "-c", "--config_path",
            help="Path to the training configuration YAML file. (Required)",
            type=str,
            required=True,
            metavar="PATH"
        )

    # --- Pipeline Stages (Primary Actions) ---
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

    # --- Modifiers ---
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

    # NEW 
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


    import json
    from nanowakeword.data_utils.preprocess import verify_and_process_directory

    # 1. Define a stable cache directory based on the user's output_dir
    #    This ensures the path is known and available from the very beginning.
    output_dir_from_config = user_config.get("output_dir", "./trained_models")
    VERIFICATION_CACHE_DIR = os.path.join(output_dir_from_config, ".cache", "verification_receipts")
    os.makedirs(VERIFICATION_CACHE_DIR, exist_ok=True)

    # Define these file names here to avoid magic strings
    VERIFICATION_RECEIPT_FILENAME_TEMPLATE = "{hash}.json" # We'll use a hash now


    import base64
    import hashlib
    import json

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



    print_step_header( 1,"Verifying and Preprocessing Data Directories")
    
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
   

    from nanowakeword.analyzer import DatasetAnalyzer
    from nanowakeword.config_generator import ConfigGenerator


    # Hardware-Only Configuration (Express Pass) 
    print_info("Determining hardware-specific configurations...")
    try:
        generator1 = ConfigGenerator() 
        intelligent_config1 = generator1.generate()

        final_config1 = intelligent_config1.copy()
        final_config1.update(user_config)

        base_config = final_config1

    except Exception as e:
        print(f"Could not generate intelligent hardware config due to an error: {e}. Proceeding with user config only.")
        base_config = user_config.copy() 

    ISgenaret_data = base_config.get("generate_clips", False)
    if args.generate_clips or ISgenaret_data:
        from nanowakeword.generate_samples import generate_samples
        print_step_header(2, "Activating Synthetic Data Generation Engine")

        # Acquire the Target Phrase
        target_phrase = base_config.get("target_phrase")
        if not target_phrase:
            print("\n" + "=" * 80)
            print("[CONFIGURATION NOTICE]: 'target_phrase' is not set in your config file.")
            print("This is required to generate audio samples.")
            print("=" * 80)
            try:
                user_input = input(">>> Please enter the target phrase to proceed: ").strip()
                if not user_input:
                    print("\n[ABORT] A target phrase is mandatory for generation. Exiting.")
                    sys.exit(1)
                target_phrase = [user_input]
                print_info(f"Using runtime target phrase: '{user_input}'")
            except (KeyboardInterrupt, EOFError):
                print("\n\nOperation cancelled by user.")
                sys.exit()


        # Define Data Generation Plan 
        user_pos_samples = base_config.get('generate_positive_samples')
        user_neg_samples = base_config.get('generate_negative_samples')
        
        if user_pos_samples is not None:
            n_pos_train = int(user_pos_samples)
        
        if user_neg_samples is not None:
            n_neg_train = int(user_neg_samples)

        # USER CONTROLLED CUSTOM NEGATIVES 
        custom_negatives = base_config.get("custom_negative_phrases", [])
        
        target_custom_count = int(base_config.get("custom_negative_count", 200))
        
        final_negative_texts = []

        if custom_negatives:            
            import math
            if len(custom_negatives) > 0:
                repeats = math.ceil(target_custom_count / len(custom_negatives))
                repeated_customs = (custom_negatives * repeats)[:target_custom_count]
                final_negative_texts.extend(repeated_customs)
        
        remaining_count = max(0, n_neg_train - len(final_negative_texts))
        if remaining_count > 0:
            auto_adversarial = generate_adversarial_texts(target_phrase[0], N=remaining_count)
            final_negative_texts.extend(auto_adversarial)
        
        pos_batch_size = base_config.get("tts_batch_size", 256)
        
        if "tts_batch_size_negative" in base_config:
            neg_batch_size = int(base_config["tts_batch_size_negative"])
        else:
            neg_batch_size = max(1, pos_batch_size // 2)

        # A unified structure for all generation tasks
        generation_plan = {
            "Positive_Train": {
                "count": n_pos_train,
                "texts": target_phrase,
                "output_dir": base_config["positive_data_path"],
                "batch_size": pos_batch_size
            },            
            "Adversarial_Train": {
                "count": n_neg_train,
                "texts": final_negative_texts,
                "output_dir": base_config["negative_data_path"],
                "batch_size": neg_batch_size # <--- Now Dynamic
            }
        }


        # Execute the Generation Plan 
        print_info(f"Initiating data generation pipeline for phrase: '{target_phrase[0]}'")
        for task_name, params in generation_plan.items():
            if params["count"] > 0 and params["texts"]:
                print_info(f"Executing task '{task_name}': {params['count']} clips -> '{params['output_dir']}'")
                os.makedirs(params["output_dir"], exist_ok=True)
                
                generate_samples(
                    text=params["texts"],
                    max_samples=params["count"],
                    output_dir=params["output_dir"],
                    batch_size=params["batch_size"]
                )
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        print_info("Synthetic data generation process finished successfully.\n")


    print_step_header(2.5, "Activating Intelligent Configuration Engine")
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
        print(f"ERROR: Missing essential path in config file for auto-config: {e}")
        exit()
    except Exception as e:
        print(f"Could not generate intelligent config due to an error: {e}. Proceeding with user config only.")
        intelligent_config = {} 


    final_config = intelligent_config.copy()
    
    # 2. Deep merge the user's configuration on top of it.
    #    This will correctly merge nested dictionaries like 'augmentation_settings'.
    final_config = deep_merge(final_config, user_config)
    
    # Now, `final_config` contains the correct, merged values.
    config_proxy = ConfigProxy(final_config)
    config = config_proxy

    show_table_flag = config.get("show_training_summary", True)
    dynamic_table = DynamicTable(config_proxy, title="Effective Training Configuration", enabled=show_table_flag)


    def GNMV(model_type: str, base_dir: str = ".", prefix: str = "nww"):
        """
        Automatically generate model name including type and maintain version professionally
        The version option will be kept the same, meaning if there is an older model, it will be overwritten.
        
        Example:
            nww_dnn_model_v1  → nww_dnn_model_v2  (if v1 already exists)
        Args:
            model_type (str): Type of the model (e.g. 'dnn', 'cnn', 'lstm').
            base_dir (str): Directory where model files are stored.
            prefix (str): Prefix used before model name (default 'nww').
        
        Returns:
            str: A unique, versioned model name.
        """
        import os
        import re
        # Normalize
        model_type = model_type.lower().strip()
        pattern = re.compile(rf"^{prefix}_{model_type}_model_v(\d+)$")

        # Find existing models in the directory
        existing = []
        for name in os.listdir(base_dir):
            match = pattern.match(name)
            if match:
                existing.append(int(match.group(1)))

        # Determine next version
        next_version = max(existing, default=0) + 1
        return f"{prefix}_{model_type}_model_v{next_version}"



    # Define and Create Professional Output Directory Structure  
    # project_dir = os.path.join(os.path.abspath(base_config["output_dir"]), base_config.get("model_name", AGMINTVP))
    project_dir = os.path.join(
        os.path.abspath(base_config["output_dir"]),
        base_config.get(
            "model_name",
            GNMV(model_type=config.get("model_type", "rnn"))
        )
    )


    feature_save_dir = os.path.join(project_dir, "1_features")
    artifacts_dir = os.path.join(project_dir, "2_training_artifacts")
    model_save_dir = os.path.join(project_dir, "3_model")

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
            
            # --- Sample clips and calculate median duration (robust version) ---
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


    # Do Data Augmentation
    ISoverwrite= config.get("overwrite", False)
    transform_data = config.get("transform_clips", False)
    if args.transform_clips is True or transform_data:
        if not os.path.exists(os.path.join(feature_save_dir, "positive_features_train.npy")) or args.overwrite is True or ISoverwrite:

            # Define the default probabilities.
            default_aug_probs = {
                "ParametricEQ": 0.25,
                "Distortion": 0.25,
                "PitchShift": 0.25,
                "BandStopFilter": 0.25,
                "ColoredNoise": 0.25,
                "BackgroundNoise": 0.75,
                "Gain": 1.0,
                "RIR": 0.5
            }

            # The Magic Happens Here 
            # 1. Get the user's settings as a proxy. The proxy handles missing keys gracefully.
            user_aug_proxy = config.get("augmentation_settings", {})

            # 2. Iterate through the complete list of default keys.
            #    For each key, get the value. The proxy will automatically provide the
            #    user's value if it exists, otherwise it will return the default.
            #    This ensures every single parameter is "touched" and tracked.
            final_aug_probs = {
                key: user_aug_proxy.get(key, default_value)
                for key, default_value in default_aug_probs.items()
            }

            # `final_aug_probs` is now a standard dictionary with correctly merged values,
            # and every augmentation parameter has been tracked by the ConfigProxy.
            # We will use this dictionary for the `augment_clips` function.

            # Let's rename the variable for clarity before passing it to the function.
            aug_probs = final_aug_probs


            positive_clips_train = [str(i) for i in Path(config["positive_data_path"]).glob("*.wav")]*config["augmentation_rounds"]
            positive_clips_train_generator = augment_clips(positive_clips_train, total_length=config["total_length"],
                                                           batch_size=config["augmentation_batch_size"],
                                                           min_snr_in_db=config['min_snr_in_db'],
                                                           max_snr_in_db=config['max_snr_in_db'],                                                           
                                                           background_clip_paths=background_paths,
                                                           RIR_paths=rir_paths,
                                                           augmentation_settings=aug_probs)


            negative_clips_train = [str(i) for i in Path(config["negative_data_path"]).glob("*.wav")]*config["augmentation_rounds"]
            negative_clips_train_generator = augment_clips(negative_clips_train, total_length=config["total_length"],
                                                           batch_size=config["augmentation_batch_size"],
                                                           min_snr_in_db=config['min_snr_in_db'],
                                                           max_snr_in_db=config['max_snr_in_db'],                                                           
                                                           background_clip_paths=background_paths,
                                                           RIR_paths=rir_paths,
                                                           augmentation_settings=aug_probs)

  
            # Compute features and save to disk via memmapped arrays
            print_step_header(3, "Computing Nanowakeword Features for Training Data")
            n_cpus = os.cpu_count()

            cpu_usage_ratio = config.get("feature_gen_cpu_ratio", 0.6) # 0.6 = 60%
            n_cpus = max(1, int(n_cpus * cpu_usage_ratio))

            # Generate positive feature
            compute_features_from_generator(positive_clips_train_generator, output_file=os.path.join(feature_save_dir, "positive_features_train.npy"),
                                            n_total=len(positive_clips_train), 
                                            clip_duration=config["total_length"],
                                            device="gpu" if torch.cuda.is_available() else "cpu",
                                            ncpu=n_cpus if not torch.cuda.is_available() else 1)

            # Generate negative feature
            compute_features_from_generator(negative_clips_train_generator, output_file=os.path.join(feature_save_dir, "negative_features_train.npy"),                                                                                        
                                            n_total=len(negative_clips_train),
                                            clip_duration=config["total_length"],
                                            device="gpu" if torch.cuda.is_available() else "cpu",
                                            ncpu=n_cpus if not torch.cuda.is_available() else 1)

            
            batch_comp_config = config.get('batch_composition')
            
            if not batch_comp_config:
                print_info("[CONFIG NOTICE] 'batch_composition' not found. Applying a robust default strategy.")
                batch_comp_config = {
                    'batch_size': 128,
                    'source_distribution': {'positive': 35, 'negative_speech': 38, 'pure_noise': 27}
                }
            
            source_dist = batch_comp_config.get('source_distribution', {})

            # Generate pure noise features if the strategy requires it and they don't already exist.
            # if source_dist.get('pure_noise', 0) > 0:
            pure_noise_output_path = os.path.join(feature_save_dir, "pure_noise_features.npy")
            
            if not os.path.exists(pure_noise_output_path) or args.overwrite or ISoverwrite:
                noise_source_paths = config.get("background_paths", [])
                pure_noise_clips = [str(i) for j in noise_source_paths for i in Path(j).glob("*.wav")]
                
                if pure_noise_clips:
                    noise_aug_rounds = max(1, config.get("augmentation_rounds", 5) // 2)
                    

                    pure_noise_generator = augment_clips(pure_noise_clips * noise_aug_rounds, total_length=config["total_length"], 
                                                            batch_size=config["augmentation_batch_size"],
                                                            background_clip_paths=background_paths,
                                                            RIR_paths=rir_paths,
                                                            augmentation_settings=aug_probs)
                    

                    compute_features_from_generator(pure_noise_generator, n_total=len(pure_noise_clips) * noise_aug_rounds,
                                                    clip_duration=config["total_length"],
                                                    output_file=pure_noise_output_path,
                                                    device="gpu" if torch.cuda.is_available() else "cpu",
                                                    ncpu=n_cpus if not torch.cuda.is_available() else 1)
                    

                else:
                    print_info("[WARNING] 'pure_noise' is configured, but no audio files were found in 'background_paths'.")
        

        else:
            logging.warning("Nanowakeword features already exist, skipping augmentation and feature generation. Verify existing files.")


    # Create nanowakeword model
    should_train = config.get("train_model", False)
    if args.train_model is True or should_train:

        # 1. Verify Feature Files and Load Configuration 
        # This block ensures all necessary .npy files exist before proceeding.
        
        required_files = {
            "Positive Features": os.path.join(feature_save_dir, "positive_features_train.npy"),
            "Negative Features": os.path.join(feature_save_dir, "negative_features_train.npy"),
            "Pure Noise Features": os.path.join(feature_save_dir, "pure_noise_features.npy")
        }

        missing_files = [desc for desc, path in required_files.items() if not os.path.exists(path)]

        if missing_files:
            print_info("\n[ERROR] Cannot start training. Required feature files are missing:")
            for desc in missing_files:
                print_info(f"- {desc}")
            print_info("\nPlease run with the '--transform_clips' flag or `transform_clips: true` to generate these files.")
            sys.exit(1)

        try:
            # Safely load the input shape from the primary feature file
            input_shape = np.load(required_files["Positive Features"]).shape[1:]
            seconds_per_example = 1280 * input_shape[0] / 16000
        except Exception as e:
            print_info(f"\n[ERROR] Failed to read 'positive_features_train.npy'. The file may be corrupted: {e}")
            sys.exit(1)
        
        # If we reach here, all files are present and valid.

        # 2. Setup Data Transformation and Batch Generation 

        # Data transform function to handle variable clip lengths
        def f(x, n=input_shape[0]):
            if x.shape[1] != n:
                x = np.vstack(x)
                return np.array([x[i:i+n, :] for i in range(0, x.shape[0]-n, n)])
            return x

        
        # Get batch composition settings
        batch_comp_config = config.get('batch_composition', {})
        total_batch_size = batch_comp_config.get('batch_size', 128)
        
        source_dist = batch_comp_config.get('source_distribution', 
            {'positive': 30, 'negative_speech': 40, 'pure_noise': 30})
        
        batch_n_per_class = {
            'positive': int(round(total_batch_size * (source_dist.get('positive', 0) / 100))),
            
            'adversarial_negative': int(round(total_batch_size * (source_dist.get('negative_speech', 0) / 100))),
            
            'pure_noise': int(round(total_batch_size * (source_dist.get('pure_noise', 0) / 100)))
        }

        # Define data sources 
        data_sources = {
            'positive': required_files["Positive Features"],
            'adversarial_negative': required_files["Negative Features"],
            'pure_noise': required_files["Pure Noise Features"]
        }
        

        # Filter out sources that are not requested in the batch composition
        final_data_files = {
            name: path for name, path in data_sources.items() 
            if batch_n_per_class.get(name) and batch_n_per_class[name] > 0
        }

        final_label_transforms = {
            key: (lambda x, k=key: [1] * len(x)) if key == 'positive' else (lambda x, k=key: [0] * len(x))
            for key in final_data_files
        }
        data_transforms = {key: f for key in final_data_files}

        # Create the memory-efficient batch generator
        batch_generator = mmap_batch_generator(
            data_files=final_data_files,
            n_per_class={k: v for k, v in batch_n_per_class.items() if k in final_data_files},
            data_transform_funcs=data_transforms,
            label_transform_funcs=final_label_transforms,
            triplet_mode=True 
        )

        class IterDataset(torch.utils.data.IterableDataset):
            def __init__(self, generator): self.generator = generator
            def __iter__(self): return self.generator

        # 3. Initialize Model, Optimizer, and DataLoader 
        print_info("Initializing Model and Training Components...")
        
        nww = Model(
            n_classes=1, 
            input_shape=input_shape,
            config=config,
            model_name=config.get("model_name", GNMV(config.get("model_type", "rnn"))),
            model_type=config.get("model_type", "rnn"),
            layer_dim=config["layer_size"],
            n_blocks=config["n_blocks"],
            dropout_prob=config.get("dropout_prob", 0.5),
            seconds_per_example=seconds_per_example
        )
      
        nww.setup_optimizer_and_scheduler(config=config)
        
        # The DataLoader wraps our iterable dataset
        X_train = torch.utils.data.DataLoader(
            IterDataset(batch_generator),
            batch_size=None, # Required for iterable datasets
            num_workers=0    # Recommended for this type of generator
        )

        # 4. Execute the Training Process 
        print_step_header(4, "Starting Training Process")
        
        model_type_str = config.get('model_type', "rnn").upper()
        print_info(f"Using model architecture: 🤍 {model_type_str}")

        best_model = nww.auto_train(
            X_train=X_train,
            steps=config.get("steps", 15000),
            debug_path=artifacts_dir,
            table_updater=dynamic_table,
            resume_from_dir=args.resume 
        )

        # 5. Post-Training Steps: Plotting and Exporting 
        nww.plot_history(artifacts_dir)
        
        nww.export_model(
            model=best_model, 
            model_name=config.get("model_name", GNMV(config.get("model_type", "rnn"))), 
            output_dir=model_save_dir
        )

if __name__ == '__main__':
    train()