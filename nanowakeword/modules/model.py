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
import copy
import torch
import random
import logging
import warnings
import torchinfo
import matplotlib
import numpy as np
import collections
from torch import nn
from nanowakeword.utils.logger import print_info

from .architectures import (
    CNNModel, LSTMModel, Net, GRUModel, RNNModel, TransformerModel, 
    CRNNModel, TCNModel, QuartzNetModel, ConformerModel, EBranchformerModel, BcResNetModel
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

        elif model_type == "bcresnet":
            self.model = BcResNetModel(
                input_shape=input_shape,
                embedding_dim=embedding_dim,
                dropout_prob=dropout_prob,
                activation_fn=self.activation_fn
            )

        else:
            raise ValueError(f"Unsupported model_type: '{model_type}'.")

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            self.activation_fn, 
            nn.Dropout(dropout_prob),
            nn.Linear(embedding_dim // 2, n_classes)
        )

        # Define logging dict (in-memory)
        self.history = collections.defaultdict(list)


    def plot_history(self, output_dir):
        """
        Single-figure, single-axes plot with a twin y-axis:
        - Left  axis (Loss):  Train Loss raw, Train Loss EMA, Val Loss
        - Right axis (Rate):  Train Recall EMA, Val Recall, Val FPR
        All lines share the same x-axis and plot box.
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt

        print_info("Generating training performance graph...")

        graph_output_dir = os.path.join(output_dir, "graphs")
        os.makedirs(graph_output_dir, exist_ok=True)

        loss_history = np.array(self.history['loss'])
        # Assuming self.config is available
        alpha = self.config.get("ema_alpha", 0.01)

        # EMA of training loss
        ema_loss_history, ema_val = [], None
        for v in loss_history:
            ema_val = v if ema_val is None else alpha * v + (1 - alpha) * ema_val
            ema_loss_history.append(ema_val)

        # EMA of training recall
        train_recall_steps = self.history.get('train_recall_steps', [])
        train_recall_vals  = self.history.get('train_recall', [])
        ema_train_recall, ema_r = [], None
        for r in train_recall_vals:
            ema_r = r if ema_r is None else 0.05 * r + 0.95 * ema_r
            ema_train_recall.append(ema_r)

        has_val_loss     = bool(self.history.get('val_loss'))
        has_val_recall   = bool(self.history.get('val_recall'))
        has_train_recall = bool(train_recall_vals)

        fig, ax_loss = plt.subplots(figsize=(13, 6))
        ax_rate = ax_loss.twinx()

        lines = []

        # Left axis Loss 
        l, = ax_loss.plot(
            loss_history,
            color="#7EB6E8", alpha=0.30, linewidth=1.0,
            label="Train Loss (Raw)"
        )
        lines.append(l)

        l, = ax_loss.plot(
            ema_loss_history,
            color="#1A5FA6", linewidth=2.2,
            label="Train Loss (EMA)"
        )
        lines.append(l)

        if has_val_loss:
            l, = ax_loss.plot(
                self.history['val_loss_steps'],
                self.history['val_loss'],
                color="#B85C00", linestyle="--",
                marker="o", markersize=4, linewidth=2.2,
                label="Val Loss"
            )
            lines.append(l)

        ax_loss.set_ylabel("Loss", fontsize=11, color="#1A5FA6")
        ax_loss.tick_params(axis='y', labelcolor="#1A5FA6")
        ax_loss.set_ylim(bottom=0)

        # Right axis Recall & FPR
        if has_train_recall:
            l, = ax_rate.plot(
                train_recall_steps, train_recall_vals,
                color="#82E0AA", alpha=0.40, linewidth=1.0,
                label="Train Recall (Raw)"
            )
            lines.append(l)

            l, = ax_rate.plot(
                train_recall_steps, ema_train_recall,
                color="#1A8A44", linewidth=2.2,
                label="Train Recall (EMA)"
            )
            lines.append(l)

        if has_val_recall:
            val_steps = self.history['val_recall_steps']
            l, = ax_rate.plot(
                val_steps, self.history['val_recall'],
                color="#C0392B", linestyle="--",
                marker="o", markersize=4, linewidth=2.2,
                label="Val Recall"
            )
            lines.append(l)

            l, = ax_rate.plot(
                val_steps, self.history['val_fpr'],
                color="#7D3C98", linestyle=":",
                marker="s", markersize=3, linewidth=2.0,
                label="Val FPR"
            )
            lines.append(l)

        ax_rate.set_ylabel("Recall / FPR", fontsize=11, color="#555555")
        ax_rate.tick_params(axis='y', labelcolor="#555555")
        ax_rate.set_ylim(-0.02, 1.05)

        # Shared decorations
        ax_loss.set_title("Training Performance", fontsize=14, weight="bold")
        ax_loss.set_xlabel("Training Steps", fontsize=11)
        ax_loss.grid(True, which="major", linestyle="--", alpha=0.25)
        ax_loss.grid(True, which="minor", linestyle=":", alpha=0.10)
        ax_loss.minorticks_on()

        labels = [l.get_label() for l in lines]
        
        ax_loss.legend(
            lines, labels,
            loc='best',        
            frameon=True,     
            framealpha=0.7,     
            facecolor='white',  
            fontsize=9
        )

        save_path = os.path.join(graph_output_dir, "training_performance_graph.png")
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

        print_info(f"Performance graph saved to: {save_path}")    

    def forward(self, x):
            """
            Takes input features and returns the final classification logits.
            Output shape: [B, 1] (batch_size, n_classes)
            """
            embeddings = self.model(x)
            logits = self.classifier(embeddings)  # Shape: [B, 1]
            return logits

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

