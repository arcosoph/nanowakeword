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
import matplotlib.pyplot as plt
from nanowakeword.utils.logger import print_info

from .architectures import (
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


    def plot_history(self, output_dir):
        """
        Creates a graph of training loss (raw + EMA) and validation loss.
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt

        print_info("Generating training performance graph...")

        graph_output_dir = os.path.join(output_dir, "graphs")
        os.makedirs(graph_output_dir, exist_ok=True)

        # ---------------------------
        # Training loss
        # ---------------------------
        loss_history = np.array(self.history['loss'])

        # EMA computation
        ema_loss_history = []
        ema_loss = None
        alpha = self.config.get("ema_alpha", 0.01)

        for loss_val in loss_history:
            if ema_loss is None:
                ema_loss = loss_val
            else:
                ema_loss = alpha * loss_val + (1 - alpha) * ema_loss
            ema_loss_history.append(ema_loss)

        # ---------------------------
        # Plot
        # ---------------------------
        plt.figure(figsize=(12, 6))

        plt.plot(
            loss_history,
            label="Train Loss (raw)",
            color="tab:blue",
            alpha=0.45,
            linewidth=1.2
        )

        # EMA training loss (main signal)
        plt.plot(
            ema_loss_history,
            label="Train Loss (EMA)",
            color="tab:blue",
            linewidth=2.5
        )

        # Validation loss (authoritative checkpoints)
        if (
            'val_loss_steps' in self.history
            and 'val_loss' in self.history
            and self.history['val_loss']
        ):
            plt.plot(
                self.history['val_loss_steps'],
                self.history['val_loss'],
                label="Validation Loss",
                color="tab:orange",
                linestyle="--",
                marker="o",
                markersize=4,
                linewidth=2
            )

        # ---------------------------
        # Styling (engineer-grade)
        # ---------------------------
        plt.title("Training & Validation Loss", fontsize=15, weight="bold")
        plt.xlabel("Training Steps", fontsize=12)
        plt.ylabel("Loss", fontsize=12)

        plt.legend(frameon=False)
        plt.grid(True, which="major", linestyle="--", alpha=0.25)
        plt.grid(True, which="minor", linestyle=":", alpha=0.1)
        plt.minorticks_on()

        plt.ylim(bottom=0)

        # ---------------------------
        # Save
        # ---------------------------
        save_path = os.path.join(graph_output_dir, "training_performance_graph.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
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

