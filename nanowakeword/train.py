# Copyright 2025 Arcosoph. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import uuid
import yaml
import copy
import scipy
import torch
import random
import logging
import warnings
import tempfile
import argparse
import torchinfo
import matplotlib
import collections
import numpy as np
import torchmetrics
import nanowakeword
matplotlib.use('Agg')
from tqdm import tqdm
from pathlib import Path
from torch import optim, nn
import matplotlib.pyplot as plt
from nanowakeword.utils.audio_processing import AudioFeatures
from nanowakeword.data_utils.preprocess import verify_and_process_directory
from nanowakeword.utils.audio_processing import compute_features_from_generator
from nanowakeword.data import augment_clips, mmap_batch_generator, generate_adversarial_texts
from nanowakeword.utils.logger import print_banner, print_step_header, print_info, print_key_value, print_final_report_header, print_table

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

def calculate_validation_duration_hours(fp_validation_data_path, input_shape):
    """
    Calculates the total duration of a validation set stored in a .npy file,
    based on the number of independent samples and the shape of each sample.

    Args:
        fp_validation_data_path (str): Path to the .npy file with validation features.
        input_shape (tuple): The shape of a single feature sample (e.g., (157, 96)).

    Returns:
        float: The total duration in hours. Defaults to 1.0 on error.
    """
    try:
        if not os.path.exists(fp_validation_data_path):
            logging.warning(f"Validation file not found: '{fp_validation_data_path}'. Defaulting to 1.0 hour.")
            return 1.0
            
        fp_validation_features = np.load(fp_validation_data_path)
        num_samples = len(fp_validation_features)

        if num_samples == 0:
            logging.info("Validation file is empty, returning 0.0 hours.")
            return 0.0

        time_steps = input_shape[0]
        # This calculation is specific to how NanoWakeWord features are generated
        seconds_per_example = (1280 * time_steps) / 16000 # 16000 is the audio sample rate

        total_seconds = num_samples * seconds_per_example
        total_hours = total_seconds / 3600
        
        print_info(f"Dynamically calculated validation set duration: {total_hours:.2f} hours.")
        return total_hours

    except Exception as e:
        print_info(f"Could not calculate validation duration due to an error: {e}. Defaulting to 1.0 hour.")
        return 1.0


# Base model class for an nanowakeword model
class Model(nn.Module):

    def __init__(self, n_classes=1, input_shape=(16, 96), model_type="dnn",
                layer_dim=128, n_blocks=1, seconds_per_example=None, dropout_prob=0.5):
        super().__init__()

        # Store inputs as attributes
        self.n_classes = n_classes
        self.input_shape = input_shape
        self.seconds_per_example = seconds_per_example
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Training progress tracking attributes 
        self.best_models = []
        self.best_model_scores = []
        self.best_val_fp = 1000
        self.best_val_accuracy = 0
        self.best_val_recall = 0
        self.best_train_recall = 0
        self.n_fp = 0
        self.val_fp = 0
        self.history = collections.defaultdict(list)
        

        if model_type == "cnn":
            class CNNModel(nn.Module):
                def __init__(self, input_shape, n_classes, dropout_prob):
                    super().__init__()
                    self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
                    self.relu1 = nn.ReLU()
                    self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
                    self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
                    self.relu2 = nn.ReLU()
                    self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
                    conv_output_size = self._get_conv_output(input_shape)
                    self.flatten = nn.Flatten()
                    self.fc1 = nn.Linear(conv_output_size, 128)
                    self.relu3 = nn.ReLU()
                    self.dropout = nn.Dropout(dropout_prob)
                    self.fc2 = nn.Linear(128, n_classes)
                    self.sigmoid = nn.Sigmoid()
                def _get_conv_output(self, shape):
                    with torch.no_grad():
                        input = torch.zeros(1, 1, *shape)
                        output = self.pool1(self.relu1(self.conv1(input)))
                        output = self.pool2(self.relu2(self.conv2(output)))
                        return int(np.prod(output.shape))
                def forward(self, x):
                    if x.dim() == 3: x = x.unsqueeze(1)
                    x = self.pool1(self.relu1(self.conv1(x)))
                    x = self.pool2(self.relu2(self.conv2(x)))
                    x = self.flatten(x)
                    x = self.relu3(self.fc1(x))
                    x = self.dropout(x)
                    x = self.sigmoid(self.fc2(x))
                    return x
            self.model = CNNModel(input_shape, n_classes, dropout_prob=dropout_prob)
            
        elif model_type == "lstm":
            class LSTMModel(nn.Module):
                def __init__(self, input_dim, hidden_dim, n_layers, n_classes, bidirectional, dropout_prob):
                    super().__init__()
                    self.n_layers = n_layers
                    self.hidden_dim = hidden_dim
                    self.bidirectional = bidirectional

                    self.lstm = nn.LSTM(
                        input_dim,
                        hidden_dim,
                        n_layers,
                        batch_first=True,
                        bidirectional=bidirectional,
                        dropout=dropout_prob if n_layers > 1 else 0
                    )
                    
                    linear_input_size = hidden_dim * 2 if bidirectional else hidden_dim
                    
                    self.dropout = nn.Dropout(dropout_prob)
                    self.fc = nn.Linear(linear_input_size, n_classes)
                    self.sigmoid = nn.Sigmoid()

                def forward(self, x):
                    lstm_out, _ = self.lstm(x)
                    last_time_step_output = lstm_out[:, -1, :]
                    out = self.dropout(last_time_step_output)
                    out = self.fc(out)
                    out = self.sigmoid(out)
                    return out
            self.model = LSTMModel(input_shape[1], layer_dim, n_blocks, n_classes=n_classes, bidirectional=True, dropout_prob=dropout_prob)
            

        elif model_type == "dnn":
            class FCNBlock(nn.Module):
                def __init__(self, layer_dim):
                    super().__init__()
                    self.fcn_layer = nn.Linear(layer_dim, layer_dim)
                    self.relu = nn.ReLU()
                    self.layer_norm = nn.LayerNorm(layer_dim)
                def forward(self, x):
                    return self.relu(self.layer_norm(self.fcn_layer(x)))

            class Net(nn.Module):
                def __init__(self, input_shape, layer_dim, n_blocks, n_classes, dropout_prob):
                    super().__init__()
                    self.flatten = nn.Flatten()
                    self.layer1 = nn.Linear(input_shape[0]*input_shape[1], layer_dim)
                    self.relu1 = nn.ReLU()
                    self.layernorm1 = nn.LayerNorm(layer_dim)
                    self.dropout = nn.Dropout(dropout_prob)
                    self.blocks = nn.ModuleList([FCNBlock(layer_dim) for i in range(n_blocks)])
                    self.last_layer = nn.Linear(layer_dim, n_classes)
                    self.last_act = nn.Sigmoid() if n_classes == 1 else nn.ReLU()
                def forward(self, x):
                    x = self.relu1(self.layernorm1(self.layer1(self.flatten(x))))
                    x = self.dropout(x)
                    for block in self.blocks:
                        x = block(x)
                    x = self.last_act(self.last_layer(x))
                    return x
            self.model = Net(input_shape, layer_dim, n_blocks=n_blocks, n_classes=n_classes, dropout_prob=dropout_prob)
            

        elif model_type == "gru":
            class GRUModel(nn.Module):
                def __init__(self, input_dim, hidden_dim, n_layers, n_classes, bidirectional, dropout_prob):
                    super().__init__()
                    self.n_layers = n_layers
                    self.hidden_dim = hidden_dim
                    self.bidirectional = bidirectional

                    self.gru = nn.GRU(
                        input_dim,
                        hidden_dim,
                        n_layers,
                        batch_first=True,
                        bidirectional=bidirectional,
                        dropout=dropout_prob if n_layers > 1 else 0
                    )
                    
                    linear_input_size = hidden_dim * 2 if bidirectional else hidden_dim
                    self.dropout = nn.Dropout(dropout_prob)
                    self.fc = nn.Linear(linear_input_size, n_classes)
                    self.sigmoid = nn.Sigmoid()

                def forward(self, x):
                    gru_out, _ = self.gru(x)
                    last_time_step_output = gru_out[:, -1, :]
                    out = self.dropout(last_time_step_output)
                    out = self.fc(out)
                    out = self.sigmoid(out)
                    return out
            self.model = GRUModel(input_shape[1], layer_dim, n_blocks, n_classes=n_classes, bidirectional=True, dropout_prob=dropout_prob)
            

        elif model_type == "rnn":
            class Net(nn.Module):
                def __init__(self, input_shape, n_classes, dropout_prob):
                    super().__init__()
                    # The number of layers (num_layers=2) is hardcoded here, so the dropout logic is safe
                    self.layer1 = nn.LSTM(input_shape[-1], 64, num_layers=2, bidirectional=True,
                                          batch_first=True, dropout=dropout_prob if 2 > 1 else 0)
                    self.dropout = nn.Dropout(dropout_prob)
                    self.layer2 = nn.Linear(64*2, n_classes)
                    self.layer3 = nn.Sigmoid() if n_classes == 1 else nn.ReLU()

                def forward(self, x):
                    out, h = self.layer1(x)
                    last_output = self.dropout(out[:, -1])
                    return self.layer3(self.layer2(last_output))
            self.model = Net(input_shape, n_classes, dropout_prob=dropout_prob)


        # ---------------------------------------------------------


        # Define metrics
        self.fp = lambda pred, y: (y-pred <= -0.5).sum()
        self.recall = torchmetrics.Recall(task='binary')
        self.accuracy = torchmetrics.Accuracy(task='binary')

        # Define logging dict (in-memory)
        self.history = collections.defaultdict(list)

        # Define optimizer and loss
        self.loss = torch.nn.functional.binary_cross_entropy
        # self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=1e-5) # <-- weight_decay 



    def setup_optimizer_and_scheduler(self, config):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config['learning_rate_max'], weight_decay=1e-2)
        self.scheduler = torch.optim.lr_scheduler.CyclicLR(
            self.optimizer,
            base_lr=config['learning_rate_base'],
            max_lr=config['learning_rate_max'],
            step_size_up=config['clr_step_size_up'],
            step_size_down=config["clr_step_size_down"],
            mode='triangular2',
            cycle_momentum=False
        )


    def plot_history(self, output_dir):
        print("\nGenerating training performance graph...")
        graph_output_dir = os.path.join(output_dir, "graphs")
        os.makedirs(graph_output_dir, exist_ok=True)
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.history['loss'], label='Training Loss')
        plt.title('Training Loss'); plt.xlabel('Steps'); plt.ylabel('Loss')
        plt.legend(); plt.grid(True)
        plt.subplot(1, 2, 2)
        plt.plot(self.history['val_recall'], label='Validation Recall')
        plt.title('Validation Recall'); plt.xlabel('Validation Steps'); plt.ylabel('Recall')
        plt.legend(); plt.grid(True)
        save_path = os.path.join(graph_output_dir, "training_performance_graph.png")
        plt.tight_layout(); plt.savefig(save_path); plt.close()
        print(f"Performance graph saved to: {save_path}")
    
   
    def save_model(self, output_path):
        """
        Saves the weights of a trained Pytorch model
        """
        if self.n_classes == 1:
            torch.save(self.model, output_path)

    def export_to_onnx(self, output_path, class_mapping=""):
        obj = self
        # Make simple model for export based on model structure
        if self.n_classes == 1:
            # Save ONNX model
            torch.onnx.export(self.model.to("cpu"), torch.rand(self.input_shape)[None, ], output_path,
                              output_names=[class_mapping])

        elif self.n_classes >= 1:
            class M(nn.Module):
                def __init__(self):
                    super().__init__()

                    # Define model
                    self.model = obj.model.to("cpu")

                def forward(self, x):
                    return torch.nn.functional.softmax(self.model(x), dim=1)

            # Save ONNX model
            torch.onnx.export(M(), torch.rand(self.input_shape)[None, ], output_path,
                              output_names=[class_mapping])

    def lr_warmup_cosine_decay(self,
                               global_step,
                               warmup_steps=0,
                               hold=0,
                               total_steps=0,
                               start_lr=0.0,
                               target_lr=1e-3
                               ):
        # Cosine decay
        learning_rate = 0.5 * target_lr * (1 + np.cos(np.pi * (global_step - warmup_steps - hold)
                                           / float(total_steps - warmup_steps - hold)))

        # Target LR * progress of warmup (=1 at the final warmup step)
        warmup_lr = target_lr * (global_step / warmup_steps)

        # Choose between `warmup_lr`, `target_lr` and `learning_rate` based on whether
        # `global_step < warmup_steps` and we're still holding.
        # i.e. warm up if we're still warming up and use cosine decayed lr otherwise
        if hold > 0:
            learning_rate = np.where(global_step > warmup_steps + hold,
                                     learning_rate, target_lr)

        learning_rate = np.where(global_step < warmup_steps, warmup_lr, learning_rate)
        return learning_rate

    def forward(self, x):
        return self.model(x)

    def summary(self):
        return torchinfo.summary(self.model, input_size=(1,) + self.input_shape, device='cpu')

    def average_models(self, models=None):
        """Averages the weights of the provided models together to make a new model"""

        if models is None:
            models = self.best_models

        # Clone a model from the list as the base for the averaged model
        averaged_model = copy.deepcopy(models[0])
        averaged_model_dict = averaged_model.state_dict()

        # Initialize a running total of the weights
        for key in averaged_model_dict:
            averaged_model_dict[key] *= 0  # set to 0

        for model in models:
            model_dict = model.state_dict()
            for key, value in model_dict.items():
                averaged_model_dict[key] += value

        for key in averaged_model_dict:
            averaged_model_dict[key] /= len(models)

        # Load the averaged weights into the model
        averaged_model.load_state_dict(averaged_model_dict)

        return averaged_model

    def _select_best_model(self, false_positive_validate_data, val_set_hrs=11.3, max_fp_per_hour=0.5, min_recall=0.20):
        """
        Select the top model based on the false positive rate on the validation data

        Args:
            false_positive_validate_data (torch.DataLoader): A dataloader with validation data
            n (int): The number of models to select

        Returns:
            list: A list of the top n models
        """
        # Get false positive rates for each model
        false_positive_rates = [0]*len(self.best_models)
        for batch in false_positive_validate_data:
            x_val, y_val = batch[0].to(self.device), batch[1].to(self.device)
            for mdl_ndx, model in tqdm(enumerate(self.best_models), total=len(self.best_models),
                                       desc="Find best checkpoints by false positive rate"):
                with torch.no_grad():
                    val_ps = model(x_val)
                    false_positive_rates[mdl_ndx] = false_positive_rates[mdl_ndx] + self.fp(val_ps, y_val[..., None]).detach().cpu().numpy()
        false_positive_rates = [fp/val_set_hrs for fp in false_positive_rates]

        candidate_model_ndx = [ndx for ndx, fp in enumerate(false_positive_rates) if fp <= max_fp_per_hour]
        candidate_model_recall = [self.best_model_scores[ndx]["val_recall"] for ndx in candidate_model_ndx]
        if max(candidate_model_recall) <= min_recall:
            logging.warning(f"No models with recall >= {min_recall} found!")
            return None
        else:
            best_model = self.best_models[candidate_model_ndx[np.argmax(candidate_model_recall)]]
            best_model_training_step = self.best_model_scores[candidate_model_ndx[np.argmax(candidate_model_recall)]]["training_step_ndx"]
            logging.info(f"Best model from training step {best_model_training_step} out of {len(candidate_model_ndx)}"
                         f"models has recall of {np.max(candidate_model_recall)} and false positive rate of"
                         f" {false_positive_rates[candidate_model_ndx[np.argmax(candidate_model_recall)]]}")

        return best_model


    def auto_train(self, X_train, X_val, false_positive_val_data, steps, max_negative_weight, target_fp_per_hour, val_set_hrs):
        """
        A modern, single-sequence training process that utilizes the globally configured
        optimizer and Cyclical Learning Rate scheduler. It trains for a specified number
        of steps and then merges the best saved checkpoints into a final robust model.
        """

        print_info("Starting modern training sequence with Cyclical Learning Rates...")
        
        # val_set_hrs = 11.3 # 
        
        val_steps = np.unique(np.linspace(start=steps//50, stop=steps, num=50, dtype=int))

      
        self.train_model(
            X=X_train,
            X_val=X_val,
            false_positive_val_data=false_positive_val_data,
            max_steps=steps,
            negative_weight_schedule=[max_negative_weight], 
            val_steps=val_steps,
            val_set_hrs=val_set_hrs
        )

        print_info("Training finished. Merging best checkpoints to create final model...")
        
        
        if not self.best_models:
            print_info("No best models were saved based on performance criteria. Returning the final model state.")
           
            combined_model = self.model
        else:
         
            try:
                # Check if there are enough data points for np.percentile
                if len(self.history["val_accuracy"]) > 1 and len(self.history["val_recall"]) > 1 and len(self.history["val_fp_per_hr"]) > 1:
                    accuracy_percentile = np.percentile(self.history["val_accuracy"], 90)
                    recall_percentile = np.percentile(self.history["val_recall"], 90)
                    fp_percentile = np.percentile(self.history["val_fp_per_hr"], 10)
                else:
                    
                    accuracy_percentile, recall_percentile, fp_percentile = 0, 0, float('inf')

                models_to_merge = []
                for model, score in zip(self.best_models, self.best_model_scores):
                    if (score["val_accuracy"] >= accuracy_percentile and
                            score["val_recall"] >= recall_percentile and
                            score["val_fp_per_hr"] <= fp_percentile):
                        models_to_merge.append(model)
                
                if models_to_merge:
                    print_info(f"Found {len(models_to_merge)} models meeting the 90th percentile criteria. Merging them...")
                    combined_model = self.average_models(models=models_to_merge)
                else:
                    print_info("No models met the strict percentile criteria. Averaging all saved best models instead...")
                    combined_model = self.average_models(models=self.best_models)
            except Exception as e:
                print_info(f"An error occurred during model merging: {e}. Averaging all saved models as a fallback.")
                combined_model = self.average_models(models=self.best_models)

        
        print_info("Calculating final performance metrics for the merged model...")
        with torch.no_grad():
            # Metric calculation by dividing validation data into batches (to save memory)
            all_val_preds, all_val_labels = [], []
            for batch in X_val:
                x, y = batch[0].to(self.device), batch[1].to(self.device)
                val_ps = combined_model(x)
                all_val_preds.append(val_ps)
                all_val_labels.append(y)
            
            final_val_preds = torch.cat(all_val_preds)
            final_val_labels = torch.cat(all_val_labels)

            combined_model_recall = self.recall(final_val_preds, final_val_labels[..., None]).detach().cpu().numpy()
            combined_model_accuracy = self.accuracy(final_val_preds, final_val_labels[..., None].to(torch.int64)).detach().cpu().numpy()

            combined_model_fp = 0
            for batch in false_positive_val_data:
                x_val, y_val = batch[0].to(self.device), batch[1].to(self.device)
                val_ps = combined_model(x_val)
                combined_model_fp += self.fp(val_ps, y_val[..., None])

            combined_model_fp_per_hr = (combined_model_fp / val_set_hrs).detach().cpu().numpy()

    
        final_results = {
            "Final Accuracy": f"{combined_model_accuracy:.4f}",
            "Final Recall": f"{combined_model_recall:.4f}",
            "False Positives per Hour": f"{combined_model_fp_per_hr:.4f}"
        }
        print_final_report_header()
        for key, value in final_results.items():
            print_key_value(key, value)
  
        return combined_model
    # ======================================================================


    def predict_on_features(self, features, model=None):
        """
        Predict on Tensors of NanoWakeWord features corresponding to single audio clips

        Args:
            features (torch.Tensor): A Tensor of NanoWakeWord features with shape (batch, features)
            model (torch.nn.Module): A Pytorch model to use for prediction (default None, which will use self.model)

        Returns:
            torch.Tensor: An array of predictions of shape (batch, prediction), where 0 is negative and 1 is positive
        """
        if len(features) < 3:
            features = features[None, ]

        features = features.to(self.device)
        predictions = []
        for x in tqdm(features, desc="Predicting on clips"):
            x = x[None, ]
            batch = []
            for i in range(0, x.shape[1]-16, 1):  # step size of 1 (80 ms)
                batch.append(x[:, i:i+16, :])
            batch = torch.vstack(batch)
            if model is None:
                preds = self.model(batch)
            else:
                preds = model(batch)
            predictions.append(preds.detach().cpu().numpy()[None, ])

        return np.vstack(predictions)

    def predict_on_clips(self, clips, model=None):
        """
        Predict on Tensors of 16-bit 16 khz audio data

        Args:
            clips (np.ndarray): A Numpy array of audio clips with shape (batch, samples)
            model (torch.nn.Module): A Pytorch model to use for prediction (default None, which will use self.model)

        Returns:
            np.ndarray: An array of predictions of shape (batch, prediction), where 0 is negative and 1 is positive
        """

        # Get features from clips
        F = AudioFeatures(device='cpu', ncpu=4)
        features = F.embed_clips(clips, batch_size=16)

        # Predict on features
        preds = self.predict_on_features(torch.from_numpy(features), model=model)

        return preds

    def export_model(self, model, model_name, output_dir):
        """Saves the trained Nanowakeword model to both onnx and tflite formats"""

        if self.n_classes != 1:
            raise ValueError("Exporting models to both onnx and tflite with more than one class is currently not supported! "
                             "Use the `export_to_onnx` function instead.")

        # Save ONNX model
        print_info(f"Saving ONNX mode as '{os.path.join(output_dir, model_name + '.onnx')}'")
        model_to_save = copy.deepcopy(model)
        torch.onnx.export(model_to_save.to("cpu"), torch.rand(self.input_shape)[None, ],
                          os.path.join(output_dir, model_name + ".onnx"), opset_version=13)

        return None

    def train_model(self, X, max_steps, X_val=None,
                    false_positive_val_data=None, positive_test_clips=None,
                    negative_weight_schedule=[1],
                    val_steps=[250], val_set_hrs=1):
        # Move models and main class to target device
        self.to(self.device)
        self.model.to(self.device)

        # Train model
        accumulation_steps = 1
        accumulated_samples = 0
        accumulated_predictions = torch.Tensor([]).to(self.device)
        accumulated_labels = torch.Tensor([]).to(self.device)
        for step_ndx, data in tqdm(enumerate(X, 0), total=max_steps, desc="Training"):
            # get the inputs; data is a list of [inputs, labels]
            x, y = data[0].to(self.device), data[1].to(self.device)
            y_ = y[..., None].to(torch.float32)

            # # Update learning rates
            # for g in self.optimizer.param_groups:
            #     g['lr'] = self.lr_warmup_cosine_decay(step_ndx, warmup_steps=warmup_steps, hold=hold_steps,
            #                                           total_steps=max_steps, target_lr=lr)

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # Get predictions for batch
            predictions = self.model(x)

            # Construct batch with only samples that have high loss
            neg_high_loss = predictions[(y == 0) & (predictions.squeeze() >= 0.001)]  # thresholds were chosen arbitrarily but work well
            pos_high_loss = predictions[(y == 1) & (predictions.squeeze() < 0.999)]
            y = torch.cat((y[(y == 0) & (predictions.squeeze() >= 0.001)], y[(y == 1) & (predictions.squeeze() < 0.999)]))
            y_ = y[..., None].to(torch.float32)
            predictions = torch.cat((neg_high_loss, pos_high_loss))

            # Set weights for batch
            if len(negative_weight_schedule) == 1:
                w = torch.ones(y.shape[0])*negative_weight_schedule[0]
                pos_ndcs = y == 1
                w[pos_ndcs] = 1
                w = w[..., None]
            else:
                if self.n_classes == 1:
                    w = torch.ones(y.shape[0])*negative_weight_schedule[step_ndx]
                    pos_ndcs = y == 1
                    w[pos_ndcs] = 1
                    w = w[..., None]

            if predictions.shape[0] != 0:
                # Do backpropagation, with gradient accumulation if the batch-size after selecting high loss examples is too small
                loss = self.loss(predictions, y_ if self.n_classes == 1 else y, w.to(self.device))
                loss = loss/accumulation_steps
                accumulated_samples += predictions.shape[0]

                if predictions.shape[0] >= 128:
                    accumulated_predictions = predictions
                    accumulated_labels = y_
                if accumulated_samples < 128:
                    accumulation_steps += 1
                    accumulated_predictions = torch.cat((accumulated_predictions, predictions))
                    accumulated_labels = torch.cat((accumulated_labels, y_))
                else:
                    loss.backward()
                    self.optimizer.step()
                    # Update the Cyclical Learning Rate scheduler at each step
                    if hasattr(self, 'scheduler'):
                        self.scheduler.step()
  
                    accumulation_steps = 1
                    accumulated_samples = 0


                    self.history["loss"].append(loss.detach().cpu().numpy())

                    # Compute training metrics and log them
                    fp = self.fp(accumulated_predictions, accumulated_labels if self.n_classes == 1 else y)
                    self.n_fp += fp
                    self.history["recall"].append(self.recall(accumulated_predictions, accumulated_labels).detach().cpu().numpy())

                    accumulated_predictions = torch.Tensor([]).to(self.device)
                    accumulated_labels = torch.Tensor([]).to(self.device)

            # Run validation and log validation metrics
            if step_ndx in val_steps and step_ndx > 1 and false_positive_val_data is not None:
                # Get false positives per hour with false positive data
                val_fp = 0
                for val_step_ndx, data in enumerate(false_positive_val_data):
                    with torch.no_grad():
                        x_val, y_val = data[0].to(self.device), data[1].to(self.device)
                        val_predictions = self.model(x_val)
                        val_fp += self.fp(val_predictions, y_val[..., None])
                val_fp_per_hr = (val_fp/val_set_hrs).detach().cpu().numpy()
                self.history["val_fp_per_hr"].append(val_fp_per_hr)

            # Get recall on test clips
            if step_ndx in val_steps and step_ndx > 1 and positive_test_clips is not None:
                tp = 0
                fn = 0
                for val_step_ndx, data in enumerate(positive_test_clips):
                    with torch.no_grad():
                        x_val = data[0].to(self.device)
                        batch = []
                        for i in range(0, x_val.shape[1]-16, 1):
                            batch.append(x_val[:, i:i+16, :])
                        batch = torch.vstack(batch)
                        preds = self.model(batch)
                        if any(preds >= 0.5):
                            tp += 1
                        else:
                            fn += 1
                self.history["positive_test_clips_recall"].append(tp/(tp + fn))


            # Run validation and log validation metrics
            if step_ndx in val_steps and step_ndx > 1:
                if X_val is not None:
                    all_val_predictions = []
                    all_val_labels = []
                    for x_val_batch, y_val_batch in X_val:
                        with torch.no_grad():
                            x_val_batch, y_val_batch = x_val_batch.to(self.device), y_val_batch.to(self.device)
                            val_predictions_batch = self.model(x_val_batch)
                            all_val_predictions.append(val_predictions_batch)
                            all_val_labels.append(y_val_batch)

                    val_predictions = torch.cat(all_val_predictions)
                    y_val = torch.cat(all_val_labels)
                    
                    val_recall = self.recall(val_predictions, y_val[..., None]).detach().cpu().numpy()
                    val_acc = self.accuracy(val_predictions, y_val[..., None].to(torch.int64))
                    val_fp_count = self.fp(val_predictions, y_val[..., None])
                    
                    self.history["val_accuracy"].append(val_acc.detach().cpu().numpy())
                    self.history["val_recall"].append(val_recall)
                    self.history["val_n_fp"].append(val_fp_count.detach().cpu().numpy())
                
                if false_positive_val_data is not None:
                    total_val_fp = 0
                    for x_fp_batch, y_fp_batch in false_positive_val_data:
                        with torch.no_grad():
                            x_fp_batch, y_fp_batch = x_fp_batch.to(self.device), y_fp_batch.to(self.device)
                            fp_predictions_batch = self.model(x_fp_batch)
                            total_val_fp += self.fp(fp_predictions_batch, y_fp_batch[..., None])
                    
                    val_fp_per_hr = (total_val_fp / val_set_hrs).detach().cpu().numpy()
                    self.history["val_fp_per_hr"].append(val_fp_per_hr)

                # Save models with a validation score above/below the targets
                if self.history["val_n_fp"][-1] <= np.percentile(self.history["val_n_fp"], 50) and \
                   self.history["val_recall"][-1] >= np.percentile(self.history["val_recall"], 5):
                    
                    self.best_models.append(copy.deepcopy(self.model))
                    self.best_model_scores.append({
                        "training_step_ndx": step_ndx, 
                        "val_n_fp": self.history["val_n_fp"][-1],
                        "val_recall": self.history["val_recall"][-1],
                        "val_accuracy": self.history["val_accuracy"][-1],
                        "val_fp_per_hr": self.history.get("val_fp_per_hr", [0])[-1]
                    })
                    self.best_val_recall = self.history["val_recall"][-1]
                    self.best_val_accuracy = self.history["val_accuracy"][-1]
                # ======================================================================

            if step_ndx == max_steps - 1:
                break

# # Separate function to convert onnx models to tflite format
def convert_onnx_to_tflite(onnx_model_path, output_path):
    """The latest Python <3.12 version does not support NanoWakeWord v1.1.0 to convert the ONNX version of 
    the NanoWakeWord model to the TensorFlow Tflight format."""
    print("The latest Python <3.12 version does not support NanoWakeWord v1.1.0 to convert the ONNX version of the NanoWakeWord model to the TensorFlow Tflight format.")

    import onnx
    from onnx_tf.backend import prepare
    import tensorflow as tf

    # Convert to tflite from onnx model
    onnx_model = onnx.load(onnx_model_path)
    tf_rep = prepare(onnx_model, device="CPU")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tf_rep.export_graph(os.path.join(tmp_dir, "tf_model"))
        converter = tf.lite.TFLiteConverter.from_saved_model(os.path.join(tmp_dir, "tf_model"))
        tflite_model = converter.convert()

        print_info(f"Saving tflite model to '{output_path}'")
        with open(output_path, 'wb') as f:
            f.write(tflite_model)

    return None



def main():
    # Get training config file
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--training_config",
        help="The path to the training config file (required)",
        type=str,
        required=True
    )
    parser.add_argument(
        "--generate_clips",
        help="Execute the synthetic data generation process",
        action="store_true"
    )

    parser.add_argument(
        "--overwrite",
        help="Overwrite existing NanoWakeWord features when the --augment_clips flag is used",
        action="store_true",
        default="False",
        required=False
    )
    parser.add_argument(
        "--train_model",
        help="Execute the model training process",
        action="store_true",
        default="False",
        required=False
    )


 
    parser.add_argument(
        "--auto-config",
        action="store_true",
        help="Automatically analyze the dataset and generate the best configuration before training."
    )

    parser.add_argument(
        "--augment_clips",
        help="Execute the synthetic data augmentation process",
        action="store_true",
        default="False",
        required=False
    )
    args = parser.parse_args()

#=====
    print_banner()

    config = yaml.load(open(args.training_config, 'r', encoding='utf-8').read(), yaml.Loader)
#=====

  

    print_step_header( 1,"Verifying and Preprocessing Data Directories")
    
    data_paths_to_process = [
        config.get("wakeword_data_path"),
        config.get("background_data_path")
    ]

    data_paths_to_process.extend(config.get("background_paths", []))
    data_paths_to_process.extend(config.get("rir_paths", []))

    unique_paths = set(p for p in data_paths_to_process if p)
    
    for path in unique_paths:
        verify_and_process_directory(path)
        
    print_info("Data verification and preprocessing complete.\n")
   

    if args.auto_config:
        print_step_header( 2,"Activating Intelligent Configuration Engine")
        

        from nanowakeword.analyzer import DatasetAnalyzer
        from nanowakeword.config_generator import ConfigGenerator
        
        try:
            analyzer = DatasetAnalyzer(
                positive_path=config["wakeword_data_path"],
                negative_path=config["background_data_path"],
                noise_path=config["background_paths"][0],
                rir_path=config["rir_paths"][0]
            )
            dataset_stats = analyzer.analyze()
           
            print_table(dataset_stats, "Dataset Statistics")

            generator = ConfigGenerator(dataset_stats)
            
            intelligent_config = generator.generate(data_generation_is_planned=args.generate_clips)
            
            print_table( intelligent_config, "Generated Intelligent Config & Data Plan")
            
            config.update(intelligent_config)
            print_info("Base configuration updated with intelligent settings.")

        except KeyError as e:
            print(f"ERROR: Missing essential path in config file for auto-config: {e}")
            exit()



    from nanowakeword.generate_samples import generate_samples

    # Define output locations
    config["output_dir"] = os.path.abspath(config["output_dir"])
    if not os.path.exists(config["output_dir"]):
        os.mkdir(config["output_dir"])
    if not os.path.exists(os.path.join(config["output_dir"], config["model_name"])):
        os.mkdir(os.path.join(config["output_dir"], config["model_name"]))

# Ensure directories exist (modified)
    positive_train_output_dir = os.path.join(config["output_dir"], config["model_name"], "positive_train")
    positive_test_output_dir = os.path.join(config["output_dir"], config["model_name"], "positive_test")
    negative_train_output_dir = os.path.join(config["output_dir"], config["model_name"], "negative_train")
    negative_test_output_dir = os.path.join(config["output_dir"], config["model_name"], "negative_test")
    feature_save_dir = os.path.join(config["output_dir"], config["model_name"])
    # Ensure directories exist (modified)
    os.makedirs(os.path.join(feature_save_dir, "positive_train"), exist_ok=True)
    os.makedirs(os.path.join(feature_save_dir, "positive_test"), exist_ok=True)
    os.makedirs(os.path.join(feature_save_dir, "negative_train"), exist_ok=True)
    os.makedirs(os.path.join(feature_save_dir, "negative_test"), exist_ok=True)

    # Get paths for impulse response and background audio files
    rir_paths = [i.path for j in config["rir_paths"] for i in os.scandir(j)]
    background_paths = []
    if len(config["background_paths_duplication_rate"]) != len(config["background_paths"]):
        config["background_paths_duplication_rate"] = [1]*len(config["background_paths"])
    for background_path, duplication_rate in zip(config["background_paths"], config["background_paths_duplication_rate"]):
        background_paths.extend([i.path for i in os.scandir(background_path)]*duplication_rate)

    
    if args.generate_clips:
       
        print_step_header(2,"Activating Synthetic Data Generation")
       
        n_pos_train, n_neg_train, n_pos_test, n_neg_test = 0, 0, 0, 0
        positive_output_dir_train = positive_train_output_dir
        negative_output_dir_train = negative_train_output_dir

        if 'data_generation_plan' in config and args.auto_config:
            
            print_info("Using intelligent data harmonization plan for TRAINING samples.")
            plan = config['data_generation_plan']
            pos_hours = plan.get('generate_positive_hours', 0.0)
            neg_hours = plan.get('generate_negative_hours', 0.0)
            n_pos_train = int((pos_hours * 3600) / 2)
            n_neg_train = int((neg_hours * 3600) / 2)
            
            
            positive_output_dir_train = config["wakeword_data_path"]
            negative_output_dir_train = config["background_data_path"]
        else:
            
            print_info("Using 'n_samples' from config file for TRAINING samples.")
            n_pos_train = config.get("n_samples", 100)
            n_neg_train = config.get("n_samples", 100)

      
        n_pos_test = config.get("n_samples_val", 20)
        n_neg_test = config.get("n_samples_val", 20)
 
        
        if n_pos_train > 0:
            print_info(f"Generating {n_pos_train} positive clips for training to '{positive_output_dir_train}'...")
            os.makedirs(positive_output_dir_train, exist_ok=True)
            generate_samples(
                text=config["target_phrase"], max_samples=n_pos_train,
                output_dir=positive_output_dir_train, batch_size=config.get("tts_batch_size", 256)
            )
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

       
        if n_neg_train > 0:
            print_info(f"Generating {n_neg_train} adversarial negative clips for training to '{negative_output_dir_train}'...")
            os.makedirs(negative_output_dir_train, exist_ok=True)
            adversarial_texts = config.get("custom_negative_phrases", [])
            if config.get("target_phrase"):
                adversarial_texts.extend(generate_adversarial_texts(
                    input_text=config["target_phrase"][0], N=n_neg_train
                ))
            if adversarial_texts:
                generate_samples(
                    text=adversarial_texts, max_samples=n_neg_train,
                    output_dir=negative_output_dir_train, batch_size=config.get("tts_batch_size", 256) // 4
                )
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

       
        if n_pos_test > 0:
            print_info(f"Generating {n_pos_test} positive clips for testing to '{positive_test_output_dir}'...")
            os.makedirs(positive_test_output_dir, exist_ok=True)
            generate_samples(
                text=config["target_phrase"], max_samples=n_pos_test,
                output_dir=positive_test_output_dir, batch_size=config.get("tts_batch_size", 256)
            )
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

       
        if n_neg_test > 0:
            print_info(f"Generating {n_neg_test} adversarial negative clips for testing to '{negative_test_output_dir}'...")
            os.makedirs(negative_test_output_dir, exist_ok=True)
            adversarial_texts = config.get("custom_negative_phrases", [])
            if config.get("target_phrase"):
                adversarial_texts.extend(generate_adversarial_texts(
                    input_text=config["target_phrase"][0], N=n_neg_test
                ))
            if adversarial_texts:
                generate_samples(
                    text=adversarial_texts, max_samples=n_neg_test,
                    output_dir=negative_test_output_dir, batch_size=config.get("tts_batch_size", 256) // 4
                )
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        print_info("Synthetic data generation process finished.\n")

# --->>>
# Based on the median clip duration created, determine the total length of the training clips,
#  rounded to the nearest 1000 samples, and if the median is close to +750 milliseconds, set it to 32000,
#  as this is a good default value.

    n = 50  # sample size
    positive_clips = [str(i) for i in Path(config["wakeword_data_path"]).glob("*.wav")]
    duration_in_samples = []
    for i in range(n):
        sr, dat = scipy.io.wavfile.read(positive_clips[np.random.randint(0, len(positive_clips))])
        duration_in_samples.append(len(dat))

    config["total_length"] = int(round(np.median(duration_in_samples)/1000)*1000) + 12000  # add 750 ms to clip duration as buffer
    if config["total_length"] < 32000:
        config["total_length"] = 32000  # set a minimum of 32000 samples (2 seconds)
    elif abs(config["total_length"] - 32000) <= 4000:
        config["total_length"] = 32000

    # Do Data Augmentation
    if args.augment_clips is True:
        if not os.path.exists(os.path.join(feature_save_dir, "positive_features_train.npy")) or args.overwrite is True:


            aug_probs = {
                "SevenBandParametricEQ": 0.25,
                "TanhDistortion": 0.25,
                "PitchShift": 0.25,
                "BandStopFilter": 0.25,
                "AddColoredNoise": 0.25,
                "AddBackgroundNoise": 0.75, 
                "Gain": 1.0,
                "RIR": 0.5 
            }

            if 'rir_probability' in config:
                aug_probs['RIR'] = config['rir_probability']
            
            if 'background_noise_probability' in config:
                aug_probs['AddBackgroundNoise'] = config['background_noise_probability']
                
            # logging.info(f"Using dynamic augmentation probabilities: {aug_probs}")
            

            positive_clips_train = [str(i) for i in Path(config["wakeword_data_path"]).glob("*.wav")]*config["augmentation_rounds"]
            positive_clips_train_generator = augment_clips(positive_clips_train, total_length=config["total_length"],
                                                           batch_size=config["augmentation_batch_size"],
                                                           background_clip_paths=background_paths,
                                                           RIR_paths=rir_paths,
                                                           augmentation_probabilities=aug_probs)

           
            positive_clips_test = [str(i) for i in Path(config["wakeword_data_path"]).glob("*.wav")]*config["augmentation_rounds"]
            positive_clips_test_generator = augment_clips(positive_clips_test, total_length=config["total_length"],
                                                          batch_size=config["augmentation_batch_size"],
                                                          background_clip_paths=background_paths,
                                                          RIR_paths=rir_paths,
                                                          augmentation_probabilities=aug_probs)

            negative_clips_train = [str(i) for i in Path(config["background_data_path"]).glob("*.wav")]*config["augmentation_rounds"]
            negative_clips_train_generator = augment_clips(negative_clips_train, total_length=config["total_length"],
                                                           batch_size=config["augmentation_batch_size"],
                                                           background_clip_paths=background_paths,
                                                           RIR_paths=rir_paths,
                                                           augmentation_probabilities=aug_probs)

            
            negative_clips_test = [str(i) for i in Path(config["background_data_path"]).glob("*.wav")]*config["augmentation_rounds"]
            negative_clips_test_generator = augment_clips(negative_clips_test, total_length=config["total_length"],
                                                          batch_size=config["augmentation_batch_size"],
                                                          background_clip_paths=background_paths,
                                                          RIR_paths=rir_paths,
                                                          augmentation_probabilities=aug_probs)

            # Compute features and save to disk via memmapped arrays
            print_step_header(3, "Computing Nanowakeword features for generated samples")
            n_cpus = os.cpu_count()
            if n_cpus is None:
                n_cpus = 1
            else:
                n_cpus = n_cpus//2
            compute_features_from_generator(positive_clips_train_generator, n_total=len(os.listdir(config["wakeword_data_path"])),
                                            clip_duration=config["total_length"],
                                            output_file=os.path.join(feature_save_dir, "positive_features_train.npy"),
                                            device="gpu" if torch.cuda.is_available() else "cpu",
                                            ncpu=n_cpus if not torch.cuda.is_available() else 1)

            # compute_features_from_generator(negative_clips_train_generator, n_total=len(os.listdir(negative_train_output_dir)),
            compute_features_from_generator(negative_clips_train_generator, n_total=len(os.listdir(config["background_data_path"])),
                                            clip_duration=config["total_length"],
                                            output_file=os.path.join(feature_save_dir, "negative_features_train.npy"),
                                            device="gpu" if torch.cuda.is_available() else "cpu",
                                            ncpu=n_cpus if not torch.cuda.is_available() else 1)

            # compute_features_from_generator(positive_clips_test_generator, n_total=len(os.listdir(positive_test_output_dir)),
            compute_features_from_generator(positive_clips_test_generator, n_total=len(os.listdir(config["wakeword_data_path"])),
                                            clip_duration=config["total_length"],
                                            output_file=os.path.join(feature_save_dir, "positive_features_test.npy"),
                                            device="gpu" if torch.cuda.is_available() else "cpu",
                                            ncpu=n_cpus if not torch.cuda.is_available() else 1)

            # compute_features_from_generator(negative_clips_test_generator, n_total=len(os.listdir(negative_test_output_dir)),
            compute_features_from_generator(negative_clips_test_generator, n_total=len(os.listdir(config["background_data_path"])),
                                            clip_duration=config["total_length"],
                                            output_file=os.path.join(feature_save_dir, "negative_features_test.npy"),
                                            device="gpu" if torch.cuda.is_available() else "cpu",
                                            ncpu=n_cpus if not torch.cuda.is_available() else 1)
        
            
          
            batch_comp_config = config.get('batch_composition', {})
            source_dist = batch_comp_config.get('source_distribution', {})
            
            if source_dist.get('pure_noise', 0) > 0:
                               
                noise_source_paths = config.get("background_paths", [])
                pure_noise_clips = [str(i) for j in noise_source_paths for i in Path(j).glob("*.wav")]
                
                if pure_noise_clips:
                    
                    noise_aug_rounds = max(1, config.get("augmentation_rounds", 5) // 2)
                    
                    pure_noise_generator = augment_clips(
                        pure_noise_clips * noise_aug_rounds,
                        total_length=config["total_length"],
                        batch_size=config["augmentation_batch_size"],
                        background_clip_paths=background_paths,
                        RIR_paths=rir_paths,
                        augmentation_probabilities=aug_probs
                    )  

                    compute_features_from_generator(pure_noise_generator, n_total=len(pure_noise_clips) * noise_aug_rounds,
                        clip_duration=config["total_length"], 
                        output_file=os.path.join(feature_save_dir, "pure_noise_features.npy"),
                        device="gpu" if torch.cuda.is_available() else "cpu",
                        ncpu=n_cpus if not torch.cuda.is_available() else 1
                    )
                else:
                    print_info("WARNING: No audio files found in 'background_paths' to create pure_noise features.")

        else:
            logging.warning("Nanowakeword features already exist, skipping data augmentation and feature generation")




    # Create nanowakeword model
    if args.train_model is True:
        F = nanowakeword.utils.audio_processing.AudioFeatures(device='cpu')
        input_shape = np.load(os.path.join(feature_save_dir, "positive_features_test.npy")).shape[1:]

        # nww = Model(n_classes=1, input_shape=input_shape, model_type=config["model_type"],
        #             layer_dim=config["layer_size"], 
        seconds_per_example= 1280*input_shape[0]/16000

        # fp_val_path = config["false_positive_validation_data_path"]
        fp_val_path_fc = config.get("false_positive_validation_data_path")

        if fp_val_path_fc and os.path.exists(fp_val_path_fc):
            fp_val_path = fp_val_path_fc
            print_info(f"Using custom validation set from: {fp_val_path}")

        else:
            fp_val_path = os.path.join(feature_save_dir, "negative_features_test.npy")
            print_info(f"Custom validation set not found. Using generated negative test set as a fallback: {fp_val_path}")
  
        val_set_hrs = calculate_validation_duration_hours(fp_val_path, input_shape)
        # print(f"DEBUG: Using val_set_hrs = {val_set_hrs} for FP/hour calculation.")
        # Create data transform function for batch generation to handle differ clip lengths (todo: write tests for this)
        def f(x, n=input_shape[0]):
            """Simple transformation function to ensure negative data is the appropriate shape for the model size"""
            if n > x.shape[1] or n < x.shape[1]:
                x = np.vstack(x)
                new_batch = np.array([x[i:i+n, :] for i in range(0, x.shape[0]-n, n)])
            else:
                return x
            return new_batch


        batch_comp_config = config.get('batch_composition', {})
        total_batch_size = batch_comp_config.get('batch_size', 32)
        source_dist = batch_comp_config.get('source_distribution', 
            {'positive': 30, 'negative_speech': 50, 'pure_noise': 20})
        
        batch_n_per_class = {
            'positive': int(round(total_batch_size * (source_dist.get('positive', 0) / 100))),
            'adversarial_negative': int(round(total_batch_size * (source_dist.get('negative_speech', 0) / 100))),
            'pure_noise': int(round(total_batch_size * (source_dist.get('pure_noise', 0) / 100)))
        }

       
        
        # Create a master list of all possible feature files
        possible_feature_files = {
            'positive': os.path.join(feature_save_dir, "positive_features_train.npy"),
            'adversarial_negative': os.path.join(feature_save_dir, "negative_features_train.npy"),
            'pure_noise': os.path.join(feature_save_dir, "pure_noise_features.npy")
        }

        # Only add sources that will be used in the batch and whose feature files exist,
        # to the final dictionary.
        final_data_files = {}
        for source_name, num_samples in batch_n_per_class.items():
            if num_samples > 0 and source_name in possible_feature_files and os.path.exists(possible_feature_files[source_name]):
                final_data_files[source_name] = possible_feature_files[source_name]

        final_label_transforms = {}
        for key in final_data_files.keys():
            if key == 'positive':
                final_label_transforms[key] = lambda x, k=key: [1] * len(x)
            else:
                final_label_transforms[key] = lambda x, k=key: [0] * len(x)
        
        data_transforms = {key: f for key in final_data_files.keys()}
        
        batch_generator = mmap_batch_generator(
            data_files=final_data_files,
            n_per_class={k: v for k, v in batch_n_per_class.items() if k in final_data_files},
            data_transform_funcs=data_transforms,
            label_transform_funcs=final_label_transforms
        )


        class IterDataset(torch.utils.data.IterableDataset):
            def __init__(self, generator):
                self.generator = generator

            def __iter__(self):
                return self.generator

        n_cpus = os.cpu_count()
        if n_cpus is None:
            n_cpus = 1
        else:
            n_cpus = n_cpus//2
        # X_train = torch.utils.data.DataLoader(IterDataset(batch_generator),
                                            #   batch_size=None, num_workers=n_cpus, prefetch_factor=16)
        X_train = torch.utils.data.DataLoader(IterDataset(batch_generator),
                                     batch_size=None, num_workers=0)
       

        # Loading and preparing false positive validation data 
        x_val_fp_numpy = np.load(fp_val_path)
        x_val_fp_labels_numpy = np.zeros(x_val_fp_numpy.shape[0]).astype(np.float32)

        val_batch_size = config.get("batch_size", 32) * 2

        X_val_fp = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(x_val_fp_numpy), torch.from_numpy(x_val_fp_labels_numpy)),
            # batch_size=len(x_val_fp_labels_numpy) if len(x_val_fp_labels_numpy) > 0 else 1
            batch_size=val_batch_size
        )

        X_val_pos = np.load(os.path.join(feature_save_dir, "positive_features_test.npy"))
        X_val_neg = np.load(os.path.join(feature_save_dir, "negative_features_test.npy"))
        labels = np.hstack((np.ones(X_val_pos.shape[0]), np.zeros(X_val_neg.shape[0]))).astype(np.float32)

        X_val = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.from_numpy(np.vstack((X_val_pos, X_val_neg))),
                torch.from_numpy(labels)
                ),
            # batch_size=len(labels)
            batch_size=val_batch_size
        )

        print_step_header(4, "Starting training...")
        
        nww = Model(n_classes=1, input_shape=input_shape,
                    model_type=config["model_type"],
                    layer_dim=config["layer_size"],
                    n_blocks=config["n_blocks"],
                    dropout_prob=config.get("dropout_prob", 0.5),
                    seconds_per_example=seconds_per_example
                    )
      
        nww.setup_optimizer_and_scheduler(config)
        
        # Run auto training
        print_info(f"Using model architecture: 🤍 {config['model_type'].upper()} (Bidirectional)")
        best_model = nww.auto_train(
            X_train=X_train,
            X_val=X_val,
            false_positive_val_data=X_val_fp,
            steps=config.get("steps", 15000), 
            max_negative_weight=config.get("max_negative_weight", 3.0), # Default value using .get()
            target_fp_per_hour=config.get("target_false_positives_per_hour", 0.1),
            val_set_hrs=val_set_hrs
        )

        nww.plot_history(config["output_dir"])
        
        # Export model
        nww.export_model(model=best_model, model_name=config["model_name"], output_dir=config["output_dir"])
        convert_onnx_to_tflite(os.path.join(config["output_dir"], config["model_name"] + ".onnx"),
                               os.path.join(config["output_dir"], config["model_name"] + ".tflite"))
        
if __name__ == '__main__':
    main()
