# architectures.py

import torch
import torch.nn as nn
import numpy as np
import math

class PositionalEncoding(nn.Module):
    """
    Injects positional information into the input embeddings.
    This is crucial for Transformers as they do not have a built-in sense of sequence order.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# CNN architecture
class CNNModel(nn.Module):
    def __init__(self, input_shape, embedding_dim, dropout_prob, activation_fn):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = activation_fn
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = activation_fn
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        conv_output_size = self._get_conv_output(input_shape)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(conv_output_size, 128)
        self.relu3 = activation_fn
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(128, embedding_dim) 
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
        x = self.fc2(x) 
        return x

# LSTM architecture
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, embedding_dim, bidirectional, dropout_prob):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, n_layers,
            batch_first=True, bidirectional=bidirectional,
            dropout=dropout_prob if n_layers > 1 else 0
        )
        linear_input_size = hidden_dim * 2 if bidirectional else hidden_dim
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(linear_input_size, embedding_dim) 
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        out = self.dropout(last_output)
        out = self.fc(out)
        return out

# DNN architecture
class FCNBlock(nn.Module):
    def __init__(self, layer_dim, activation_fn):
        super().__init__()
        self.fcn_layer = nn.Linear(layer_dim, layer_dim)
        self.relu = activation_fn
        self.layer_norm = nn.LayerNorm(layer_dim)
    def forward(self, x):
        return self.relu(self.layer_norm(self.fcn_layer(x)))
class Net(nn.Module):
    def __init__(self, input_shape, layer_dim, n_blocks, embedding_dim, dropout_prob, activation_fn):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(input_shape[0]*input_shape[1], layer_dim)
        self.relu1 = activation_fn
        self.layernorm1 = nn.LayerNorm(layer_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.blocks = nn.ModuleList([FCNBlock(layer_dim, activation_fn) for _ in range(n_blocks)])
        self.last_layer = nn.Linear(layer_dim, embedding_dim) 
    def forward(self, x):
        x = self.relu1(self.layernorm1(self.layer1(self.flatten(x))))
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.last_layer(x) 
        return x
        
# GRU architecture
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, embedding_dim, bidirectional, dropout_prob):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim, n_layers,
            batch_first=True, bidirectional=bidirectional,
            dropout=dropout_prob if n_layers > 1 else 0
        )
        linear_input_size = hidden_dim * 2 if bidirectional else hidden_dim
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(linear_input_size, embedding_dim) # <--  
    def forward(self, x):
        gru_out, _ = self.gru(x)
        last_output = gru_out[:, -1, :]
        out = self.dropout(last_output)
        out = self.fc(out)
        return out
           
# RNN architecture
class RNNModel(nn.Module):
    def __init__(self, input_shape, embedding_dim, n_blocks, dropout_prob):
        super().__init__()
        self.layer1 = nn.LSTM(
            input_shape[-1], 64, num_layers=n_blocks, bidirectional=True,
            batch_first=True, dropout=dropout_prob if n_blocks > 1 else 0
        )
        self.dropout = nn.Dropout(dropout_prob)
        self.layer2 = nn.Linear(64*2, embedding_dim)   
    def forward(self, x):
        out, _ = self.layer1(x)
        last_output = self.dropout(out[:, -1])
        return self.layer2(last_output)

# Transformer architecture
class TransformerModel(nn.Module):
# This is our new Transformer model implementation
    def __init__(self, input_dim, d_model, n_head, n_layers, embedding_dim, dropout_prob):
        super().__init__()
        # 1. Input projection: Project input features to the model's dimension
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout_prob)
        
        # 3. Transformer Encoder Layers
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_head, 
            dim_feedforward=d_model * 4, # A common practice
            dropout=dropout_prob,
            batch_first=True # IMPORTANT: Makes handling batch dimension easier
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=n_layers)
        
        # 4. Output projection: Project to the final embedding dimension
        self.output_proj = nn.Linear(d_model, embedding_dim)
        self.d_model = d_model

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        
        # Project input to d_model
        x = self.input_proj(x) * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x.permute(1, 0, 2)).permute(1, 0, 2) # Permute for pos_encoder, then back
        
        # Pass through transformer
        transformer_out = self.transformer_encoder(x)
        
        # Pooling: Average the outputs of all timesteps to get a single vector
        pooled_out = transformer_out.mean(dim=1)
        
        # Final projection to embedding space
        out = self.output_proj(pooled_out)
        return out


# CRNN architecture
class CRNNModel(nn.Module):
# A powerful hybrid architecture combining CNNs for feature extraction
# and RNNs (LSTM/GRU) for temporal sequence modeling.
    def __init__(self, input_shape, rnn_type, rnn_hidden_size, n_rnn_layers, 
                    cnn_channels, embedding_dim, dropout_prob, activation_fn):
        super().__init__()

        # --- 1. CNN Frontend ---
        # Dynamically builds CNN layers based on the config.
        cnn_layers = []
        in_channels = 1 # Input is a single-channel spectrogram
        for out_channels in cnn_channels:
            cnn_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            cnn_layers.append(nn.BatchNorm2d(out_channels))
            cnn_layers.append(activation_fn)
            cnn_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels
        self.cnn = nn.Sequential(*cnn_layers)

        # --- 2. Determine RNN input size ---
        # We run a dummy tensor through the CNN to find its output shape.
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, *input_shape)
            conv_output = self.cnn(dummy_input)
            batch_size, channels, height, width = conv_output.shape
            rnn_input_size = channels * height

        # --- 3. RNN Backend ---
        # User can choose between LSTM or GRU.
        if rnn_type.lower() == 'gru':
            self.rnn = nn.GRU(
                input_size=rnn_input_size,
                hidden_size=rnn_hidden_size,
                num_layers=n_rnn_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout_prob if n_rnn_layers > 1 else 0
            )
        else: # Default to LSTM
            self.rnn = nn.LSTM(
                input_size=rnn_input_size,
                hidden_size=rnn_hidden_size,
                num_layers=n_rnn_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout_prob if n_rnn_layers > 1 else 0
            )
        
        self.dropout = nn.Dropout(dropout_prob)
        
        # --- 4. Final Classifier ---
        # Projects the RNN output to the final embedding dimension.
        self.fc = nn.Linear(rnn_hidden_size * 2, embedding_dim) # *2 for bidirectional

    def forward(self, x):
        # Input x shape: [batch_size, height, width]
        if x.dim() == 3:
            x = x.unsqueeze(1) # Add channel dimension -> [batch_size, 1, height, width]

        # Pass through CNN frontend
        conv_out = self.cnn(x)
        
        # Reshape and Permute for RNN
        # The magic that connects CNN to RNN
        B, C, H, W = conv_out.shape
        # Combine Channels and Height into a single feature dimension
        rnn_in = conv_out.view(B, C * H, W)
        # Permute to make Width the sequence dimension for the RNN
        rnn_in = rnn_in.permute(0, 2, 1) # -> [batch_size, width_seq_len, features]

        # Pass through RNN backend
        rnn_out, _ = self.rnn(rnn_in)
        
        # We take the output of the last time step
        last_step_out = rnn_out[:, -1, :]
        
        # Apply dropout and final projection
        out = self.dropout(last_step_out)
        out = self.fc(out)
        return out


# TCN architecture
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        # First causal convolution
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                stride=stride, padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Second causal convolution
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                stride=stride, padding=padding, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # 1x1 convolution for the residual connection if channel sizes don't match
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        
        # We store padding value to use it in the forward pass
        self.padding = padding

    def forward(self, x):
        # Apply first convolution block
        out = self.conv1(x)
        out = out[:, :, :-self.padding] # Apply chomp manually
        out = self.relu1(out)
        out = self.dropout1(out)

        # Apply second convolution block
        out = self.conv2(out)
        out = out[:, :, :-self.padding] # Apply chomp manually
        out = self.relu2(out)
        out = self.dropout2(out)
        
        # Apply residual connection
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

# Now, we define the main TCN model that stacks these blocks.
class TCNModel(nn.Module):
    def __init__(self, input_dim, num_channels, embedding_dim, kernel_size, dropout_prob):
        super(TCNModel, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                        padding=(kernel_size-1) * dilation_size, dropout=dropout_prob)]

        self.tcn_blocks = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], embedding_dim)

    def forward(self, x):
        # Input x shape: [batch_size, seq_len, input_dim]
        # TCN expects: [batch_size, input_dim, seq_len]
        # So, we permute the dimensions.
        x = x.permute(0, 2, 1)

        # Pass through the TCN blocks
        tcn_out = self.tcn_blocks(x)

        # We take the output of the last time step for classification
        last_step_out = tcn_out[:, :, -1]
        
        # Final projection to embedding space
        out = self.fc(last_step_out)
        return out


# QuartzNet architecture
class QuartzNetBlock(nn.Module):
# An implementation of QuartzNet, a highly parameter-efficient and powerful
# architecture from NVIDIA, based on 1D time-channel separable convolutions.

# This is the core building block of the QuartzNet model.
    def __init__(self, in_channels, out_channels, kernel_size, dropout):
        super().__init__()
        # Time-channel separable convolution consists of two parts:
        # 1. Depthwise convolution (applied to each channel independently)
        self.depthwise_conv = nn.Conv1d(in_channels, in_channels, kernel_size,
                                        padding='same', groups=in_channels)
        # 2. Pointwise convolution (a 1x1 conv to combine channel information)
        self.pointwise_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        
        # Residual connection might require a projection if channel sizes differ
        self.residual_connector = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels)
        ) if in_channels != out_channels else None

    def forward(self, x):
        residual = x
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.batch_norm(x)
        
        # Add residual connection
        if self.residual_connector:
            residual = self.residual_connector(residual)
        x += residual
        
        x = self.activation(x)
        x = self.dropout(x)
        return x

# The main QuartzNet model that stacks the blocks based on the config.
class QuartzNetModel(nn.Module):
    def __init__(self, input_dim, quartznet_config, embedding_dim, dropout_prob):
        super().__init__()
        layers = []
        in_channels = input_dim

        # Build the network layer by layer from the config
        for channels, kernel_size, repetitions in quartznet_config:
            for _ in range(repetitions):
                layers.append(QuartzNetBlock(in_channels, channels, kernel_size, dropout_prob))
                in_channels = channels # Output of one block is input to the next
        
        self.quartznet_blocks = nn.Sequential(*layers)
        self.fc = nn.Linear(in_channels, embedding_dim)

    def forward(self, x):
        # Input x shape: [batch_size, seq_len, input_dim]
        # Conv1d expects: [batch_size, input_dim, seq_len]
        x = x.permute(0, 2, 1)

        x = self.quartznet_blocks(x)

        # Average pooling over the time dimension to get a single vector
        pooled_out = x.mean(dim=2)
        
        # Final projection to the embedding space
        out = self.fc(pooled_out)
        return out


# ###
class Swish(nn.Module): # The Swish activation function
    def forward(self, x):
        return x * torch.sigmoid(x)

# conformer architecture
class ConvolutionModule(nn.Module):
# An implementation of the Conformer architecture, which combines Transformers
# and CNNs to achieve state-of-the-art performance in speech tasks.

# Helper Modules for the Conformer Block 
    def __init__(self, d_model, kernel_size=31):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        # Pointwise -> Depthwise -> Pointwise convolution structure
        self.conv1 = nn.Conv1d(d_model, d_model * 2, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(d_model, d_model, kernel_size,
                                        groups=d_model, padding='same')
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.swish = Swish()
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        x = self.layer_norm(x)
        x = x.permute(0, 2, 1) # -> [B, d_model, seq_len]
        
        x = self.conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.swish(x)
        x = self.conv2(x)
        x = self.dropout(x)
        
        return x.permute(0, 2, 1) # -> [B, seq_len, d_model]

class FeedForwardModule(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.swish = Swish()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = self.swish(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x

# This is the main Conformer block combining all pieces.
class ConformerBlock(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        self.ff1 = FeedForwardModule(d_model, dropout)
        self.attention = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.conv_module = ConvolutionModule(d_model)
        self.ff2 = FeedForwardModule(d_model, dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # The Conformer architecture applies modules with half-step residual connections
        x = x + 0.5 * self.ff1(x)
        
        attention_out, _ = self.attention(x, x, x)
        x = x + attention_out
        
        x = x + self.conv_module(x)
        x = x + 0.5 * self.ff2(x)
        x = self.layer_norm(x)
        return x

# The final Conformer model that stacks the blocks.
class ConformerModel(nn.Module):
    def __init__(self, input_dim, d_model, n_head, n_layers, embedding_dim, dropout_prob):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.dropout = nn.Dropout(p=dropout_prob)
        
        self.conformer_blocks = nn.Sequential(
            *[ConformerBlock(d_model, n_head, dropout_prob) for _ in range(n_layers)]
        )
        self.output_proj = nn.Linear(d_model, embedding_dim)

    def forward(self, x):
        # Input x shape: [batch_size, seq_len, input_dim]
        x = self.input_proj(x)
        x = self.dropout(x)
        
        x = self.conformer_blocks(x)
        
        # Average pooling over time
        pooled_out = x.mean(dim=1)
        
        out = self.output_proj(pooled_out)
        return out


# E-branchformer architecture
class MergingModule(nn.Module):
# An implementation of E-Branchformer, a state-of-the-art evolution of the
# Conformer model, which uses parallel branches for attention and convolution.
# This is considered the final, most advanced architecture in our collection.

# We reuse the Swish and FeedForwardModule from the Conformer implementation above.

# A special module to smartly merge the outputs of the two parallel branches.
    def __init__(self, d_model):
        super().__init__()
        self.gate = nn.Linear(d_model, d_model)

    def forward(self, attn_out, conv_out):
        gate_val = torch.sigmoid(self.gate(conv_out))
        return attn_out * gate_val + conv_out * (1 - gate_val)

# The main E-Branchformer block.
class EBranchformerBlock(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        # --- Main Branches ---
        # 1. Attention Branch
        self.attn_branch_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        # 2. Convolution Branch
        self.conv_branch = ConvolutionModule(d_model) # Reusing from Conformer

        # --- Merging and Final Layer ---
        self.merger = MergingModule(d_model)
        self.final_norm = nn.LayerNorm(d_model)
        self.ffn = FeedForwardModule(d_model, dropout) # Reusing from Conformer

    def forward(self, x):
        # --- Parallel Execution ---
        # Attention Branch
        attn_input = self.attn_branch_norm(x)
        attn_out, _ = self.attention(attn_input, attn_input, attn_input)
        
        # Convolution Branch
        conv_out = self.conv_branch(x)

        # Merging
        merged_out = self.merger(attn_out, conv_out)
        
        # Final Feed-Forward Network 
        x = x + merged_out # Add to the original input
        x = self.final_norm(x)
        x = x + self.ffn(x) # Add FFN output
        return x

# The final E-Branchformer model that stacks the blocks.
class EBranchformerModel(nn.Module):
    def __init__(self, input_dim, d_model, n_head, n_layers, embedding_dim, dropout_prob):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.dropout = nn.Dropout(p=dropout_prob)
        
        self.branchformer_blocks = nn.Sequential(
            *[EBranchformerBlock(d_model, n_head, dropout_prob) for _ in range(n_layers)]
        )
        self.output_proj = nn.Linear(d_model, embedding_dim)
    
    def forward(self, x):
        x = self.input_proj(x)
        x = self.dropout(x)
        
        x = self.branchformer_blocks(x)
        
        pooled_out = x.mean(dim=1)
        out = self.output_proj(pooled_out)
        return out


