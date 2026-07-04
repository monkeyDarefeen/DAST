from speechbrain.lobes.models.ECAPA_TDNN import *
import os
import csv
import numpy as np
import torch
import os
import numpy as np
import torch
from time import time

def resample_tensor(tensor, target_length):
    """
    Resamples a tensor (B, D, T) along the temporal dimension (T)
    to a new target_length.

    Downsampling uses average pooling.
    Upsampling uses cyclical repetition (tiling).

    Args:
        tensor (torch.Tensor): Input tensor of shape (B, D, T)
        target_length (int): Desired temporal length

    Returns:
        torch.Tensor: Resampled tensor of shape (B, D, target_length)
    """
    # Get original shape, B=Batch, D=Dimension, T=Time
    B, D, T = tensor.shape

    if T == target_length:
        return tensor

    # --- Downsampling (Average pooling) ---
    if T > target_length:
        base_bin = T // target_length
        extra = T % target_length

        n_base = target_length - extra
        n_extra = extra

        outputs = []
        idx = 0

        for _ in range(n_base):
            slice_ = tensor[:, :, idx:idx + base_bin]
            mean_ = slice_.mean(dim=2, keepdim=True)
            outputs.append(mean_)
            idx += base_bin

        for _ in range(n_extra):
            slice_ = tensor[:, :, idx:idx + base_bin + 1]
            mean_ = slice_.mean(dim=2, keepdim=True)
            outputs.append(mean_)
            idx += base_bin + 1
        
        return torch.cat(outputs, dim=2)

    # --- Upsampling (Cyclical Repetition / Tiling) ---
    else:
        # This is the new logic as requested
        num_repeats = target_length // T
        remainder = target_length % T

        # Repeat the tensor 'num_repeats' times
        # e.g., if T=3, target=10 -> num_repeats=3. 
        # (a,b,c) -> (a,b,c,a,b,c,a,b,c)
        if num_repeats > 0:
            repeated_part = tensor.repeat(1, 1, num_repeats)
        else:
            # Handle case where T > target_length but we are in the else block
            # (shouldn't happen with the first 'if' check, but good for safety)
            # Or more likely, T < target_length but num_repeats is 0
            # (e.g., T=7, target=5, which is handled by downsampling)
            # We'll create an empty tensor to cat with
             repeated_part = torch.empty(B, D, 0, device=tensor.device, dtype=tensor.dtype)

        # Get the remaining 'remainder' elements from the start
        # e.g., if T=3, target=10 -> remainder=1.
        # (a,b,c) -> (a)
        remainder_part = tensor[:, :, :remainder]

        # Concatenate the parts
        # (a,b,c,a,b,c,a,b,c) + (a) -> (a,b,c,a,b,c,a,b,c,a)
        return torch.cat([repeated_part, remainder_part], dim=2)

class WeightedLayerSum(nn.Module):
    """
    Computes a learnable weighted sum of a stack of layer outputs.

    This module takes a single tensor of shape (L, ...), where L is
    the number of layers, and computes a weighted sum along dim=0.

    The weights are normalized using softmax to ensure they sum to 1.
    """
    def __init__(self, num_layers=24):
        """
        Args:
            num_layers (int): The number of layers to be summed (L).
            initial_weights (list or torch.Tensor, optional): Initial values for weights.
                                                              Should be of length num_layers.
            layer_emphasis (list or torch.Tensor, optional): Fixed bias added to weights before softmax.
                                                             Allows emphasizing specific layers permanently.
        """
        super().__init__()
        self.num_layers = num_layers
        self.layer_weights = nn.Parameter(torch.ones(self.num_layers))
        # if initial_weights is not None:
        #     if len(initial_weights) != num_layers:
        #         raise ValueError(f"initial_weights must have length {num_layers}")
        #     # Convert to tensor if it's a list
        #     if isinstance(initial_weights, list):
        #         initial_weights = torch.tensor(initial_weights, dtype=torch.float32)
        #     self.layer_weights = nn.Parameter(initial_weights)
        # else:
        #     # Define learnable parameters, one for each layer.
        #     self.layer_weights = nn.Parameter(torch.ones(self.num_layers))

        # if layer_emphasis is not None:
        #     if len(layer_emphasis) != num_layers:
        #         raise ValueError(f"layer_emphasis must have length {num_layers}")
        #     if isinstance(layer_emphasis, list):
        #         layer_emphasis = torch.tensor(layer_emphasis, dtype=torch.float32)
        #     self.register_buffer('layer_emphasis', layer_emphasis)
        # else:
        #     self.register_buffer('layer_emphasis', torch.zeros(num_layers))

    def forward(self, layer_results):
        """
        Forward pass for the module.

        Args:
            layer_results (torch.Tensor): A single stacked tensor of layer outputs.
                Expected shape: (L, T, B, D) or (num_layers, ...)
                e.g., [24, 814, 64, 1024]

        Returns:
            torch.Tensor: The weighted sum of the layers, with the first
                          dimension (L) collapsed.
                          e.g., [814, 64, 1024]
        """
        # --- Step 1: Compute normalized weights ---
        # self.layer_weights shape: (24)
        # normalized_weights shape: (24)
        normalized_weights = nn.functional.softmax(self.layer_weights, dim=0)

        # --- Step 2: Reshape weights for broadcasting ---
        # We need to add dimensions to match the input tensor's shape.
        # (24) -> (24, 1, 1, 1)
        # This allows (24, 1, 1, 1) to multiply with (24, 814, 64, 1024)
        
        # Get the number of dimensions in the input tensor
        num_dims = layer_results.dim() 
        
        # Create the new shape: (L, 1, 1, ...)
        # e.g., for 4 dims: (24, 1, 1, 1)
        # e.g., for 3 dims: (24, 1, 1)
        reshape_dims = (self.num_layers,) + (1,) * (num_dims - 1)
        weights_reshaped = normalized_weights.view(reshape_dims)

        # --- Step 3: Compute weighted sum ---
        # 1. Multiply layer_results by weights (broadcasting)
        #    (24, 814, 64, 1024) * (24, 1, 1, 1)
        # 2. Sum along the first dimension (dim=0) to collapse the 24 layers
        # Output shape: (814, 64, 1024)
        weighted_sum = (layer_results * weights_reshaped).sum(dim=0)

        return weighted_sum
    
    
class ECAPA_TDNN_test(torch.nn.Module):
    """An implementation of the speaker embedding model in a paper.
    "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in
    TDNN Based Speaker Verification" (https://arxiv.org/abs/2005.07143).

    Arguments
    ---------
    input_size : int
        Expected size of the input dimension.
    device : str
        Device used, e.g., "cpu" or "cuda".
    lin_neurons : int
        Number of neurons in linear layers.
    activation : torch class
        A class for constructing the activation layers.
    channels : list of ints
        Output channels for TDNN/SERes2Net layer.
    kernel_sizes : list of ints
        List of kernel sizes for each layer.
    dilations : list of ints
        List of dilations for kernels in each layer.
    attention_channels: int
        The number of attention channels.
    res2net_scale : int
        The scale of the Res2Net block.
    se_channels : int
        The number of output channels after squeeze.
    global_context: bool
        Whether to use global context.
    groups : list of ints
        List of groups for kernels in each layer.
    dropout : float
        Rate of channel dropout during training.

    Example
    -------
    >>> input_feats = torch.rand([5, 120, 80])
    >>> compute_embedding = ECAPA_TDNN(80, lin_neurons=192)
    >>> outputs = compute_embedding(input_feats)
    >>> outputs.shape
    torch.Size([5, 1, 192])
    """

    def __init__(
        self,
        input_size=80,
        device="cpu",
        lin_neurons=192,
        activation=torch.nn.ReLU,
        channels=[1024, 1024, 1024, 1024, 3072],
        kernel_sizes=[5, 3, 3, 3, 1],
        dilations=[1, 2, 3, 4, 1],
        attention_channels=128,
        res2net_scale=8,
        se_channels=128,
        global_context=True,
        groups=[1, 1, 1, 1, 1],
        dropout=0.0,
        # weighted_sum_initial_weights=None,
        # weighted_sum_layer_emphasis=None,
    ):
        super().__init__()
        assert len(channels) == len(kernel_sizes)
        assert len(channels) == len(dilations)
        self.channels = channels
        self.blocks = nn.ModuleList()
        self.blocks_features = nn.ModuleList()

        # The initial TDNN layer
        self.blocks.append(
            TDNNBlock(
                input_size,
                channels[0],
                kernel_sizes[0],
                dilations[0],
                activation,
                groups[0],
                dropout,
            )
        )

        # SE-Res2Net layers
        for i in range(1, len(channels) - 1):
            self.blocks.append(
                SERes2NetBlock(
                    channels[i - 1],
                    channels[i],
                    res2net_scale=res2net_scale,
                    se_channels=se_channels,
                    kernel_size=kernel_sizes[i],
                    dilation=dilations[i],
                    activation=activation,
                    groups=groups[i],
                    dropout=dropout,
                )
            )
            
        
        # The initial TDNN layer
        # self.blocks_features.append(
        #     TDNNBlock(
        #         1024,
        #         channels[0],
        #         kernel_sizes[0],
        #         dilations[0],
        #         activation,
        #         groups[0],
        #         dropout,
        #     )
        # )

        # SE-Res2Net layers
        for i in range(0, len(channels) - 1):
            self.blocks_features.append(
                SERes2NetBlock(
                    channels[i],
                    channels[i],
                    res2net_scale=res2net_scale,
                    se_channels=se_channels,
                    kernel_size=kernel_sizes[i],
                    dilation=dilations[i],
                    activation=activation,
                    groups=groups[i],
                    dropout=dropout,
                )
            )
            
            
        

        # Multi-layer feature aggregation
        self.mfa = TDNNBlock(
            channels[-2] * (len(channels) - 2),
            channels[-1],
            kernel_sizes[-1],
            dilations[-1],
            activation,
            groups=groups[-1],
            dropout=dropout,
        )
        # --- Custom Layer Emphasis Initialization ---
        # You can define your specific layer emphasis here if not passed as an argument.
        # if weighted_sum_layer_emphasis is None:
        #     # Example: Emphasize the last 4 layers (20-23)
        #     emphasis = torch.zeros(24)
        #     emphasis[6] = 3.0 
        #     emphasis[10] = 3.0 
        #     emphasis[18] = 3.0 
        #     emphasis[20] = 3.0 
        #     weighted_sum_layer_emphasis = emphasis
        #     # pass

        self.weighted_sum = WeightedLayerSum(
            # initial_weights=weighted_sum_initial_weights,
            # layer_emphasis=weighted_sum_layer_emphasis
        )
        # Attentive Statistical Pooling
        self.asp = AttentiveStatisticsPooling(
            channels[-1],
            attention_channels=attention_channels,
            global_context=global_context,
        )
        self.asp_bn = BatchNorm1d(input_size=channels[-1] * 2)

        # Final linear transformation
        self.fc = Conv1d(
            in_channels=channels[-1] * 2,
            out_channels=lin_neurons,
            kernel_size=1,
        )
        # breakpoint()

    def forward(self, x, features=None, folder_name="None", lengths=None):
        """Returns the embedding vector.

        Arguments
        ---------
        x : torch.Tensor
            Tensor of shape (batch, time, channel).
        lengths : torch.Tensor
            Corresponding relative lengths of inputs.

        Returns
        -------
        x : torch.Tensor
            Embedding vector.
        """
        # Minimize transpose for efficiency
        # breakpoint()
        features = self.weighted_sum(features)
        x = x.transpose(1, 2)
        features = features.transpose(0, 1)
        features = features.transpose(1, 2)
        # breakpoint()
        xl = []
        xl_features = []
        for layer in self.blocks:
            try:
                x = layer(x, lengths=lengths)
            except TypeError:
                x = layer(x)
            xl.append(x)
        # breakpoint()
        for layer in self.blocks_features:
            try:
                features = layer(features, lengths=lengths)
            except TypeError:
                features = layer(features)
            xl_features.append(features)
        # Multi-layer feature aggregation
        
        x_cat = torch.cat(xl[1:], dim=1)
        x_feature_cat = torch.cat(xl_features[1:], dim=1)
        # breakpoint()
        x_feature_cat = resample_tensor(x_feature_cat, len(x_cat[0][0]))
        x_mul = torch.mul(x_feature_cat, x_cat)

        x_mfa = self.mfa(x_mul)

        # Attentive Statistical Pooling
        x_asp = self.asp(x_mfa, lengths=lengths)
        x_aspbn = self.asp_bn(x_asp)
        # Final linear transformation
        x_fc = self.fc(x_aspbn)
        x_out = x_fc.transpose(1, 2)
        return x_out
    
