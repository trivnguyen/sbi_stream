
from typing import List, Dict, Any, Tuple

import torch
import torch.nn as nn

class MLPBatchNorm(nn.Module):
    """
    MLP with a variable number of hidden layers.

    Attributes
    ----------
    layers : nn.ModuleList
        The layers of the MLP.
    activation_fn : callable
        The activation function to use.
    """
    def __init__(self, input_size, output_size, hidden_sizes=[512],
                 activation_fn=nn.ReLU(), batch_norm=False, dropout=0.0):
        """
        Parameters
        ----------
        input_size : int
            The size of the input
        output_size : int
            The number of classes
        hidden_sizes : list of int, optional
            The sizes of the hidden layers. Default: [512]
        activation_fn : callable, optional
            The activation function to use. Default: nn.ReLU()
        batch_norm: bool, optional
            Whether to use batch normalization. Default: False
        dropout: float, optional
            The dropout rate. Default: 0.0
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.activation_fn = activation_fn

        # Create a list of all layer sizes: input, hidden, and output
        layer_sizes = [input_size] + hidden_sizes + [output_size]

        # Create layers dynamically
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            in_dim = layer_sizes[i]
            out_dim = layer_sizes[i + 1]
            self.layers.append(nn.Linear(in_dim, out_dim))
            self.layers.append(activation_fn)
            if batch_norm:
                self.layers.append(nn.BatchNorm1d(out_dim))
            self.layers.append(nn.Dropout(dropout))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)
