
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TransformerFeaturizer(nn.Module):
    """
    Featurizer based on the TransformerEncoder module from PyTorch.
    """

    def __init__(
        self,
        d_feat_in,
        d_time_in,
        d_feat=32,
        d_time=32,
        nhead=4,
        num_encoder_layers=4,
        dim_feedforward=128,
        sum_features=False,
        batch_first=True,
        use_embedding=True,
        activation_fn=None,
    ):
        """
        Parameters
        ----------
        d_feat_in : int
            The dimension of the input features.
        d_time_in : int
            The dimension of the input time.
        d_feat : int
            The dimension of the output features.
        d_time : int
            The dimension of the output time.
        nhead : int
            The number of heads in the multiheadattention models.
        num_encoder_layers : int
            The number of sub-encoder-layers in the encoder.
        sum_features : bool, optional
            Whether to sum the features along the sequence dimension. Default: False
        dim_feedforward : int
            The dimension of the feedforward network model.
        batch_first : bool, optional
            If True, then the input and output tensors are provided as
            (batch, seq, feature). Default: True
        activation_fn : callable, optional
            The activation function to use for the embedding layer. Default: None
        """
        super().__init__()
        self.d_feat = d_feat
        self.d_time = d_time
        self.d_model = d_feat + d_time
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.batch_first = True
        self.use_embedding = use_embedding
        self.activation_fn = activation_fn

        encoder_layer = TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers)
        self.feat_embedding_layer = nn.Linear(d_feat_in, d_feat)
        self.time_embedding_layer = nn.Linear(d_time_in, d_time)

    def forward(self, x, t, padding_mask=None):
        x = self.feat_embedding_layer(x)
        t = self.time_embedding_layer(t)
        src = torch.cat((x, t), dim=-1)
        output = self.transformer_encoder(
            src, src_key_padding_mask=padding_mask)

        # NOTE: dimension only works when batch_first=True
        if padding_mask is None:
            output = output.sum(dim=1)
        else:
            if not self.training:
                # apply correct padding mask for evaluation
                # this happens because with both self.eval() and torch.no_grad()
                # the transformer encoder changes the length of the output to
                # match the max non-padded length in the batch
                max_seq_len = output.shape[1]
                padding_mask = padding_mask[:, :max_seq_len]
            output = output.masked_fill(padding_mask.unsqueeze(-1), 0)
            output = output.sum(dim=1)

        return output

class MLP(nn.Module):
    """
    MLP with a variable number of hidden layers.

    Attributes
    ----------
    layers : nn.ModuleList
        The layers of the MLP.
    activation_fn : callable
        The activation function to use.
    """
    def __init__(self, input_size, hidden_sizes=[512],
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
        layer_sizes = [input_size] + hidden_sizes

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
