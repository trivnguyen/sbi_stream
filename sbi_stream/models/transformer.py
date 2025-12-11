
from typing import Optional, Callable, Dict

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch import Tensor

from sbi_stream.models import utils
from sbi_stream.models import mlp


class Transformer(nn.Module):
    """ Transformer model """

    def __init__(
        self, feat_input_size: int, pos_input_size: int, feat_embed_size: int = 32,
        pos_embed_size: int = 32, nhead: int = 4, num_encoder_layers: int = 4,
        dim_feedforward: int = 128, sum_features: bool = False,
        activation_name: str = None, activation_args: Optional[Dict] = None,
    ) -> None:
        """
        Parameters
        ----------
        feat_input_size : int
            The dimension of the input features (per token).
        pos_input_size : int
            The dimension of the input positional features.
        feat_embed_size : int
            The dimension of the projected feature embedding.
        pos_embed_size : int
            The dimension of the projected positional embedding.
        nhead : int
            The number of heads in the multihead attention modules.
        num_encoder_layers : int
            The number of sub-encoder-layers in the encoder.
        sum_features : bool, optional
            Whether to sum the features along the sequence dimension. Default: False
        dim_feedforward : int
            The dimension of the feedforward network model.
        activation_fn : callable, optional
            The activation function to use for the embedding layer. Default: None
        """
        super().__init__()
        self.feat_embed_size = feat_embed_size
        self.pos_embed_size = pos_embed_size
        self.d_model = feat_embed_size + pos_embed_size
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.activation_fn = activation_fn

        encoder_layer = TransformerEncoderLayer(
            d_model=self.d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            batch_first=True)
        self.transformer_encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers)
        self.feat_embedding_layer = nn.Linear(feat_input_size, feat_embed_size)
        self.pos_embedding_layer = nn.Linear(pos_input_size, pos_embed_size)

    def forward(self, x: Tensor, pos: Tensor, padding_mask: Optional[Tensor] = None) -> Tensor:
        """
        x: (batch, seq, feat_input_size)
        pos: (batch, seq, pos_input_size)
        padding_mask: (batch, seq) boolean mask where True indicates padding
        """
        x = self.feat_embedding_layer(x)
        pos = self.pos_embedding_layer(pos)
        src = torch.cat((x, pos), dim=-1)
        output = self.transformer_encoder(src, src_key_padding_mask=padding_mask)

        # NOTE: dimension only works when batch_first=True
        if padding_mask is None:
            output = output.sum(dim=1)
        else:
            if not self.training:
                # apply correct padding mask for evaluation
                # this happens because with both self.eval() and torch.no_grad()
                # the transformer encoder may shorten the output length to
                # the max non-padded length in the batch
                max_seq_len = output.shape[1]
                padding_mask = padding_mask[:, :max_seq_len]
            output = output.masked_fill(padding_mask.unsqueeze(-1), 0)
            output = output.sum(dim=1)

        return output


class TransformerEmbedding(nn.Module):
    """ Transformer + MLP embedding model """

    def __init__(
        self, transformer_args: Dict, mlp_args: Dict):
        """
        Parameters
        ----------
        transformer_args : dict
            The arguments for the Transformer model.
        mlp_args : dict
            The arguments for the MLP model.
        """
        super().__init__()
        self.transformer_args = transformer_args
        self.mlp_args = mlp_args
        self.transformer = Transformer(
            feat_input_size=transformer_args['feat_input_size'],
            pos_input_size=transformer_args['pos_input_size'],
            feat_embed_size=transformer_args.get('feat_embed_size', 32),
            pos_embed_size=transformer_args.get('pos_embed_size', 32),
            nhead=transformer_args.get('nhead', 4),
            num_encoder_layers=transformer_args.get('num_encoder_layers', 4),
            dim_feedforward=transformer_args.get('dim_feedforward', 128),
            sum_features=transformer_args.get('sum_features', False),
            activation_name=transformer_args.get('activation_name', 'ReLU'),
            activation_args=transformer_args.get('activation_args', None),
        )
        self.mlp = mlp.MLPBatchNorm(
            input_size=self.transformer.d_model,
            output_size=mlp_args['output_size'],
            hidden_sizes=mlp_args.get('hidden_sizes', []),
            activation_fn=utils.get_activation(
                mlp_args.get('activation_name', 'relu'),
                mlp_args.get('activation_args', {})
            ),
            batch_norm=mlp_args.get('batch_norm', False),
            dropout=mlp_args.get('dropout', 0.0)
        )

    def forward(self, batch_dict):
        """
        Run a forward pass through the model's transformer backbone followed by the MLP head.

        Parameters
        ----------
        batch_dict : dict
            A dictionary containing the inputs required for the forward pass:
            - 'x' (torch.Tensor): Input feature tensor, typically shaped (batch_size, seq_len, feature_dim).
            - 't' (torch.Tensor): Positional encodings or time indices; shape must be compatible with the transformer's positional input.
            - 'padding_mask' (Optional[torch.Tensor]): Optional boolean or byte mask of shape (batch_size, seq_len) indicating padding positions to be ignored by the transformer. If omitted or None, no padding is applied.

        Returns
        -------
        torch.Tensor
            The output embeddings of shape (batch_size, output_size)
        """
        x = batch_dict['x']
        t = batch_dict['t']
        padding_mask = batch_dict.get('padding_mask', None)
        x = self.transformer(x, t, padding_mask)
        x = self.mlp(x)
        return x
