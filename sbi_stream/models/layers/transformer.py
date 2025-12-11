
from typing import Optional, Callable, Dict

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch import Tensor


class MultiHeadAttentionBlock(nn.Module):
    """Multi-head attention block with pre-layer normalization.

    Uses pre-LN configuration (layer normalization within residual stream),
    which tends to provide better training stability than post-LN.

    Parameters
    ----------
    d_in : int
        Input dimension
    d_model : int
        Model embedding dimension
    d_mlp : int
        Hidden dimension of the feed-forward MLP
    n_heads : int
        Number of attention heads
    """
    def __init__(
        self,
        d_in: int,
        d_model: int,
        d_mlp: int,
        n_heads: int
    ):
        super().__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.d_mlp = d_mlp
        self.n_heads = n_heads
        self.ln_attn1 = nn.LayerNorm(d_in)
        self.ln_attn2 = nn.LayerNorm(d_in)
        self.ln_mlp = nn.LayerNorm(d_in)
        self.attention = nn.MultiheadAttention(d_in, n_heads, batch_first=True)
        self.mlp1 = nn.Linear(d_in, d_mlp)
        self.mlp2 = nn.Linear(d_mlp, d_in)
        self.activation = nn.GELU()

        # Initialize attention weights
        nn.init.xavier_uniform_(self.attention.in_proj_weight)
        nn.init.zeros_(self.attention.in_proj_bias)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        conditioning: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Multi-head attention
        if x is y:
            # Self-attention
            x_sa = self.ln_attn1(x)
            x_sa = self.attention(
                x_sa, x_sa, x_sa, key_padding_mask=mask, need_weights=False
            )[0]
        else:
            # Cross-attention
            x_sa = self.ln_attn1(x)
            y_sa = self.ln_attn2(y)
            x_sa = self.attention(
                x_sa, y_sa, y_sa, key_padding_mask=mask, need_weights=False
            )[0]

        # Add to residual stream
        x = x + x_sa

        # Feed-forward MLP with pre-LN
        x_mlp = self.ln_mlp(x)
        x_mlp = self.activation(self.mlp1(x_mlp))
        x_mlp = self.mlp2(x_mlp)

        # Add to residual stream
        x = x + x_mlp
        return x


class TransformerA(nn.Module):
    """ Decoder-only Transformer. This is the original Transformer model used
    in https://arxiv.org/abs/2512.07960v1.

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

    def __init__(
        self, feat_input_size: int, pos_input_size: int, feat_embed_size: int = 32,
        pos_embed_size: int = 32, nhead: int = 4, num_encoder_layers: int = 4,
        dim_feedforward: int = 128, sum_features: bool = False,
        activation_name: str = None, activation_args: Optional[Dict] = None,
    ) -> None:
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


class TransformerB(nn.Module):
    """Decoder-only Transformer. This is an alternative Transformer model, copied
    from https://github.com/trivnguyen/jgnn.

    Parameters
    ----------
    d_in : int
        Number of input (and output) features
    d_model : int
        Dimension of the model embedding space
    d_mlp : int
        Dimension of the feed-forward MLP
    n_layers : int
        Number of transformer layers
    n_heads : int
        Number of attention heads
    d_pos : int, optional
        Dimension of positional encoding features
    d_cond : int, optional
        Dimension of conditioning features
    concat_conditioning : bool
        Whether to concatenate conditioning to the input
    use_pos_enc : bool
        Whether to use positional encoding
    pooling : str, optional
        Type of pooling to aggregate sequence ('mean', 'max', 'sum', 'cls', or None)
        If None, returns full sequence
    """
    def __init__(
        self,
        d_in: int,
        d_model: int = 128,
        d_mlp: int = 512,
        n_layers: int = 4,
        n_heads: int = 4,
        d_pos: Optional[int] = None,
        d_cond: Optional[int] = None,
        concat_conditioning: bool = False,
        use_pos_enc: bool = False,
        pooling: Optional[str] = None
    ):
        super().__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.d_mlp = d_mlp
        self.d_pos = d_pos
        self.d_cond = d_cond
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.use_pos_enc = use_pos_enc
        self.concat_conditioning = concat_conditioning
        self.pooling = pooling

        self._setup_model()

    def _setup_model(self):
        # Input embedding layer
        self.input_embed = nn.Linear(self.d_in, self.d_model)

        # Positional encoding layer
        if self.use_pos_enc and self.d_pos is not None:
            self.pos_encoding_layer = nn.Linear(self.d_pos, self.d_model)
        else:
            self.pos_encoding_layer = None

        # Conditioning layers
        if self.d_cond is not None:
            self.conditioning_embed = nn.Linear(self.d_cond, self.d_model)
            self.conditioning_concat_embed = nn.Linear(
                self.d_model + self.d_in, self.d_model
            )
        else:
            self.conditioning_embed = None
            self.conditioning_concat_embed = None

        # Transformer layers
        self.transformer_layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.transformer_layers.append(
                MultiHeadAttentionBlock(
                    self.d_model, self.d_model, self.d_mlp, self.n_heads
                )
            )

        # Final layer norm and output projection
        self.final_ln = nn.LayerNorm(self.d_model)
        self.unembed = nn.Linear(self.d_model, self.d_in)
        torch.nn.init.zeros_(self.unembed.weight)

    def _pool_sequence(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Pool sequence representations to single vector.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, seq_len, features)
        mask : torch.Tensor, optional
            Shape (batch_size, seq_len), True for padding positions

        Returns
        -------
        torch.Tensor
            Shape (batch_size, features) if pooling is applied, else (batch_size, seq_len, features)
        """
        if self.pooling is None:
            return x
        elif self.pooling == 'mean':
            # Masked mean pooling
            if mask is not None:
                x_masked = x.masked_fill(mask.unsqueeze(-1), 0)
                lengths = (~mask).sum(dim=1, keepdim=True).clamp(min=1)
                return x_masked.sum(dim=1) / lengths.float()
            else:
                return x.mean(dim=1)
        elif self.pooling == 'max':
            # Masked max pooling
            if mask is not None:
                x_masked = x.masked_fill(mask.unsqueeze(-1), float('-inf'))
                return x_masked.max(dim=1)[0]
            else:
                return x.max(dim=1)[0]
        elif self.pooling == 'sum':
            # Masked sum pooling
            if mask is not None:
                x_masked = x.masked_fill(mask.unsqueeze(-1), 0)
                return x_masked.sum(dim=1)
            else:
                return x.sum(dim=1)
        elif self.pooling == 'cls':
            # Use first token (assumes CLS token)
            return x[:, 0]
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

    def forward(
        self,
        x: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        pos_enc: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Input embedding
        x = self.input_embed(x)

        # Add positional encoding
        if pos_enc is not None and self.use_pos_enc:
            pos_enc = self.pos_encoding_layer(pos_enc)
            if mask is not None:
                pos_enc = torch.where(~mask[:, :, None], pos_enc, 0)
            x = x + pos_enc

        # Add conditioning
        if conditioning is not None:
            conditioning = self.conditioning_embed(conditioning)
            if self.concat_conditioning:
                conditioning = conditioning[:, None, :].repeat(1, x.size(1), 1)
                x = torch.cat([x, conditioning], dim=-1)
                x = self.conditioning_concat_embed(x)

        # Apply transformer layers
        for i in range(self.n_layers):
            if conditioning is not None and not self.concat_conditioning:
                x = x + conditioning[:, None, :]
            x = self.transformer_layers[i](x, x, mask, conditioning)

        # Final layer normalization
        x = self.final_ln(x)

        # Output projection
        x = self.unembed(x)

        # Apply pooling if specified
        x = self._pool_sequence(x, mask)

        return x
