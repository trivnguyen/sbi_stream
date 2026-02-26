"""CNN-based embedding model with PyTorch Lightning."""

from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import pytorch_lightning as pl

from ..layers import CNN, ResNet, MLP
from ..utils import get_activation, configure_optimizers, build_embedding_loss


class CNNEmbedding(pl.LightningModule):
    """CNN-based embedding model for matched-filter image-like inputs.

    Processes 2D image inputs (e.g. stacked matched-filter histograms) through
    a CNN backbone followed by an MLP projection head to produce fixed-size
    embeddings suitable for SBI.

    Parameters
    ----------
    in_channels : int
        Number of input image channels (e.g. 1 for signal only, 3 for
        signal + LSST bg + Roman bg stacked along the channel dimension).
    cnn_args : dict
        Configuration for the CNN backbone:

        - ``channels`` (list of int): output channels per conv block
        - ``kernel_size`` (int or list): kernel size(s). Default 3.
        - ``stride`` (int or list): stride(s). Default 1.
        - ``downsample`` (bool or list): 2×2 MaxPool after each block. Default False.
        - ``batch_norm`` (bool): BatchNorm2d after each conv. Default True.
        - ``dropout`` (float): Dropout2d probability. Default 0.0.
        - ``pooling_output_size`` (int): AdaptiveAvgPool2d output spatial size. Default 1.
        - ``act_name`` (str): activation function name. Default ``'relu'``.
        - ``act_args`` (dict): kwargs for the activation. Default ``{}``.
        - ``type`` (str): backbone type — ``'cnn'`` (default) or ``'resnet'``.
          ``'resnet'`` accepts the same keys as :class:`ResNet`.

    mlp_args : dict
        Configuration for the MLP projection head:

        - ``output_size`` (int): embedding dimension
        - ``hidden_sizes`` (list of int): hidden layer widths
        - ``act_name`` (str): activation function name
        - ``act_args`` (dict): kwargs for the activation
        - ``batch_norm`` (bool): BatchNorm1d. Default False.
        - ``dropout`` (float): dropout probability. Default 0.0.

    loss_type : str
        ``'mse'`` or ``'vmim'``. Default ``'mse'``.
    loss_args : dict, optional
        Arguments for the loss function (see ``build_embedding_loss``).
    optimizer_args : dict, optional
        Optimizer configuration (name, lr, weight_decay).
    scheduler_args : dict, optional
        Scheduler configuration.
    pre_transforms : callable, optional
        Applied to each raw batch before unpacking.
    norm_dict : dict, optional
        Normalization parameters persisted as a hyperparameter.
    """

    def __init__(
        self,
        in_channels: int,
        cnn_args: Dict[str, Any],
        mlp_args: Dict[str, Any],
        loss_type: str = 'mse',
        loss_args: Optional[Dict[str, Any]] = None,
        optimizer_args: Optional[Dict[str, Any]] = None,
        scheduler_args: Optional[Dict[str, Any]] = None,
        pre_transforms=None,
        norm_dict=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.cnn_args = cnn_args
        self.mlp_args = mlp_args
        self.loss_type = loss_type
        self.loss_args = loss_args or {}
        self.optimizer_args = optimizer_args or {}
        self.scheduler_args = scheduler_args or {}
        self.pre_transforms = pre_transforms
        self.norm_dict = norm_dict
        self.output_size = None  # set in _setup_model
        self.save_hyperparameters(ignore=['pre_transforms'])

        self._setup_model()

    def _setup_model(self):
        """Initialize CNN/ResNet backbone, MLP head, and loss function."""
        cnn_config = dict(self.cnn_args)
        cnn_config['in_channels'] = self.in_channels
        cnn_config['act'] = get_activation(
            cnn_config.pop('act_name', 'relu'),
            cnn_config.pop('act_args', {}),
        )
        backbone_type = cnn_config.pop('type', 'cnn')
        if backbone_type == 'resnet':
            self.cnn = ResNet(**cnn_config)
        else:
            self.cnn = CNN(**cnn_config)

        # Build MLP head — input size comes from the CNN's flattened output
        mlp_config = dict(self.mlp_args)
        mlp_config['input_size'] = self.cnn.output_size
        mlp_config['act'] = get_activation(
            mlp_config.pop('act_name', 'relu'),
            mlp_config.pop('act_args', {}),
        )
        self.mlp = MLP(**mlp_config)
        self.output_size = self.mlp_args['output_size']

        # Build loss
        loss_config = dict(self.loss_args)
        if self.loss_type == 'vmim' and 'context_features' not in loss_config:
            loss_config['context_features'] = self.output_size
        self.loss_fn, self.flow = build_embedding_loss(self.loss_type, loss_config)

    def forward(self, batch_dict: dict) -> torch.Tensor:
        """Forward pass: CNN → flatten → MLP.

        Parameters
        ----------
        batch_dict : dict
            Must contain ``'x'``: image tensor of shape ``(B, C, H, W)``.

        Returns
        -------
        torch.Tensor
            Embedding of shape ``(B, output_size)``.
        """
        features = self.cnn(batch_dict['x'])
        return self.mlp(features)

    def _prepare_batch(self, batch) -> dict:
        """Unpack a dataloader batch into the forward dict.

        Expects either:
        - a tuple/list ``(images, labels)`` (e.g. from ``TensorDataset``), or
        - a dict with keys ``'x'`` and ``'target'``.

        ``images`` should have shape ``(B, C, H, W)``.
        """
        if self.pre_transforms is not None:
            batch = self.pre_transforms(batch)

        if isinstance(batch, (tuple, list)):
            x, y = batch[0], batch[1]
        else:
            x, y = batch['x'], batch['target']

        x = x.to(self.device)
        y = y.to(self.device)
        return {'x': x, 'target': y, 'batch_size': x.size(0)}

    def training_step(self, batch, batch_idx):
        batch_dict = self._prepare_batch(batch)
        embedding = self.forward(batch_dict)
        loss = self.loss_fn(embedding, batch_dict['target'])
        self.log(
            'train/loss', loss, on_step=True, on_epoch=True, prog_bar=True,
            logger=True, batch_size=batch_dict['batch_size'], sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        batch_dict = self._prepare_batch(batch)
        embedding = self.forward(batch_dict)
        loss = self.loss_fn(embedding, batch_dict['target'])
        self.log(
            'val/loss', loss, on_step=False, on_epoch=True, prog_bar=True,
            logger=True, batch_size=batch_dict['batch_size'], sync_dist=True,
        )
        return loss

    def configure_optimizers(self):
        return configure_optimizers(
            self.parameters(), self.optimizer_args, self.scheduler_args
        )
