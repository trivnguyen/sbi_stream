
import torch
import torch.nn as nn
import pytorch_lightning as pl

from models.zuko import flows_utils
from models import models, models_utils

class Regressor(pl.LightningModule):
    def __init__(
        self,
        output_size,
        featurizer_args,
        flows_args,
        mlp_args=None,
        optimizer_args=None,
        scheduler_args=None,
        d_time=1,
        d_time_projection=16,
        norm_dict=None,
    ):
        """
        Parameters
        ----------
        output_size: int
            The size of the output
        featurizer_args : dict
            Arguments for the featurizer
        flows_args : dict
            Arguments for the normalizing flow
        mlp_args: dict, optional
            Arguments for the MLP after the featurizer
        optimizer_args : dict, optional
            Arguments for the optimizer. Default: None
        scheduler_args : dict, optional
            Arguments for the scheduler. Default: None
        norm_dict : dict, optional
            The normalization dictionary. For bookkeeping purposes only.
            Default: None
        """
        super().__init__()
        self.output_size = output_size
        self.featurizer_args = featurizer_args
        self.flows_args = flows_args
        self.mlp_args = mlp_args or {}
        self.optimizer_args = optimizer_args or {}
        self.scheduler_args = scheduler_args or {}
        self.norm_dict = norm_dict
        self.batch_first = True # always True
        self.save_hyperparameters()

        self._setup_model()

    def _setup_model(self):

        # create the featurizer
        if self.featurizer_args.name == 'transformer':
            activation_fn = models_utils.get_activation(
                self.featurizer_args.activation)
            self.featurizer = models.TransformerFeaturizer(
                d_feat_in=self.featurizer_args.d_feat_in,
                d_time_in=self.featurizer_args.d_time_in,
                d_feat=self.featurizer_args.d_feat,
                d_time=self.featurizer_args.d_time,
                nhead=self.featurizer_args.nhead,
                num_encoder_layers=self.featurizer_args.num_encoder_layers,
                dim_feedforward=self.featurizer_args.dim_feedforward,
                batch_first=self.batch_first,
                activation_fn=activation_fn,
            )
        else:
            raise ValueError(
                f'Featurizer {featurizer_name} not supported')

        # create the MLP
        if len(self.mlp_args) > 0:
            activation_fn = models_utils.get_activation(self.mlp_args.activation)
            self.mlp = models.MLP(
                input_size=self.featurizer.d_model,
                hidden_sizes=self.mlp_args.hidden_sizes,
                activation_fn=activation_fn,
                batch_norm=self.mlp_args.batch_norm,
                dropout=self.mlp_args.dropout,
            )
            flows_context_features = self.mlp_args.hidden_sizes[-1]
        else:
            self.mlp = None
            flows_context_features = self.featurizer.d_model

        # create the flows
        activation_fn = models_utils.get_activation_zuko(
            self.flows_args.activation)
        self.flows = flows_utils.build_flows(
            features=self.output_size,
            hidden_features=self.flows_args.hidden_sizes,
            context_features=flows_context_features,
            num_transforms=self.flows_args.num_transforms,
            num_bins=self.flows_args.num_bins,
            activation=activation_fn
        )


    def _prepare_training_batch(self, batch):
        """ Prepare the batch for training. """
        x, y, t, padding_mask = batch
        x = x.to(self.device)
        y = y.to(self.device)
        t = t.to(self.device)
        padding_mask = padding_mask.to(self.device)

        # return a dictionary of the inputs
        return_dict = {
            'x': x,
            'y': y,
            't': t,
            'padding_mask': padding_mask,
        }
        return return_dict

    def forward(self, x, t,  padding_mask=None):
        x = self.featurizer(x, t, padding_mask=padding_mask)
        if self.mlp is not None:
            x = self.mlp(x)
        return x

    def training_step(self, batch, batch_idx):
        batch_dict = self._prepare_training_batch(batch)
        x = batch_dict['x']
        y = batch_dict['y']
        t = batch_dict['t']
        padding_mask = batch_dict['padding_mask']

        flow_context = self(x, t, padding_mask=padding_mask)
        log_prob = self.flows(flow_context).log_prob(y)
        loss = -log_prob.mean()

        self.log(
            'train_loss', loss, on_step=True, on_epoch=True,
            prog_bar=True, batch_size=len(x))
        return loss

    def validation_step(self, batch, batch_idx):
        batch_dict = self._prepare_training_batch(batch)
        x = batch_dict['x']
        y = batch_dict['y']
        t = batch_dict['t']
        padding_mask = batch_dict['padding_mask']

        flow_context = self(x, t, padding_mask=padding_mask)
        log_prob = self.flows(flow_context).log_prob(y)
        loss = -log_prob.mean()

        self.log(
            'val_loss', loss, on_step=True, on_epoch=True,
            prog_bar=True, batch_size=len(x))
        return loss

    def configure_optimizers(self):
        """ Initialize optimizer and LR scheduler """
        return models_utils.configure_optimizers(
            self.parameters(), self.optimizer_args, self.scheduler_args)
