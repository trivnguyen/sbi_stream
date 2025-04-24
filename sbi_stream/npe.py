
import torch
import torch.nn as nn
import pytorch_lightning as pl

from sbi.utils import (
    clamp_and_warn,
    repeat_rows,
)
from sbi.utils.torchutils import assert_all_finite
from sbi.neural_nets.estimators.shape_handling import (
    reshape_to_batch_event,
    reshape_to_sample_batch_event,
)

from snpe_stream import flows_utils
from snpe_stream import models, models_utils


class NPE(pl.LightningModule):
    def __init__(
        self,
        featurizer_args,
        flows_args,
        mlp_args=None,
        optimizer_args=None,
        scheduler_args=None,
        d_time=1,
        d_time_projection=16,
        norm_dict=None,
        num_atoms=10,
        use_atomic_loss=False
    ):
        """
        Parameters
        ----------
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
        num_atoms: int
            Number of atoms for atomic proposal posterior
        """
        super().__init__()
        self.featurizer_args = featurizer_args
        self.flows_args = flows_args
        self.mlp_args = mlp_args or {}
        self.optimizer_args = optimizer_args or {}
        self.scheduler_args = scheduler_args or {}
        self.norm_dict = norm_dict
        self.num_atoms = num_atoms
        self.batch_first = True # always True
        self.prior = None
        self.round = None
        self.use_atomic_loss = use_atomic_loss
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
            raise ValueError(f'Featurizer {featurizer_name} not supported')

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
            features=self.flows_args.features,
            hidden_features=self.flows_args.hidden_sizes,
            context_features=flows_context_features,
            num_transforms=self.flows_args.num_transforms,
            num_bins=self.flows_args.num_bins,
            activation=activation_fn
        )
    def set_prior(self, prior):
        self.prior = prior

    def set_round(self, round):
        if (self.round == 0) or (not self.use_atomic_loss):
            print('Using normal NPE loss')
        else:
            print(f'Using atomic loss with {self.num_atoms}')
        self.round = round

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

    def _log_prob_proposal_posterior_atomic(self, x, theta):
        """ Calculate the log atomic proposal posterior, function taken from SBI package """

        batch_size = theta.shape[0]
        num_atoms = int(clamp_and_warn(
            "num_atoms", self.num_atoms, min_val=2, max_val=batch_size)
        )

        # Each set of parameter atoms is evaluated using the same x,
        # so we repeat rows of the data x, e.g. [1, 2] -> [1, 1, 2, 2]
        repeated_x = repeat_rows(x, num_atoms)

        # To generate the full set of atoms for a given item in the batch,
        # we sample without replacement num_atoms - 1 times from the rest
        # of the theta in the batch.
        probs = torch.ones(batch_size, batch_size) * (1 - torch.eye(batch_size)) / (batch_size - 1)

        choices = torch.multinomial(probs, num_samples=num_atoms - 1, replacement=False)
        contrasting_theta = theta[choices]

        # We can now create our sets of atoms from the contrasting parameter sets
        # we have generated.
        atomic_theta = torch.cat((theta[:, None, :], contrasting_theta), dim=1).reshape(
            batch_size * num_atoms, -1
        )

        # Get (batch_size * num_atoms) log prob prior evals.
        log_prob_prior = self.prior.log_prob(atomic_theta)
        log_prob_prior = log_prob_prior.reshape(batch_size, num_atoms)
        assert_all_finite(log_prob_prior, "prior eval")

        log_prob_posterior = self.flows(repeated_x).log_prob(atomic_theta)
        log_prob_posterior = log_prob_posterior.reshape(batch_size, num_atoms)
        assert_all_finite(log_prob_posterior, "posterior eval")

        # Evaluate large batch giving (batch_size * num_atoms) log prob posterior evals.
        atomic_theta = reshape_to_sample_batch_event(atomic_theta, atomic_theta.shape[1:])
        repeated_x = reshape_to_batch_event(repeated_x, repeated_x.shape[1:])

        unnormalized_log_prob = log_prob_posterior - log_prob_prior

        # Normalize proposal posterior across discrete set of atoms.
        log_prob_proposal_posterior = unnormalized_log_prob[:, 0] - torch.logsumexp(
            unnormalized_log_prob, dim=-1
        )
        assert_all_finite(log_prob_proposal_posterior, "proposal posterior eval")

        return log_prob_proposal_posterior

    def _log_prob_posterior(self, x, theta):
        log_prob_posterior = self.flows(x).log_prob(theta)
        return log_prob_posterior

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

        conditions = self(x, t, padding_mask=padding_mask)
        if (self.round == 0) or (not self.use_atomic_loss):
            log_prob = self._log_prob_posterior(conditions, y)
        else:
            assert 1==2
            log_prob = self._log_prob_proposal_posterior_atomic(conditions, y)
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

        conditions = self(x, t, padding_mask=padding_mask)
        if (self.round == 0) or (not self.use_atomic_loss):
            log_prob = self._log_prob_posterior(conditions, y)
        else:
            log_prob = self._log_prob_proposal_posterior_atomic(conditions, y)
        loss = -log_prob.mean()
        self.log(
            'val_loss', loss, on_step=True, on_epoch=True,
            prog_bar=True, batch_size=len(x))
        return loss

    def configure_optimizers(self):
        """ Initialize optimizer and LR scheduler """
        return models_utils.configure_optimizers(
            self.parameters(), self.optimizer_args, self.scheduler_args)


class SimpleNPE(pl.LightningModule):
    def __init__(
        self,
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
        flows_args : dict
            Arguments for the normalizing flow
        optimizer_args : dict, optional
            Arguments for the optimizer. Default: None
        scheduler_args : dict, optional
            Arguments for the scheduler. Default: None
        norm_dict : dict, optional
            The normalization dictionary. For bookkeeping purposes only.
            Default: None
        num_atoms: int
            Number of atoms for atomic proposal posterior
        """
        super().__init__()
        self.flows_args = flows_args
        self.optimizer_args = optimizer_args or {}
        self.scheduler_args = scheduler_args or {}
        self.norm_dict = norm_dict
        self.batch_first = True # always True
        self.save_hyperparameters()

        self._setup_model()

    def _setup_model(self):
        # create the flows
        activation_fn = models_utils.get_activation_zuko(
            self.flows_args.activation)
        self.flows = flows_utils.build_flows(
            features=self.flows_args.features,
            hidden_features=self.flows_args.hidden_sizes,
            context_features=self.flows_args.context_features,
            num_transforms=self.flows_args.num_transforms,
            num_bins=self.flows_args.num_bins,
            activation=activation_fn
        )

    def _prepare_training_batch(self, batch):
        """ Prepare the batch for training. """
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        # return a dictionary of the inputs
        return_dict = {
            'x': x,
            'y': y,
        }
        return return_dict

    def forward(self, x):
        return x

    def training_step(self, batch, batch_idx):
        batch_dict = self._prepare_training_batch(batch)
        x = batch_dict['x']
        y = batch_dict['y']
        log_prob = self.flows(x).log_prob(y)
        loss = -log_prob.mean()
        self.log(
            'train_loss', loss, on_step=True, on_epoch=True,
            prog_bar=True, batch_size=len(x))
        return loss

    def validation_step(self, batch, batch_idx):
        batch_dict = self._prepare_training_batch(batch)
        x = batch_dict['x']
        y = batch_dict['y']
        log_prob = self.flows(x).log_prob(y)
        loss = -log_prob.mean()
        self.log(
            'val_loss', loss, on_step=True, on_epoch=True,
            prog_bar=True, batch_size=len(x))
        return loss

    def configure_optimizers(self):
        """ Initialize optimizer and LR scheduler """
        return models_utils.configure_optimizers(
            self.parameters(), self.optimizer_args, self.scheduler_args)
