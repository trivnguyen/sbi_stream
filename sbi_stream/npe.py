
import torch
import torch.nn as nn
import pytorch_lightning as pl

from sbi_stream import flows_utils, training_utils
from sbi_stream.models.transformer import TransformerEmbedding
from sbi_stream.models.gnn import GNNEmbedding

class NPE(pl.LightningModule):
    def __init__(
        self, model_args, optimizer_args=None, scheduler_args=None, norm_dict=None,
        freeze_components=None
        """
        Parameters
        ----------
        model_args : dict
            Arguments for the model. Can include 'batch_prep_args' for custom
            batch preparation settings.
        optimizer_args : dict, optional
            Arguments for the optimizer. Default: None
        scheduler_args : dict, optional
            Arguments for the scheduler. Default: None
        norm_dict : dict, optional
            The normalization dictionary. For bookkeeping purposes only.
            Default: None
        freeze_components : list, optional
            List of model components to freeze during training. Options arez
            'embedding_model' and 'flows'. Default: None
        """
        super().__init__()
        self.model_args = model_args
        self.optimizer_args = optimizer_args or {}
        self.scheduler_args = scheduler_args or {}
        self.norm_dict = norm_dict
        self.freeze_components = freeze_components or []
        self.batch_prep_args = model_args.get('batch_prep_args', {})
        self.save_hyperparameters()
        self._setup_model()

    def _freeze_component(self, component):
        """ Freeze the parameters of a model component. """
        for param in component.parameters():
            param.requires_grad = False

    def _setup_model(self):
        """ Initialize the model components. """
        embedding_type = self.model_args.get('embedding_type')
        embedding_args = self.model_args.get('embedding_args')
        flows_args = self.model_args.get('flows_args')

        # Create the embedding model
        if embedding_type == 'transformer':
            self.embedding_model = TransformerEmbedding(
                transformer_args=embedding_args['transformer_args'],
                mlp_args=embedding_args['mlp_args']
            )
        elif embedding_type == 'gnn':
            self.embedding_model = GNNEmbedding(
                gnn_args=embedding_args['gnn_args'],
                mlp_args=embedding_args['mlp_args']
            )
        else:
            raise ValueError(f'Unknown embedding type: {embedding_type}')

        # Create the normalizing flows
        self.flows = flows_utils.build_flows(
            features=flows_args['features'],
            hidden_features=flows_args['hidden_sizes'],
            context_features=flows_args['context_size'],
            num_transforms=flows_args['num_transforms'],
            num_bins=flows_args['num_bins'],
            activation=flows_args.get('activation', 'relu'),
            activation_args=flows_args.get('activation_args', None),
        )

        # Freeze the components if specified
        for name, component in zip(
            ['embedding_model', 'flows'], [self.embedding_model, self.flows]
        ):
            is_frozen = name in self.freeze_components
            if is_frozen:
                self._freeze_component(component)

            # Verify that the freezing was successful
            for param in component.parameters():
                assert param.requires_grad == (not is_frozen), (
                    f'Parameter in {name} has requires_grad={param.requires_grad}, '
                    f'expected {not is_frozen}'
                )

    def forward(self, batch_dict):
        return self.embedding_model(batch_dict)

    def training_step(self, batch, batch_idx):
        batch_dict = training_utils.prepare_batch(
            batch, self.model_args['embedding_type'], self.device,
            batch_prep_args=self.batch_prep_args)
        context = self.forward(batch_dict)
        log_prob = self.flows(context).log_prob(batch_dict['y'])
        loss = -log_prob.mean()
        self.log(
            'train_loss', loss, on_step=True, on_epoch=True,
            prog_bar=True, batch_size=batch_dict['batch_size'])
        return loss

    def validation_step(self, batch, batch_idx):
        batch_dict = training_utils.prepare_batch(
            batch, self.model_args['embedding_type'], self.device,
            batch_prep_args=self.batch_prep_args)
        context = self.forward(batch_dict)
        log_prob = self.flows(context).log_prob(batch_dict['y'])
        loss = -log_prob.mean()
        self.log(
            'val_loss', loss, on_step=False, on_epoch=True,
            prog_bar=True, batch_size=batch_dict['batch_size'])
        return loss

    def configure_optimizers(self):
        """ Initialize optimizer and LR scheduler """
        return training_utils.configure_optimizers(
            self.parameters(), self.optimizer_args, self.scheduler_args)
