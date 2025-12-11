
import math
import torch
import torch_geometric.transforms as T

class WarmUpCosineAnnealingLR(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, decay_steps, warmup_steps, eta_min=0, last_epoch=-1, restart=False):
        self.decay_steps = decay_steps
        self.warmup_steps = warmup_steps
        self.eta_min = eta_min
        self.restart = restart
        super().__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if (step >= self.decay_steps):
            if self.restart:
                step = step % self.decay_steps
            else:
                step = self.decay_steps
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return self.eta_min + (
            0.5 * (1 + math.cos(math.pi * (step - self.warmup_steps) / (self.decay_steps - self.warmup_steps))))


def configure_optimizers(parameters, optimizer_args, scheduler_args=None):
    """ Return optimizer and scheduler for Pytorch Lightning """
    scheduler_args = scheduler_args or {}

    # setup the optimizer
    if optimizer_args.name == "Adam":
        optimizer = torch.optim.Adam(
            parameters, lr=optimizer_args.lr,
            weight_decay=optimizer_args.weight_decay)
    elif optimizer_args.name == "AdamW":
        optimizer = torch.optim.AdamW(
            parameters, lr=optimizer_args.lr,
            weight_decay=optimizer_args.weight_decay)
    else:
        raise NotImplementedError(
            "Optimizer {} not implemented".format(optimizer_args.name))

    # setup the scheduler
    if scheduler_args.name is None:
        scheduler = None
    elif scheduler_args.name == 'ReduceLROnPlateau':
        scheduler =  torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', factor=scheduler_args.factor,
            patience=scheduler_args.patience)
    elif scheduler_args.name == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=scheduler_args.T_max,
            eta_min=scheduler_args.eta_min)
    elif scheduler_args.name == 'WarmUpCosineAnnealingLR':
        scheduler = WarmUpCosineAnnealingLR(
            optimizer,
            decay_steps=scheduler_args.decay_steps,
            warmup_steps=scheduler_args.warmup_steps,
            eta_min=scheduler_args.eta_min,
            restart=scheduler_args.get('restart', False))
    else:
        raise NotImplementedError(
            "Scheduler {} not implemented".format(scheduler_args.name))

    if scheduler is None:
        return optimizer
    else:
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'train_loss',
                'interval': scheduler_args.interval,
                'frequency': 1
            }
        }

def prepare_batch_transformer(batch, device='cpu', batch_prep_args=None):
    """ Prepare batch for transformer model

    Parameters
    ----------
    batch : tuple
        Input batch containing (x, y, t, padding_mask)
    device : str, default='cpu'
        Device to use
    batch_prep_args : dict, optional
        Additional arguments for batch preparation. Currently unused for transformer.
    """
    batch_prep_args = batch_prep_args or {}

    x, y, t, padding_mask = batch
    x = x.to(device)
    y = y.to(device)
    t = t.to(device)
    padding_mask = padding_mask.to(device)

    return {
        'x': x,
        'y': y,
        't': t,
        'padding_mask': padding_mask,
        'batch_size': x.size(0)
    }

def prepare_batch_gnn(batch, device='cpu', batch_prep_args=None):
    """ Prepare batch for graph model

    Parameters
    ----------
    batch : torch_geometric.data.Batch
        Input batch
    device : str, default='cpu'
        Device to use
    batch_prep_args : dict, optional
        Additional arguments for batch preparation:
        - k : int, default=10
            Number of nearest neighbors for KNN graph
        - loop : bool, default=False
            Include self-loops in graph
        - transform : callable, optional
            Custom graph transformation. If provided, k and loop are ignored.
    """
    batch_prep_args = batch_prep_args or {}

    # Allow custom transform or use default KNN graph construction
    if 'transform' in batch_prep_args:
        transform = batch_prep_args['transform']
    else:
        k = batch_prep_args.get('k', 10)
        loop = batch_prep_args.get('loop', False)
        transform = T.Compose([
            T.ToDevice('cpu'),   # remove if torch_cluster is compiled with GPU support
            T.KNNGraph(k=k, loop=loop),
            T.ToDevice(device)
        ])

    batch = transform(batch)
    return {
        'x': batch.x,
        'y': batch.y,
        'edge_index': batch.edge_index,
        'edge_attr': batch.edge_attr,
        'edge_weight': batch.edge_weight,
        'batch': batch.batch,
        'batch_size': len(batch),
    }

def prepare_batch_cnn(batch, device='cpu', batch_prep_args=None):
    """ Prepare batch for CNN model

    Parameters
    ----------
    batch : tuple or torch.Tensor
        Input batch
    device : str, default='cpu'
        Device to use
    batch_prep_args : dict, optional
        Additional arguments for batch preparation (e.g., image transforms).
    """
    batch_prep_args = batch_prep_args or {}
    raise NotImplementedError("CNN batch preparation not implemented yet.")

def prepare_batch(batch, embedding_type, device='cpu', batch_prep_args=None):
    """ Prepare batch for training

    Parameters
    ----------
    batch : tuple or torch_geometric.data.Batch
        Input batch
    embedding_type : str
        Type of embedding model ('transformer', 'gnn', or 'cnn')
    device : str, default='cpu'
        Device to use
    batch_prep_args : dict, optional
        Additional arguments for batch preparation. The available options
        depend on the embedding_type. See individual prepare_batch_* functions
        for details.
    """
    batch_prep_args = batch_prep_args or {}

    if embedding_type == 'transformer':
        batch_dict = prepare_batch_transformer(
            batch, device=device, batch_prep_args=batch_prep_args)
    elif embedding_type == 'gnn':
        batch_dict = prepare_batch_gnn(
            batch, device=device, batch_prep_args=batch_prep_args)
    elif embedding_type == 'cnn':
        batch_dict = prepare_batch_cnn(
            batch, device=device, batch_prep_args=batch_prep_args)

    return batch_dict
