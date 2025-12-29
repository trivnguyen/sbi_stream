
import math
import torch
import torch_geometric.transforms as T


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
