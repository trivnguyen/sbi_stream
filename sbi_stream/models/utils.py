
from typing import Optional, Dict, Callable

import torch.nn as nn

def get_activation(name: str, args: Optional[Dict] = None) -> Callable:
    """Get an activation callable (class or factory). If args is provided, returns a
    callable that will instantiate the activation with those kwargs (using functools.partial)."""

    key = name.lower().replace('-', '').replace('_', '')
    mapping = {
        'identity': nn.Identity,
        'relu': nn.ReLU,
        'tanh': nn.Tanh,
        'sigmoid': nn.Sigmoid,
        'gelu': nn.GELU,
        'silu': nn.SiLU,
        'leakyrelu': nn.LeakyReLU,
        'elu': nn.ELU,
        'prelu': nn.PReLU,
        'selu': nn.SELU,
    }

    act_cls = mapping.get(key)
    if act_cls is None:
        raise ValueError(f'Unknown activation function: {name}')

    return act_cls(**args) if args else act_cls()
