
import functools
from typing import Dict, List, Optional, Tuple, Callable

import torch
import torch.nn as nn

import zuko
from zuko.flows import (
    Flow,
    MaskedAutoregressiveTransform,
    UnconditionalTransform,
)
from zuko.distributions import DiagNormal


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

    if args:
        return functools.partial(act_cls, **args)
    return act_cls

def build_flows(
    features: int, context_features: int, num_transforms: int,
    hidden_features: List[int], num_bins: int, activation: str,
    activation_args: Optional[Dict] = None, randperm: bool = True
):
    """ Build neural spline flow

    Parameters
    ----------
    features : int
        Number of features
    context_features : int
        Number of context features
    num_transforms : int
        Number of flow transforms
    hidden_features : List[int]
        Number of hidden features of the MLP
    num_bins : int
        Number of bins of the spline
    activation : Callable
        Activation function of the MLP
    randperm : bool
        Whether to apply random permutation to the features
    """
    transforms = []
    for i in range(num_transforms):
        order = torch.arange(features)
        if randperm:
            order = order[torch.randperm(order.size(0))]
        shapes = ([num_bins], [num_bins], [num_bins - 1])
        activation_fn = get_activation(activation, activation_args)
        transform = zuko.flows.MaskedAutoregressiveTransform(
            features=features, context=context_features,
            univariate=zuko.transforms.MonotonicRQSTransform,
            shapes=shapes, hidden_features=hidden_features, order=order,
            activation=activation_fn,
        )
        transforms.append(transform)

    flow = zuko.flows.Flow(
        transform=transforms,
        base=UnconditionalTransform(
            DiagNormal, torch.zeros(features), torch.ones(features), buffer=True)
    )
    return flow
