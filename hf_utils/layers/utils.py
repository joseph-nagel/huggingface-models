'''PyTorch utilities.'''

from typing import Any
from collections.abc import Sequence
from inspect import isclass

import torch.nn as nn


ActivType = str | type[nn.Module]


ACTIVATIONS = {
    'identity': nn.Identity,
    'sigmoid': nn.Sigmoid,
    'tanh': nn.Tanh,
    'relu': nn.ReLU,
    'leaky_relu': nn.LeakyReLU,
    'elu': nn.ELU,
    'softplus': nn.Softplus,
    'swish': nn.SiLU
}


def make_activation(mode: ActivType | None = 'leaky_relu', **kwargs: Any) -> nn.Module | None:
    '''Create activation function.'''
    if mode is None:
        activ = None
    elif isclass(mode):
        activ = mode(**kwargs)
    elif isinstance(mode, str) and mode in ACTIVATIONS.keys():
        activ = ACTIVATIONS[mode](**kwargs)
    else:
        raise ValueError(f'Unknown activation: {mode}')
    return activ


def make_block(layers: nn.Module | Sequence[nn.Module | None]) -> nn.Module:
    '''Assemble a block of layers.'''

    if isinstance(layers, nn.Module):
        block = layers

    elif isinstance(layers, (list, tuple)):
        not_none_layers = [l for l in layers if l is not None]
        if len(not_none_layers) == 0:
            raise ValueError('No layers to assemble')
        elif len(not_none_layers) == 1:
            block = not_none_layers[0]
        else:
            block = nn.Sequential(*not_none_layers)

    else:
        raise TypeError(f'Invalid layers type: {type(layers)}')

    return block


def make_dropout(drop_rate: float | None = None) -> nn.Module | None:
    '''Create a dropout layer.'''
    if drop_rate is None:
        dropout = None
    else:
        dropout = nn.Dropout(p=drop_rate)
    return dropout


def make_dense(
    in_features: int,
    out_features: int,
    bias: bool = True,
    activation: ActivType | None = None,
    batchnorm: bool = False,
    drop_rate: float | None = None
) -> nn.Module:
    '''
    Create fully connected layer.

    Parameters
    ----------
    in_features : int
        Number of inputs.
    out_features : int
        Number of outputs.
    bias : bool
        Determines whether a bias is used.
    activation : str or None
        Nonlinearity type.
    batchnorm : bool
        Determines whether batchnorm is used.
    drop_rate : float or None
        Dropout probability.

    '''

    # create dropout layer
    dropout = make_dropout(drop_rate=drop_rate)

    # create dense layer
    linear = nn.Linear(
        in_features,
        out_features,
        bias=bias  # the bias should be disabled if a batchnorm directly follows after the linear layer
    )

    # create activation function
    activ = make_activation(activation)

    # create normalization
    norm = nn.BatchNorm1d(out_features) if batchnorm else None

    # assemble block
    layers = [dropout, linear, activ, norm]  # note that the normalization follows the activation (which could be reversed of course)
    dense_block = make_block(layers)

    return dense_block
