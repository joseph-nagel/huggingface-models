'''Dense block.'''

from collections.abc import Sequence

import torch.nn as nn

from .utils import ActivType, make_dense


class DenseBlock(nn.Sequential):
    '''
    Block of multiple dense layers.

    Parameters
    ----------
    num_features : list of tuple
        Number of features.
    activation : str or None
        Nonlinearity type.
    last_activation : str or None
        Last nonlinearity type.
    batchnorm : bool
        Determines whether batchnorm is used.
    normalize_last : bool
        Determines whether batchnorm is used last.
    drop_rate : float or None
        Dropout probability.

    '''

    def __init__(
        self,
        num_features: Sequence[int],
        activation: ActivType | None = 'leaky_relu',
        last_activation: ActivType | None = None,
        batchnorm: bool = False,
        normalize_last: bool = True,
        drop_rate: float | None = None
    ) -> None:

        # determine last activation
        if last_activation == 'same':
            last_activation = activation

        # check number of layers
        if len(num_features) >= 2:
            num_layers = len(num_features) - 1
        else:
            raise ValueError('Number of features needs at least two entries')

        # assemble layers
        layers = []

        for idx, (in_features, out_features) in enumerate(zip(num_features[:-1], num_features[1:])):
            is_not_last = (idx < num_layers - 1)

            dense = make_dense(
                in_features,
                out_features,
                activation=activation if is_not_last else last_activation,
                batchnorm=batchnorm if is_not_last else (batchnorm and normalize_last),
                drop_rate=drop_rate
            )

            layers.append(dense)

        # initialize module
        super().__init__(*layers)

