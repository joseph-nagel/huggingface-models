'''Base model.'''

from abc import ABC, abstractmethod
from typing import Any
from collections.abc import Sequence

import torch
import torch.nn as nn


class BaseClassif(nn.Module, ABC):
    '''
    Base model for transfer learning sequence classification.

    Parameters
    ----------
    num_labels : int
        Number of labels.
    label_names : list or tuple
        Label names.
    class_weights : list, tuple or tensor
        Class weights.

    '''

    def __init__(
        self,
        num_labels: int,
        label_names: Sequence[str] | None = None,
        class_weights: Sequence[float] | torch.Tensor | None = None
    ) -> None:

        super().__init__()

        # set class weights
        if class_weights is not None:
            class_weights = torch.as_tensor(class_weights).view(-1)

            if len(class_weights) != num_labels:
                raise ValueError(f'Invalid number of class weights: {len(class_weights)}')

        # create loss function
        if num_labels == 2:
            self.criterion = nn.BCEWithLogitsLoss(
                reduction='mean',
                pos_weight=class_weights
            )

        elif num_labels > 2:
            self.criterion = nn.CrossEntropyLoss(
                reduction='mean',
                weight=class_weights
            )

        else:
            raise ValueError('Invalid number of labels')

        # set label names
        if label_names is None:
            self.label_names = list(range(num_labels))

        elif len(label_names) == num_labels:
            self.label_names = label_names

        else:
            raise ValueError('Number of labels mismatch')

    @property
    def num_labels(self) -> int:
        '''Get number of labels.'''
        return len(self.label_names)

    @property
    def is_binary(self) -> bool:
        '''Check binary classification.'''
        return self.num_labels == 2

    @property
    def is_multiclass(self) -> bool:
        '''Check multiclass classification.'''
        return self.num_labels > 2

    @property
    def id2label(self) -> dict[int, str]:
        '''Get idx-to-label dict.'''
        return {idx: label for idx, label in enumerate(self.label_names)}

    @property
    def label2id(self) -> dict[str, int]:
        '''Get label-to-idx dict.'''
        return {label: idx for idx, label in enumerate(self.label_names)}

    @property
    @abstractmethod
    def embed_dim(self) -> int:
        '''Get embedding dimensionality.'''
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Any
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
