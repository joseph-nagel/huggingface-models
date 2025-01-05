'''Base model.'''

from abc import ABC, abstractmethod
from typing import Any
from collections.abc import Sequence

import torch
import torch.nn as nn


class SeqClassifBaseModel(nn.Module, ABC):
    '''
    Base model for transfer learning sequence classification.

    Parameters
    ----------
    num_labels : int
        Number of labels.
    label_names : list or tuple
        Label names.

    '''

    def __init__(
        self,
        num_labels: int,
        label_names: Sequence[str] | None = None
    ) -> None:

        super().__init__()

        # create loss function
        if num_labels == 2:
            self.criterion = nn.BCEWithLogitsLoss(reduction='mean')

        elif num_labels > 2:
            self.criterion = nn.CrossEntropyLoss(reduction='mean')

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
    def id2label(self) -> dict[int, str]:
        '''Get idx-to-label dict.'''
        return {idx: label for idx, label in enumerate(self.label_names)}

    @property
    def label2id(self) -> dict[str, int]:
        '''Get label-to-idx dict.'''
        return {label: idx for idx, label in enumerate(self.label_names)}

    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any
    ) -> torch.Tensor:
        raise NotImplementedError

