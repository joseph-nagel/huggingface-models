'''Base model.'''

from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch
import torch.nn as nn


class SeqClassifBaseModel(nn.Module, ABC):
    '''
    Base model for transfer learning sequence classification.

    Parameters
    -----------
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
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

