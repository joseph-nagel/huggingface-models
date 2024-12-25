'''Autoregressive GPT-like sequence classifiers.'''

from collections.abc import Sequence
from numbers import Number

import torch
import torch.nn as nn
from transformers import GPT2Model

from .dense import ActivType, DenseBlock


class DistilGPT2SeqClassif(nn.Module):
    '''Sequence classifier with custom head.'''

    def __init__(
        self,
        num_labels: int,
        num_hidden: int | Sequence[int] | None = None,
        activation: ActivType | None = 'leaky_relu',
        drop_rate: float | None = None
    ) -> None:

        super().__init__()

        # create feature extractor
        self.feature_extractor = GPT2Model.from_pretrained(
            'distilgpt2'
        )

        # create classification head
        if num_hidden is None:
            num_hidden = []

        elif isinstance(num_hidden, Number):
            num_hidden = [num_hidden]

        if isinstance(num_hidden, Sequence):
            num_features = [768, *num_hidden, num_labels]
        else:
            raise TypeError(f'Invalid type : {type(num_hidden)}')

        self.classif_head = DenseBlock(
            num_features=num_features,
            batchnorm=False,
            activation=activation,
            last_activation=None,
            normalize_last=False,
            drop_rate=drop_rate
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:

        # extract features
        features_out = self.feature_extractor(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        features = features_out['last_hidden_state'] # (batch, sequence, features)

        # get last token
        last_token_features = features[:, -1] # (batch, features)

        # compute logits
        logits = self.classif_head(last_token_features) # (batch, labels)

        return logits
