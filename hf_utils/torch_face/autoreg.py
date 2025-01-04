'''Autoregressive GPT-like sequence classifiers.'''

from typing import Any
from collections.abc import Sequence
from numbers import Number

import torch
from transformers import GPT2Model

from .base import SeqClassifBaseModel
from .dense import ActivType, DenseBlock


class DistilGPT2SeqClassif(SeqClassifBaseModel):
    '''
    GPT-like sequence classifier with custom head.

    Parameters
    ----------
    num_labels : int
        Number of labels.
    label_names : list or tuple
        Label names.
    num_hidden : int, list or None
        Number of hidden units.
    activation : str or None
        Nonlinearity type.
    drop_rate : float or None
        Dropout probability.

    '''

    model_name = 'distilbert/distilgpt2'

    def __init__(
        self,
        num_labels: int,
        label_names: Sequence[str] | None = None,
        num_hidden: int | Sequence[int] | None = None,
        activation: ActivType | None = 'leaky_relu',
        drop_rate: float | None = None
    ) -> None:

        # call base class init
        super().__init__(
            num_labels=num_labels,
            label_names=label_names
        )

        # create feature extractor
        self.feature_extractor = GPT2Model.from_pretrained(
            self.model_name
        )

        # create classification head
        if num_hidden is None:
            num_hidden = []

        elif isinstance(num_hidden, Number):
            num_hidden = [num_hidden]

        if isinstance(num_hidden, Sequence):
            num_features = [
                self.embed_dim, # number of inputs
                *num_hidden, # number of hidden units
                num_labels if num_labels > 2 else 1 # number of outputs
            ]
        else:
            raise TypeError(f'Invalid type : {type(num_hidden)}')

        self.classif_head = DenseBlock(
            num_features=num_features,
            activation=activation,
            last_activation=None,
            batchnorm=False,
            normalize_last=False,
            drop_rate=drop_rate
        )

        # freeze/unfreeze parameters
        for p in self.feature_extractor.parameters():
            p.requires_grad = False

        for p in self.classif_head.parameters():
            p.requires_grad = True

    @property
    def embed_dim(self):
        '''Get feature dimensionality.'''
        return self.feature_extractor.embed_dim # self.feature_extractor.config.n_embd

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any
    ) -> torch.Tensor:

        # extract features
        features_out = self.feature_extractor(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

        features = features_out['last_hidden_state'] # (batch, sequence, features)

        # get last token
        last_token_features = features[:, -1] # (batch, features)

        # compute logits
        logits = self.classif_head(last_token_features) # (batch, labels)

        return logits

