'''Autoregressive GPT-like sequence classifiers.'''

from collections.abc import Sequence
from numbers import Number

import torch
from transformers import GPT2Model

from ...layers import ActivType, DenseBlock
from .base import BaseClassif


class DistilGPT2Classif(BaseClassif):
    '''
    GPT-like sequence classifier with custom head.

    Parameters
    ----------
    num_labels : int
        Number of labels.
    label_names : list or tuple
        Label names.
    class_weights : list, tuple or tensor
        Class weights.
    num_hidden : int, list or None
        Number of hidden units.
    activation : str or None
        Nonlinearity type.
    drop_rate : float or None
        Dropout probability.
    pad_token_id: int or None
        Pad token ID.

    '''

    model_name = 'distilbert/distilgpt2'

    def __init__(
        self,
        num_labels: int,
        label_names: Sequence[str] | None = None,
        class_weights: Sequence[float] | torch.Tensor | None = None,
        num_hidden: int | Sequence[int] | None = None,
        activation: ActivType | None = 'leaky_relu',
        drop_rate: float | None = None,
        pad_token_id: int | None = None
    ):

        # call base class init
        super().__init__(
            num_labels=num_labels,
            label_names=label_names,
            class_weights=class_weights
        )

        # create feature extractor
        self.base_model = GPT2Model.from_pretrained(
            self.model_name
        )

        # set pad token (for batch sizes larger than one)
        if pad_token_id is not None:
            if self.base_model.config.pad_token_id is None:
                self.base_model.config.pad_token_id = self.base_model.config.eos_token_id
            elif self.base_model.config.pad_token_id != pad_token_id:
                raise ValueError('Pad token ID mismatch')

        # create classification head
        if num_hidden is None:
            num_hidden = []

        elif isinstance(num_hidden, Number):
            num_hidden = [num_hidden]

        if isinstance(num_hidden, Sequence):
            num_features = [
                self.embed_dim,  # number of inputs
                *num_hidden,  # number of hidden units
                num_labels if num_labels > 2 else 1  # number of outputs
            ]
        else:
            raise TypeError(f'Invalid type: {type(num_hidden)}')

        self.classif_head = DenseBlock(
            num_features=num_features,
            activation=activation,
            last_activation=None,
            batchnorm=False,
            normalize_last=False,
            drop_rate=drop_rate
        )

        # freeze/unfreeze parameters
        for p in self.base_model.parameters():
            p.requires_grad = False

        for p in self.classif_head.parameters():
            p.requires_grad = True

    @property
    def embed_dim(self) -> int:
        '''Get embedding dimensionality.'''
        return self.base_model.embed_dim  # self.base_model.config.n_embd

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:

        # check for pad token and batch size
        if self.base_model.config.pad_token_id is None and len(input_ids) > 1:
            raise RuntimeError('Pad token required for batch sizes larger than one')

        # compute embedding
        base_out = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        last_hidden_state = base_out.last_hidden_state  # (batch, sequence, features)

        # get last token (which is not padded)
        if self.base_model.config.pad_token_id is None:
            last_token = last_hidden_state[:, -1]  # (batch, features)

        else:
            first_padded_ids = (input_ids == self.base_model.config.pad_token_id).int().argmax(-1)
            last_non_padded_ids = first_padded_ids - 1  # note that 0 becomes -1

            last_token = last_hidden_state[
                torch.arange(len(last_hidden_state), device=last_hidden_state.device),
                last_non_padded_ids
            ]  # (batch, features)

        # compute logits
        logits = self.classif_head(last_token)  # (batch, labels)

        if labels is None:
            return logits

        # compute loss
        else:
            # for a single output unit, this uses nn.BCEWithLogitsLoss
            if logits.shape[-1] == 1:
                loss = self.criterion(logits.squeeze(-1), labels.to(logits.dtype))
            # for multiple output units, nn.CrossEntropyLoss is used
            else:
                loss = self.criterion(logits, labels)

            return loss, logits  # this is compatible with transformers.Trainer
