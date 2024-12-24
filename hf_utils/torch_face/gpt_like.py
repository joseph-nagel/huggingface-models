'''GPT-like sequence classifiers.'''

import torch
import torch.nn as nn
from transformers import GPT2Model


class DistilGPT2SeqClassif(nn.Module):
    '''Sequence classifier with custom head.'''

    def __init__(
        self,
        num_labels: int
    ) -> None:

        super().__init__()

        # create feature extractor
        self.feature_extractor = GPT2Model.from_pretrained(
            'distilgpt2'
        )

        # create classification head
        self.classif_head = nn.Linear(768, num_labels)

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

