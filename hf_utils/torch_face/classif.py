'''Sequence classifiers.'''

import torch
import torch.nn as nn
from transformers import DistilBertModel


class DistilBertClassifier(nn.Module):
    '''Sequence classifier with custom head.'''

    def __init__(
        self,
        num_labels: int
    ) -> None:

        super().__init__()

        # create feature extractor
        self.feature_extractor = DistilBertModel.from_pretrained(
            'distilbert-base-uncased'
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

        classif_token_features = features[:, 0] # (batch, features)

        # compute logits
        logits = self.classif_head(classif_token_features) # (batch, labels)

        return logits

