'''Base wrapper.'''

import torch
import torch.nn as nn
from lightning.pytorch import LightningModule


class LightningBaseModel(LightningModule):
    '''
    Lightning wrapper for Hugging Face models.

    Parameters
    ----------
    model : Hugging Face model
        Hugging Face transformers model.
    lr : float
        Initial optimizer learning rate.

    '''

    def __init__(self, model: nn.Module, lr: float = 1e-04) -> None:

        super().__init__()

        # set Hugging Face model
        self.model = model

        # set initial learning rate
        self.lr = abs(lr)

        # store hyperparams
        self.save_hyperparameters(
            ignore=['model'],
            logger=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Run the model.'''
        outputs = self.model(x)
        return outputs['logits']

    def loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        '''Compute the loss.'''
        outputs = self.model(**batch)
        return outputs['loss']

    def training_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:

        loss = self.loss(batch)
        self.log('train_loss', loss.item()) # Lightning logs batch-wise scalars during training per default

        return loss

    def validation_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:

        loss = self.loss(batch)
        self.log('val_loss', loss.item()) # Lightning automatically averages scalars over batches for validation

        return loss

    def test_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:

        loss = self.loss(batch)
        self.log('test_loss', loss.item()) # Lightning automatically averages scalars over batches for testing

        return loss

    # TODO: enable LR scheduling
    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

