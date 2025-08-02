'''Base wrapper.'''

import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from transformers import (
    get_cosine_schedule_with_warmup,
    get_constant_schedule_with_warmup
)


class LightningBaseModel(LightningModule):
    '''
    Lightning wrapper for Hugging Face models.

    Parameters
    ----------
    model : Hugging Face model
        Hugging Face transformers model.
    lr : float
        Initial learning rate.
    lr_schedule : {"constant", "cosine"}
        Learning rate schedule type.
    lr_interval : {"epoch", "step"}
        Learning rate update interval.
    lr_warmup : int
        Warmup steps/epochs.

    '''

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-04,
        lr_schedule: str | None = 'constant',
        lr_interval: str = 'epoch',
        lr_warmup: int = 0
    ) -> None:

        super().__init__()

        # set Hugging Face model
        self.model = model

        # set LR params
        self.lr = abs(lr)
        self.lr_schedule = lr_schedule
        self.lr_interval = lr_interval
        self.lr_warmup = abs(int(lr_warmup))

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
        self.log('train_loss', loss.item())  # Lightning logs batch-wise scalars during training per default

        return loss

    def validation_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:

        loss = self.loss(batch)
        self.log('val_loss', loss.item())  # Lightning automatically averages scalars over batches for validation

        return loss

    def test_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:

        loss = self.loss(batch)
        self.log('test_loss', loss.item())  # Lightning automatically averages scalars over batches for testing

        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer | tuple[list, list]:

        # create optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        # if no LR schedule is set, return optimizer only
        if self.lr_schedule is None:
            return optimizer

        # otherwise, create LR schedule
        else:

            # get number of training time units
            if self.lr_interval == 'epoch':
                num_training_steps = self.trainer.max_epochs
            elif self.lr_interval == 'step':
                num_training_steps = self.trainer.estimated_stepping_batches
            else:
                raise ValueError(f'Unknown LR interval: {self.lr_interval}')

            # create LR scheduler
            if self.lr_schedule == "constant":
                lr_scheduler = get_constant_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=self.lr_warmup
                )
            elif self.lr_schedule == "cosine":
                lr_scheduler = get_cosine_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=self.lr_warmup,
                    num_training_steps=num_training_steps
                )
            else:
                raise ValueError(f'Unknown LR schedule type: {self.lr_schedule}')

            # create LR config
            lr_config = {
                'scheduler': lr_scheduler,  # set LR scheduler
                'interval': self.lr_interval,  # set time unit (step or epoch)
                'frequency': 1  # set update frequency
            }

            return [optimizer], [lr_config]

