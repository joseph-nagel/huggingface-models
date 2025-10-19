'''Base wrapper.'''

from typing import Any

import torch
from lightning.pytorch import LightningModule
from transformers import PreTrainedModel

from .lr_schedule import make_lr_schedule


class LightningBaseModel(LightningModule):
    '''
    Lightning wrapper for Hugging Face models.

    Parameters
    ----------
    model : Hugging Face model
        Hugging Face transformers model.
    lr : float
        Initial learning rate.
    lr_schedule : str or None
        Learning rate schedule type.
    lr_interval : {'epoch', 'step'} or None
        Learning rate update interval.
    lr_warmup : int or None
        Number of warmup steps/epochs.
    lr_cycles : int or None
        Number of hard restarts.

    '''

    def __init__(
        self,
        model: PreTrainedModel,
        lr: float = 1e-04,
        lr_schedule: str | None = 'constant',
        lr_interval: str | None = 'epoch',
        lr_warmup: int | None = 0,
        lr_cycles: int | None = 1
    ) -> None:

        super().__init__()

        # set Hugging Face model
        if not isinstance(model, PreTrainedModel):
            raise TypeError(f'Invalid model type: {type(model)}')

        model = model.train()
        self.model = model

        # set LR params
        self.lr = abs(lr)
        self.lr_schedule = lr_schedule
        self.lr_interval = lr_interval
        self.lr_warmup = abs(int(lr_warmup)) if lr_warmup is not None else None
        self.lr_cycles = abs(int(lr_cycles)) if lr_cycles is not None else None

        # store hyperparams
        self.save_hyperparameters(
            ignore=['model'],
            logger=True
        )

    def forward(self, *args: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        '''Run the model.'''
        outputs = self.model(*args, **kwargs)
        return outputs.logits

    def loss(self, *args: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        '''Compute the loss.'''
        outputs = self.model(*args, **kwargs)
        return outputs.loss

    def training_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:

        loss = self.loss(**batch)
        self.log('train_loss', loss.item())  # Lightning logs batch-wise scalars during training per default

        return loss

    def validation_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:

        loss = self.loss(**batch)
        self.log('val_loss', loss.item())  # Lightning automatically averages scalars over batches for validation

        return loss

    def test_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:

        loss = self.loss(**batch)
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
                num_total = self.trainer.max_epochs
            elif self.lr_interval == 'step':
                num_total = self.trainer.estimated_stepping_batches
            else:
                raise ValueError(f'Unknown LR interval: {self.lr_interval}')

            # create LR scheduler
            lr_scheduler = make_lr_schedule(
                optimizer=optimizer,
                mode=self.lr_schedule,
                num_total=num_total,
                num_warmup=self.lr_warmup,
                num_cycles=self.lr_cycles
            )

            # create LR config
            lr_config = {
                'scheduler': lr_scheduler,  # set LR scheduler
                'interval': self.lr_interval,  # set time unit (step or epoch)
                'frequency': 1  # set update frequency
            }

            return [optimizer], [lr_config]
