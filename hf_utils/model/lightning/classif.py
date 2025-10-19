'''Image classifier.'''

from typing import Any
from pathlib import Path

import torch
from torchmetrics.classification import Accuracy
from transformers import AutoModelForImageClassification

from .base import LightningBaseModel


class LightningImageClassifier(LightningBaseModel):
    '''
    Lightning wrapper for a Hugging Face image classifier.

    Parameters
    ----------
    model_name : str
        Name of the model checkpoint.
    data_dir : str or None
        Directory for storing the checkpoint.
    num_labels : int or None
        Number of target labels.
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
        model_name: str = 'google/vit-base-patch16-224',
        data_dir: str | None = None,
        num_labels: int | None = None,
        lr: float = 1e-04,
        lr_schedule: str | None = 'constant',
        lr_interval: str | None = 'epoch',
        lr_warmup: int | None = 0,
        lr_cycles: int | None = 1
    ) -> None:

        # load pretrained model
        further_opts = {} if num_labels is None else {'num_labels': num_labels, 'ignore_mismatched_sizes': True}

        model = AutoModelForImageClassification.from_pretrained(
            model_name,
            cache_dir=data_dir,
            **further_opts
            # num_labels=num_labels,  # this argument should not be None
            # ignore_mismatched_sizes=False if num_labels is None else True
        )

        model = model.eval()

        # freeze/unfreeze parameters
        for p in model.parameters():
            p.requires_grad = False

        for p in model.classifier.parameters():
            p.requires_grad = True

        # initialize parent class
        super().__init__(
            model=model,
            lr=lr,
            lr_schedule=lr_schedule,
            lr_interval=lr_interval,
            lr_warmup=lr_warmup,
            lr_cycles=lr_cycles
        )

        # store hyperparams
        if data_dir is not None:
            abs_data_dir = str(Path(data_dir).resolve())

            self.save_hyperparameters(
                {'data_dir': abs_data_dir},  # store absolute custom cache path for later re-import
                logger=True
            )
        else:
            self.save_hyperparameters(logger=True)

        # create accuracy metrics
        num_labels = num_labels if num_labels is not None else model.config.num_labels

        self.train_acc = Accuracy(task='multiclass', num_classes=num_labels)
        self.val_acc = Accuracy(task='multiclass', num_classes=num_labels)
        self.test_acc = Accuracy(task='multiclass', num_classes=num_labels)

    def loss(
        self,
        return_logits: bool = False,
        *args: torch.Tensor,
        **kwargs: Any
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        '''Compute loss (and optionally return logits).'''

        outputs = self.model(*args, **kwargs)

        if return_logits:
            return outputs.loss, outputs.logits
        else:
            return outputs.loss

    def training_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:

        loss, logits = self.loss(return_logits=True, **batch)

        _ = self.train_acc(logits, batch['labels'])

        self.log('train_loss', loss.item())  # Lightning logs batch-wise scalars during training per default
        self.log('train_acc', self.train_acc)  # the same applies to torchmetrics.Metric objects

        return loss

    def validation_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:

        loss, logits = self.loss(return_logits=True, **batch)

        _ = self.val_acc(logits, batch['labels'])

        self.log('val_loss', loss.item())  # Lightning automatically averages scalars over batches for validation
        self.log('val_acc', self.val_acc)  # the batch size is considered when logging torchmetrics.Metric objects

        return loss

    def test_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:

        loss, logits = self.loss(return_logits=True, **batch)

        _ = self.test_acc(logits, batch['labels'])

        self.log('test_loss', loss.item())  # Lightning automatically averages scalars over batches for testing
        self.log('test_acc', self.test_acc)  # the batch size is considered when logging torchmetrics.Metric objects

        return loss
