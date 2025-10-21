'''Lightning image classifier with LoRA.'''

from collections.abc import Sequence
from typing import Literal

from ..peft import make_lora
from .classif import LightningHFImageClassif


class LightningHFImageClassifLoRA(LightningHFImageClassif):
    '''
    Lightning wrapper for Hugging Face image classifiers.

    Parameters
    ----------
    model_name : str
        Name of the model checkpoint.
    data_dir : str or None
        Directory for storing the HF checkpoint.
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
    freeze_backbone: bool
        Determines whether backbone is frozen.
    lora_rank : int
        LoRA rank.
    lora_alpha : float or None
        LoRA weighting parameter.
    lora_dropout : float or None
        LoRA dropout rate.
    lora_bias : {'none', 'all', 'lora_only'}
        Determines where a bias is used.
    lora_target_modules: str, Sequence[str] or None
        Modules to apply LoRA to.

    '''

    def __init__(
        self,
        model_name: str = 'facebook/dinov2-small-imagenet1k-1-layer',
        data_dir: str | None = None,
        num_labels: int | None = None,
        lr: float = 1e-04,
        lr_schedule: str | None = 'constant',
        lr_interval: str | None = 'epoch',
        lr_warmup: int | None = None,
        lr_cycles: int | None = None,
        freeze_backbone : bool = True,
        lora_rank: int = 8,
        lora_alpha: float | None = None,
        lora_dropout: float | None = None,
        lora_bias: Literal['none', 'all', 'lora_only'] = 'none',
        lora_target_modules: str | Sequence[str] | None = None
    ) -> None:

        # initialize parent class
        super().__init__(
            model_name=model_name,
            data_dir=data_dir,
            num_labels=num_labels,
            lr=lr,
            lr_schedule=lr_schedule,
            lr_interval=lr_interval,
            lr_warmup=lr_warmup,
            lr_cycles=lr_cycles,
            freeze_backbone=freeze_backbone or (lora_rank > 0)
        )

        # enable LoRA
        if lora_rank > 0:
            self.model = make_lora(
                self.model,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
                bias=lora_bias,
                target_modules=lora_target_modules,  # specify layers to apply LoRA (linear, conv, MHA, etc.)
                modules_to_save=['classifier']  # specify layers to unfreeze and update
            )

        # store hyperparams
        self.save_hyperparameters(logger=True)
