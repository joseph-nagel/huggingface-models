'''PEFT tools.'''

from collections.abc import Sequence
from typing import Literal

import torch.nn as nn
from peft import LoraConfig, LoraModel, get_peft_model
from peft.tuners.lora.layer import LoraLayer


def make_lora(
    model: nn.Module,
    rank: int,
    alpha: float | None = None,
    dropout: float | None = None,
    bias: Literal['none', 'all', 'lora_only'] = 'none',
    target_modules: str | Sequence[str] | None = None,
    modules_to_save: Sequence[str] | None = None
):
    '''Integrate LoRA layers into model.'''

    rank = abs(int(rank))
    alpha = abs(float(alpha)) if alpha is not None else 2 * float(rank)
    dropout = abs(float(dropout)) if dropout is not None else 0.0

    # create LoRA config
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias=bias,
        init_lora_weights=True,
        target_modules=target_modules,  # specify layers to apply LoRA (linear, conv, MHA, etc.)
        modules_to_save=modules_to_save  # specify layers to unfreeze and update
    )

    # create LoRA model
    model = get_peft_model(model, config)

    # check LoRA model
    is_lora_model = isinstance(model, LoraModel)
    has_lora_layers = any(isinstance(m, LoraLayer) for m in model.modules())

    if not (is_lora_model and has_lora_layers):
        raise RuntimeError('LoRA model not correctly initialized')

    return model
