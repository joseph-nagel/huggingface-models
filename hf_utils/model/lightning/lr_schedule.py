'''Learning rate scheduling.'''

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers import (
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup
)


LR_SCHEDULES = [
    'constant',
    'linear',
    'cosine',
    'cosine_with_hard_restarts'
]


def make_lr_schedule(
    optimizer: Optimizer,
    mode: str | None = 'constant',
    num_warmup: int | None = None,
    num_total: int | None = None,
    num_cycles: int | None = None
) -> LRScheduler:
    '''
    Create learning rate scheduler.

    Summary
    -------
    This function creates a learning rate scheduler for an optimizer.
    Constant, linear and cosine schedules with an optional warmup phase are supported.
    The implementations from `transformers.optimization` are used.

    Parameters
    ----------
    optimizer : PyTorch optimizer
        Optimizer to apply the learning rate schedule to.
    mode : str or None
        Learning rate schedule type.
    num_warmup : int or None
        Number of warmup steps.
    num_total : int or None
        Total number of steps (for the linear and cosine schedules).
    num_cycles : int or None
        Number of hard restarts.

    '''

    num_warmup = max(0, num_warmup) if num_warmup is not None else 0
    num_total = max(0, num_total) if num_total is not None else 0
    num_cycles = max(1, num_cycles) if num_cycles is not None else 1

    # set constant mode as default
    if mode is None:
        mode = 'constant'

    # create LR scheduler
    if mode == 'constant':
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup
        )

    elif mode == 'linear':
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup,
            num_training_steps=num_total
        )

    elif mode == 'cosine':
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup,
            num_training_steps=num_total
        )

    elif mode == 'cosine_with_hard_restarts':
        lr_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup,
            num_training_steps=num_total,
            num_cycles=num_cycles
        )

    else:
        raise ValueError(f'Unknown LR schedule type: {mode}')

    return lr_scheduler
