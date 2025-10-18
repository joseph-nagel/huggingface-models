'''Lightning for Hugging Face models.'''

from . import base, classif, lr_schedule

from .base import LightningBaseModel

from .classif import LightningImgClassif

from .lr_schedule import make_lr_schedule
