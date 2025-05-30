'''PyTorch and Lightning for Hugging Face models.'''

from . import lightning, torch

from .lightning import (
    LightningBaseModel,
    LightningImgClassif
)

from .torch import (
    BaseClassif,
    DistilBertClassif,
    DistilGPT2Classif
)

