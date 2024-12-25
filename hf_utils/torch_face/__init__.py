'''PyTorch tools for Hugging Face models.'''

from . import (
    autoreg,
    bidirect,
    dense,
    utils
)

from .autoreg import DistilGPT2SeqClassif

from .bidirect import DistilBertSeqClassif

from .dense import DenseBlock

from .utils import (
    ACTIVATIONS,
    make_activation,
    make_block,
    make_dropout,
    make_dense
)

