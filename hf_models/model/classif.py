'''Image classifier.'''

from pathlib import Path

from transformers import AutoModelForImageClassification

from .base import LightningBaseModel


class LightningImageClassifier(LightningBaseModel):
    '''
    Lightning wrapper for a Hugging Face image classifier.

    Parameters
    ----------
    ckpt_name : str
        Name of the model checkpoint.
    cache_dir : str
        Directory for storing the checkpoint.
    num_labels : int
        Number of target labels.
    lr : float
        Initial optimizer learning rate.

    '''

    def __init__(self,
                 ckpt_name='google/vit-base-patch16-224',
                 cache_dir=None,
                 num_labels=10,
                 lr=1e-04):

        # load pretrained model
        ignore_mismatched_sizes = False if num_labels is None else True

        model = AutoModelForImageClassification.from_pretrained(
            ckpt_name,
            cache_dir=cache_dir,
            num_labels=num_labels,
            ignore_mismatched_sizes=ignore_mismatched_sizes
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
            lr=lr
        )

        # store hyperparams
        self.save_hyperparameters(
            {'cache_dir': str(Path(cache_dir).resolve())}, # store absolute cache path
            logger=True
        )

