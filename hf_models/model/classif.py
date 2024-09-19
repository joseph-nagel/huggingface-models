'''Image classifier.'''

from transformers import AutoModelForImageClassification

from .base import LightningBaseModel


class LightningImageClassifier(LightningBaseModel):
    '''Lightning wrapper for a Hugging Face image classifier.'''

    def __init__(self,
                 ckpt_name='google/vit-base-patch16-224',
                 num_labels=10,
                 lr=1e-04):

        # load pretrained model
        ignore_mismatched_sizes = False if num_labels is None else True

        model = AutoModelForImageClassification.from_pretrained(
            ckpt_name,
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
        self.save_hyperparameters(logger=True)

