'''Image classifier.'''

from pathlib import Path

from torchmetrics.classification import Accuracy
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
        super().__init__(model=model, lr=lr)

        # store hyperparams
        abs_cache_dir = str(Path(cache_dir).resolve())

        self.save_hyperparameters(
            {'cache_dir': abs_cache_dir}, # store absolute cache path for later re-import
            logger=True
        )

        # create accuracy metrics
        self.train_acc = Accuracy(task='multiclass', num_classes=num_labels)
        self.val_acc = Accuracy(task='multiclass', num_classes=num_labels)
        self.test_acc = Accuracy(task='multiclass', num_classes=num_labels)

    def loss(self, batch, return_logits=False):
        '''Compute loss (and return logits).'''
        outputs = self.model(**batch)

        if not return_logits:
            return outputs['loss']
        else:
            return outputs['loss'], outputs['logits']

    def training_step(self, batch, batch_idx):
        loss, logits = self.loss(batch, return_logits=True)

        _ = self.train_acc(logits, batch['labels'])

        self.log('train_loss', loss.item()) # Lightning logs batch-wise scalars during training per default
        self.log('train_acc', self.train_acc) # the same applies to torchmetrics.Metric objects
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits = self.loss(batch, return_logits=True)

        _ = self.val_acc(logits, batch['labels'])

        self.log('val_loss', loss.item()) # Lightning automatically averages scalars over batches for validation
        self.log('val_acc', self.val_acc) # the batch size is considered when logging torchmetrics.Metric objects
        return loss

    def test_step(self, batch, batch_idx):
        loss, logits = self.loss(batch, return_logits=True)

        _ = self.test_acc(logits, batch['labels'])

        self.log('test_loss', loss.item()) # Lightning automatically averages scalars over batches for testing
        self.log('test_acc', self.test_acc) # the batch size is considered when logging torchmetrics.Metric objects
        return loss
