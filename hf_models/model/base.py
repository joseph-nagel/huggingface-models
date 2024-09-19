'''Base wrapper.'''

import torch
from lightning.pytorch import LightningModule


class LightningBaseModel(LightningModule):
    '''Lightning wrapper for Hugging Face models.'''

    def __init__(self, model, lr=1e-04):

        super().__init__()

        # set Hugging Face model
        self.model = model

        # set initial learning rate
        self.lr = abs(lr)

        # store hyperparams
        self.save_hyperparameters(
            ignore=['model'],
            logger=True
        )

    def forward(self, x):
        '''Run the model.'''
        outputs = self.model(x)
        return outputs.logits

    def loss(self, batch):
        '''Compute the loss.'''
        outputs = self.model(**batch)
        return outputs.loss

    def training_step(self, batch, batch_idx):
        loss = self.loss(batch)
        self.log('train_loss', loss.item()) # Lightning logs batch-wise scalars during training per default
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.loss(batch)
        self.log('val_loss', loss.item()) # Lightning automatically averages scalars over batches for validation
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.loss(batch)
        self.log('test_loss', loss.item()) # Lightning automatically averages scalars over batches for testing
        return loss

    # TODO: enable LR scheduling
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

