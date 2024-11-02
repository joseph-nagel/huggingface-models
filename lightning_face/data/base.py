'''Base datamodule.'''

from collections.abc import Callable

from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from lightning import LightningDataModule


class BaseDataModule(LightningDataModule):
    '''
    Lightning base DataModule for Hugging Face datasets.

    Parameters
    ----------
    transform : None or callable
        Image transformation for all splits.
    train_transform : None or callable
        Image transformation for the train set.
    val_transform : None or callable
        Image transformation for the val. set.
    test_transform : None or callable
        Image transformation for the test set.
    batch_size : int
        Batch size of the data loader.
    num_workers : int
        Number of workers for the loader.

    '''

    def __init__(
        self,
        transform: Callable[[Image.Image], torch.tensor] | None = None,
        train_transform: Callable[[Image.Image], torch.tensor] | None = None,
        val_transform: Callable[[Image.Image], torch.tensor] | None = None,
        test_transform: Callable[[Image.Image], torch.tensor] | None = None,
        batch_size: int = 32,
        num_workers: int  = 0
    ) -> None:

        super().__init__()

        # set loader parameters
        self.batch_size = batch_size
        self.num_workers = num_workers

        # define datasets as None
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

        # set transforms
        specific_transforms = (train_transform, val_transform, test_transform)

        all_spec_transforms_are_none = all([t is None for t in specific_transforms])
        no_spec_transform_is_none = all([t is not None for t in specific_transforms])

        # if no transform is passed, initialize the universal one with a default
        if (transform is None) and all_spec_transforms_are_none:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.5, 0.5, 0.5),
                    std=(0.5, 0.5, 0.5)
                )
            ])

        # if a universal transform is passed, set all specific ones accordingly
        if (transform is not None) and all_spec_transforms_are_none:
            self.train_transform = transform
            self.val_transform = transform
            self.test_transform = transform

        # set specific transforms individually
        elif (transform is None) and no_spec_transform_is_none:
            self.train_transform = train_transform
            self.val_transform = val_transform
            self.test_transform = test_transform

        else:
            raise ValueError('Invalid combination of transforms')

    def train_dataloader(self) -> DataLoader:
        if hasattr(self, 'train_ds') and self.train_ds is not None:
            return DataLoader(
                self.train_ds, # DataLoaders accept datasets.Dataset objects
                batch_size=self.batch_size,
                drop_last=True,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self.num_workers > 0
            )
        else:
            raise AttributeError('Train set has not been set')

    def val_dataloader(self) -> DataLoader:
        if hasattr(self, 'val_ds') and self.val_ds is not None:
            return DataLoader(
                self.val_ds,
                batch_size=self.batch_size,
                drop_last=False,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.num_workers > 0
            )
        else:
            raise AttributeError('Val. set has not been set')

    def test_dataloader(self) -> DataLoader:
        if hasattr(self, 'test_ds') and self.test_ds is not None:
            return DataLoader(
                self.test_ds,
                batch_size=self.batch_size,
                drop_last=False,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.num_workers > 0
            )
        else:
            raise AttributeError('Test set has not been set')

