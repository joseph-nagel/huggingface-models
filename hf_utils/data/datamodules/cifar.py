'''CIFAR datamodules.'''

import torch
from torchvision import transforms
from datasets import load_dataset

from .base import BaseDataModule
from .transform import DataTransform


# define type aliases
IntOrInts = int | tuple[int, int]
FloatOrFloats = float | tuple[float, float, float]


class CIFAR10DataModule(BaseDataModule):
    '''
    Lightning DataModule for the Hugging Face CIFAR-10 dataset.

    Parameters
    ----------
    data_dir : str or None
        Directory for storing the data.
    img_size : int or (int, int)
        Target image size.
    img_mean : float or (float, float, float)
        Mean for data normalization.
    img_std : float or (float, float, float)
        Standard deviation for normalization.
    tiny : bool
        Determines whether a tiny version of
        the dataset is used for debugging.
    random_state : int
        Random generator seed.
    batch_size : int
        Batch size of the data loader.
    num_workers : int
        Number of workers for the loader.

    '''

    def __init__(
        self,
        data_dir: str | None = None,
        img_size: IntOrInts = (224, 224),
        img_mean: FloatOrFloats = (0.5, 0.5, 0.5),
        img_std: FloatOrFloats = (0.5, 0.5, 0.5),
        random_state: int = 42,
        tiny: bool = False,
        batch_size: int = 32,
        num_workers: int = 0
    ) -> None:

        # create transforms
        train_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=img_mean, std=img_std)
        ])

        val_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=img_mean, std=img_std)
        ])

        test_transform = val_transform

        # call base class init
        super().__init__(
            transform=None,
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            batch_size=batch_size,
            num_workers=num_workers
        )

        # create inverse normalization
        img_mean = torch.as_tensor(img_mean).view(-1, 1, 1)
        img_std = torch.as_tensor(img_std).view(-1, 1, 1)

        self.renormalize = transforms.Compose([
            transforms.Lambda(lambda x: x * img_std + img_mean),  # reverse normalization
            transforms.Lambda(lambda x: x.clamp(0, 1))  # clip to valid range
        ])

        # set data location
        self.data_dir = data_dir

        # set tiny flag
        self.tiny = tiny

        # set random state
        self.random_state = random_state

    @property
    def label_names(self) -> list[str]:
        '''Get label names.'''

        if hasattr(self, 'ds'):
            return self.ds['train'].features['label'].names
        else:
            raise AttributeError('Data has not been loaded/initialized yet')

    @property
    def id2label(self) -> dict[int, str]:
        '''Get idx-to-label dict.'''
        return {idx: label for idx, label in enumerate(self.label_names)}

    @property
    def label2id(self) -> dict[str, int]:
        '''Get label-to-idx dict.'''
        return {label: idx for idx, label in enumerate(self.label_names)}

    def prepare_data(self) -> None:
        '''Download data.'''

        # initialize a datasets.Dataset
        if not self.tiny:
            self.ds = load_dataset(
                'cifar10',
                cache_dir=self.data_dir
            )

        # use a minimal version of the data
        else:
            self.ds = load_dataset(
                'cifar10',
                cache_dir=self.data_dir,
                split='train[:200]'
            ).train_test_split(0.2, seed=0)

    def setup(self, stage: str) -> None:
        '''Set up train/test/val. datasets.'''

        # create train/val. datasets
        if stage in ('fit', 'validate'):
            train_ds = self.ds['train']

            split_ds = train_ds.train_test_split(
                test_size=0.2,
                seed=self.random_state
            )

            self.train_ds = split_ds['train']
            self.val_ds = split_ds['test']

            # set image transformations
            self.train_ds.set_transform(DataTransform(self.train_transform))
            self.val_ds.set_transform(DataTransform(self.val_transform))

        # create test dataset
        elif stage == 'test':
            self.test_ds = self.ds['test']

            # set image transformation
            self.test_ds.set_transform(DataTransform(self.test_transform))
