'''CIFAR datamodules.'''

from torchvision import transforms
from datasets import load_dataset

from .base import BaseDataModule
from .transform import DataTransform


class CIFAR10DataModule(BaseDataModule):
    '''
    Lightning DataModule for the Hugging Face CIFAR-10 dataset.

    Parameters
    ----------
    cache_dir : str
        Directory for storing the data.
    img_size : int or (int, int)
        Target image size.
    img_mean : float
        Mean for data normalization.
    img_std : float
        Standard deviation for normalization.
    random_state : int
        Random generator seed.
    batch_size : int
        Batch size of the data loader.
    num_workers : int
        Number of workers for the loader.

    '''

    def __init__(self,
                 cache_dir=None,
                 img_size=224,
                 img_mean=(0.5, 0.5, 0.5),
                 img_std=(0.5, 0.5, 0.5),
                 random_state=42,
                 batch_size=32,
                 num_workers=0):

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

        # set data location
        self.cache_dir = cache_dir

        # set random state
        self.random_state = random_state

    @property
    def label_names(self):
        '''Get label names.'''
        if hasattr(self, 'ds'):
            return self.ds['train'].features['label'].names
        else:
            raise AttributeError('Data has not been loaded/initialized yet')

    @property
    def id2label(self):
        '''Get idx-to-label dict.'''
        return {idx: label for idx, label in enumerate(self.label_names)}

    @property
    def label2id(self):
        '''Get label-to-idx dict.'''
        return {label: idx for idx, label in enumerate(self.label_names)}

    def prepare_data(self):
        '''Download data.'''

        # initialize a datasets.Dataset
        self.ds = load_dataset(
            'cifar10',
            cache_dir=self.cache_dir
        )

    def setup(self, stage):
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

