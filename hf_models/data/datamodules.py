'''Datamodules.'''

from torch.utils.data import DataLoader
from torchvision import transforms
from lightning import LightningDataModule
from datasets import load_dataset

from .transform import DataTransform


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

    def __init__(self,
                 transform=None,
                 train_transform=None,
                 val_transform=None,
                 test_transform=None,
                 batch_size=32,
                 num_workers=0):

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

    def train_dataloader(self):
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

    def val_dataloader(self):
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

    def test_dataloader(self):
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

