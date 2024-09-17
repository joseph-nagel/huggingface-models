'''Datamodules.'''

from torch.utils.data import DataLoader
from torchvision import transforms
from lightning import LightningDataModule
from datasets import load_dataset


class ImageTransform:
    '''Helper class for applying an image transform to a datasets.Dataset.'''

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, batch_dict):

        # apply transform to images only
        if self.transform is not None:
            batch_dict['img'] = [self.transform(img) for img in batch_dict['img']]

        return batch_dict


class CIFAR10DataModule(LightningDataModule):
    '''
    Lightning DataModule for the Hugging Face CIFAR-10 dataset.

    Parameters
    ----------
    data_dir : str
        Directory for storing the data.
    mean : float
        Mean for data normalization.
    std : float, optional
        Standard deviation for normalization.
    random_state : int
        Random generator seed.
    batch_size : int, optional
        Batch size of the data loader.
    num_workers : int, optional
        Number of workers for the loader.

    '''

    def __init__(self,
                 data_dir=None,
                 mean=(0.5, 0.5, 0.5),
                 std=(0.5, 0.5, 0.5),
                 random_state=42,
                 batch_size=32,
                 num_workers=0):

        super().__init__()

        # set data location
        self.data_dir = data_dir

        # set random state
        self.random_state = random_state

        # set loader parameters
        self.batch_size = batch_size
        self.num_workers = num_workers

        # create transforms
        transforms_list = [transforms.ToTensor()]

        if (mean is not None) and (std is not None): # normalize (e.g. scale to [-1, 1])
            normalize_fn = transforms.Normalize(mean=mean, std=std)
            transforms_list.append(normalize_fn)

        self.transform = transforms.Compose(transforms_list)

    @property
    def label_names(self):
        '''Get label names.'''
        if hasattr(self, 'ds'):
            return self.ds['train'].features['label'].names
        else:
            raise AttributeError('Data has not been loaded/initialited yet')

    def prepare_data(self):
        '''Download data.'''

        # initialize a datasets.Dataset
        self.ds = load_dataset(
            'cifar10',
            data_dir=self.data_dir
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

            # set image transformation
            if self.transform is not None:
                train_transform = ImageTransform(self.transform)
                val_transform = ImageTransform(self.transform)

                self.train_ds.set_transform(train_transform)
                self.val_ds.set_transform(val_transform)
            else:
                self.train_ds.with_format('torch')
                self.val_ds.with_format('torch')

        # create test dataset
        elif stage == 'test':
            self.test_ds = self.ds['test']

            # set image transformation
            if self.transform is not None:
                test_transform = ImageTransform(self.transform)
                self.test_ds.set_transform(test_transform)
            else:
                self.test_ds.with_format('torch')

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, # DataLoaders accept datasets.Dataset objects
            batch_size=self.batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.num_workers > 0
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.num_workers > 0
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.num_workers > 0
        )

