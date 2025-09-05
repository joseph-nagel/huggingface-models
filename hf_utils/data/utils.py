'''Data downloading.'''

from datasets import load_dataset, DatasetDict


def load_yelp(
    random_seed: int = 42,
    tiny: bool = False
) -> DatasetDict:
    '''Load Yelp dataset.'''

    # load data
    ds = load_dataset('yelp_review_full')

    # split data
    train_ds = ds['train']

    split_ds = train_ds.train_test_split(
        test_size=len(ds['test']),
        seed=random_seed
    )

    train_ds = split_ds['train']
    val_ds = split_ds['test']

    test_ds = ds['test']

    # return datasets
    if not tiny:
        return DatasetDict({
            'train': train_ds,
            'val': val_ds,
            'test': test_ds
        })

    # return tiny datasets
    else:
        return DatasetDict({
            'train': train_ds.shuffle(seed=23).select(range(100)),
            'val': val_ds.shuffle(seed=23).select(range(20)),
            'test': test_ds.shuffle(seed=23).select(range(20))
        })


def load_imdb(
    random_seed: int = 42,
    tiny: bool = False
) -> DatasetDict:
    '''Load IMDB dataset.'''

    # load data
    ds = load_dataset('imdb')

    # split data
    train_ds = ds['train']

    test_ds = ds['test']

    split_ds = test_ds.train_test_split(
        test_size=10000,
        seed=random_seed
    )

    val_ds = split_ds['train']
    test_ds = split_ds['test']

    # return datasets
    if not tiny:
        return DatasetDict({
            'train': train_ds,
            'val': val_ds,
            'test': test_ds
        })

    # return tiny datasets
    else:
        return DatasetDict({
            'train': train_ds.shuffle(seed=23).select(range(100)),
            'val': val_ds.shuffle(seed=23).select(range(20)),
            'test': test_ds.shuffle(seed=23).select(range(20))
        })
