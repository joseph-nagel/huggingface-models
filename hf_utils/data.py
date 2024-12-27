'''Data tools.'''

from datasets import load_dataset, DatasetDict


def load_yelp(
    random_seed: int = 42,
    tiny: bool = False
) -> DatasetDict:

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
        tiny_train_ds = train_ds.shuffle(seed=23).select(range(1000))
        tiny_val_ds = val_ds.shuffle(seed=23).select(range(200))
        tiny_test_ds = test_ds.shuffle(seed=23).select(range(200))

        return DatasetDict({
            'train': tiny_train_ds,
            'val': tiny_val_ds,
            'test': tiny_test_ds
        })

