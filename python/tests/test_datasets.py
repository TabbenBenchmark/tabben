"""
Test the dataset files themselves and whether they can be loaded properly.
"""

from tabben.datasets import OpenTabularDataset, validate_dataset_file
from torch.utils.data import DataLoader


################################################################################
# Common Utilities
################################################################################

def check_sizes(ds, num_examples, num_features, num_outputs=1):
    assert len(ds) == num_examples
    
    # check the input and output arrays themselves
    X, y = ds.numpy()
    assert X.shape == (num_examples, num_features)
    assert y.size == num_examples * num_outputs
    
    # test sizes for batch size of 1
    dl = DataLoader(ds)
    item = next(iter(dl))
    
    assert len(item) == 2
    assert item[0].numel() == num_features
    
    # test sizes for batch size of 4
    dl = DataLoader(ds, batch_size=4)
    item = next(iter(dl))
    
    assert len(item) == 2
    assert item[0].size() == (4, num_features)
    
    if num_outputs > 1:
        assert item[1].size() == (4, num_outputs)
    else:
        assert item[1].numel() == 4 * num_outputs


def check_split_sizes(directory, ds_name, splits_dict):
    for split, expected_size in splits_dict.items():
        ds = OpenTabularDataset(directory, ds_name, split=split)
        check_sizes(ds, *expected_size)


def check_attributes(directory, ds_name, attributes_dict):
    ds = OpenTabularDataset(directory, ds_name)  # which split doesn't matter here
    for attr_name, attr_value in attributes_dict.items():
        assert getattr(ds, attr_name) == attr_value


################################################################################
# Tests for each individual dataset
################################################################################

def test_adult(tmp_path):
    num_features = 14
    train_examples = 32_561
    test_examples = 16_281
    
    check_split_sizes(
        tmp_path, 'adult', {
            'train': (train_examples, num_features),
            'test': (test_examples, num_features),
        }
    )
    
    check_attributes(
        tmp_path, 'adult', {
            'task': 'classification',
            'num_classes': 2,
        }
    )
    
    validate_dataset_file(tmp_path / 'adult.npz')


def test_amazon(tmp_path):
    num_features = 9
    train_examples = 26_215
    test_examples = 6_554
    
    check_split_sizes(
        tmp_path, 'amazon', {
            'train': (train_examples, num_features),
            'test': (test_examples, num_features),
        }
    )
    
    check_attributes(
        tmp_path, 'amazon', {
            'task': 'classification',
            'num_classes': 2,
        }
    )
    
    validate_dataset_file(tmp_path / 'amazon.npz')


def test_arcene(tmp_path):
    train_test_shape = (100, 10_000)
    
    check_split_sizes(
        tmp_path, 'arcene', {
            'train': train_test_shape,
            'test': train_test_shape,  # arcene's "official" validation set used as test set
        }
    )
    
    check_attributes(
        tmp_path, 'arcene', {
            'task': 'classification',
            'num_classes': 2,
        }
    )
    
    validate_dataset_file(tmp_path / 'arcene.npz')


def test_covertype(tmp_path):
    num_features = 54
    
    check_split_sizes(
        tmp_path, 'covertype', {
            'train': (11_340, num_features),
            'valid': (3_780, num_features),
            'test': (565_892, num_features),
        }
    )
    
    check_attributes(
        tmp_path, 'covertype', {
            'task': 'classification',
            'num_classes': 7,
        }
    )
    
    validate_dataset_file(tmp_path / 'covertype.npz')


def test_duolingo_original(tmp_path):
    num_features = 10
    
    check_split_sizes(
        tmp_path, 'duolingo-original', {
            'train': (10_275_881, num_features),
            'test': (2_578_345, num_features),
        }
    )
    
    check_attributes(
        tmp_path, 'duolingo-original', {
            'task': 'regression',
        }
    )
    
    validate_dataset_file(tmp_path / 'duolingo-original.npz')


# test for higgs not included here because of its large size

def test_musk(tmp_path):
    num_features = 166
    train_examples = 5548
    test_examples = 1050
    
    check_split_sizes(
        tmp_path, 'musk', {
            'train': (train_examples, num_features),
            'test': (test_examples, num_features),
        }, )
    
    check_attributes(
        tmp_path, 'musk', {
            'task': 'classification',
            'num_outputs': 1,
            'num_classes': 2,
        }
    )

    validate_dataset_file(tmp_path / 'musk.npz')


def test_parkinsons(tmp_path):
    num_features = 16
    num_outputs = 2
    train_examples = 4646
    test_examples = 1229
    
    check_split_sizes(
        tmp_path, 'parkinsons', {
            'train': (train_examples, num_features, num_outputs),
            'test': (test_examples, num_features, num_outputs),
        }, )
    
    check_attributes(
        tmp_path, 'parkinsons', {
            'task': 'regression',
            'num_outputs': 2,
        }
    )

    validate_dataset_file(tmp_path / 'parkinsons.npz')


def test_poker(tmp_path):
    num_features = 10
    
    check_split_sizes(
        tmp_path, 'poker', {
            'train': (25_010, num_features),
            'test': (1_000_000, num_features),
        }
    )
    
    check_attributes(
        tmp_path, 'poker', {
            'task': 'classification',
            'num_classes': 10,
        }
    )
    
    validate_dataset_file(tmp_path / 'poker.npz')


def test_rossman(tmp_path):
    num_features = 18
    train_examples = 814_688
    test_examples = 202_521
    
    check_split_sizes(
        tmp_path, 'rossman', {
            'train': (train_examples, num_features),
            'test': (test_examples, num_features),
        }
    )
    
    check_attributes(
        tmp_path, 'rossman', {
            'task': 'regression',
        }
    )
    
    validate_dataset_file(tmp_path / 'rossman.npz')
