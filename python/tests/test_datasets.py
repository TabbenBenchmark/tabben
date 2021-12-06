from torch.utils.data import DataLoader

from tabben.datasets import OpenTabularDataset, validate_dataset_file


def check_sizes(ds, num_examples, num_features, num_outputs=1):
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


def test_adult(tmp_path):
    num_features = 14
    train_examples = 32_561
    test_examples = 16_281
    
    check_split_sizes(tmp_path, 'adult', {
        'train': (train_examples, num_features),
        'test': (test_examples, num_features),
    })
    
    check_attributes(tmp_path, 'adult', {
        'task': 'classification',
        'num_classes': 2,
    })
    
    validate_dataset_file(tmp_path / 'adult.npz')


def test_arcene(tmp_path):
    train_test_shape = (100, 10_000)
    
    check_split_sizes(tmp_path, 'arcene', {
        'train': train_test_shape,
        'test': train_test_shape,  # arcene's "official" validation set used as test set
    })
    
    check_attributes(tmp_path, 'arcene', {
        'task': 'classification',
        'num_classes': 2,
    })
    
    validate_dataset_file(tmp_path / 'arcene.npz')


def test_covertype(tmp_path):
    num_features = 54
    
    check_split_sizes(tmp_path, 'covertype', {
        'train': (11_340, num_features),
        'valid': (3_780, num_features),
        'test': (565_892, num_features),
    })

    check_attributes(tmp_path, 'covertype', {
        'task': 'classification',
        'num_classes': 7,
    })

    validate_dataset_file(tmp_path / 'covertype.npz')


# test for higgs not included here because of its large size

def test_poker(tmp_path):
    num_features = 10
    
    check_split_sizes(tmp_path, 'poker', {
        'train': (25_010, num_features),
        'test': (1_000_000, num_features),
    })

    check_attributes(tmp_path, 'poker', {
        'task': 'classification',
        'num_classes': 10,
    })

    validate_dataset_file(tmp_path / 'poker.npz')


def test_sarcos(tmp_path):
    num_features = 21
    num_outputs = 7
    
    check_split_sizes(tmp_path, 'sarcos', {
        'train': (44_484, num_features, num_outputs),
        'test': (4_449, num_features, num_outputs),
    })

    check_attributes(tmp_path, 'sarcos', {
        'task': 'regression',
        'num_outputs': 7
    })

    validate_dataset_file(tmp_path / 'sarcos.npz')

