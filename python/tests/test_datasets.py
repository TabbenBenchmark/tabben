from torch.utils.data import DataLoader

from tabben.datasets import OpenTabularDataset


def check_sizes(ds, num_examples, num_features, num_outputs=1):
    # check the input and output arrays themselves
    assert ds.X.size() == (num_examples, num_features)
    assert ds.y.numel() == num_examples * num_outputs
    
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


def test_arcene(tmp_path):
    train_valid_shape = (100, 10_000)
    
    check_split_sizes(tmp_path, 'arcene', {
        'train': train_valid_shape,
        'valid': train_valid_shape,
    })


def test_covertype(tmp_path):
    num_features = 54
    
    check_split_sizes(tmp_path, 'covertype', {
        'train': (11_340, num_features),
        'valid': (3_780, num_features),
        'test': (565_892, num_features),
    })


def test_poker(tmp_path):
    num_features = 10
    
    check_split_sizes(tmp_path, 'poker', {
        'train': (25_010, num_features),
        'test': (1_000_000, num_features),
    })


def test_sarcos(tmp_path):
    num_features = 21
    num_outputs = 7
    
    check_split_sizes(tmp_path, 'sarcos', {
        'train': (44_484, num_features, num_outputs),
        'test' : (4_449, num_features, num_outputs),
    })

