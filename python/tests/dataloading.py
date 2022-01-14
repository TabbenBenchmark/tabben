from torch.utils.data import DataLoader

from tabben.datasets import OpenTabularDataset


################################################################################
# Common assertions for datasets
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
