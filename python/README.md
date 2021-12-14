# A Python package for the Open Benchmark for Tabular Data

[![Package Unit Tests (Latest)](https://github.com/umd-otb/OpenTabularDataBenchmark/actions/workflows/test.yml/badge.svg)](https://github.com/umd-otb/OpenTabularDataBenchmark/actions/workflows/test.yml)

## Set Up

For the most recent *stable* release, you can install the `tabben` package from PyPI:
```shell
pip3 install tabben --upgrade
```

If testing/using locally from source, you can install the `tabben` package locally; first clone this repository locally, and then install from this directory:
```shell
pip3 install -e .
```

## Documentation

There are PyTorch Dataset objects available from `tabben.datasets`. For example,
```python
from tabben.datasets import OpenTabularDataset
from torch.utils.data import DataLoader

# load the arcene dataset (default is train split) and
# save the data to the current directory 
ds = OpenTabularDataset('./', 'arcene')
for inputs, labels in DataLoader(ds, batch_size=4):
    # do stuff with inputs and labels
    pass
```

All the currently implemented datasets are accessible this way, except for the CIFAR10 dataset. Since it's a standard computer vision dataset, we just wrap the CIFAR10 Dataset from `torchvision` and convert the images into a vector.

```python
from tabben.datasets import TabularCIFAR10Dataset
from torch.utils.data import DataLoader

cifar_ds = TabularCIFAR10Dataset('./')
for inputs, labels in DataLoader(cifar_ds, batch_size=4):
    # do something with inputs and labels
    pass
```

The non-image tabular datasets can also be accessed as either numpy arrays or pandas dataframes:
```python
from tabben.datasets import OpenTabularDataset

# load the training set as numpy arrays (these are *not* copies) 
ds = OpenTabularDataset('./', 'covertype')  # defaults are numpy arrays of the training set
train_X, train_y = ds.numpy()

# load as a single pandas dataframe
df = ds.dataframe()

ds_inputs = df[ds.input_attributes]
ds_outputs = df[ds.output_attributes]
```

For a list of all the currently implemented datasets in the benchmark (except for CIFAR10), there's the function:
```python
from tabben.datasets import list_datasets

print(list_datasets())
```

## Testing
After installing `pytest`, all the tests can be run by just running
```shell
pytest
```
from a parent directory.
