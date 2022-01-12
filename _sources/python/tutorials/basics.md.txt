
# Basic Package Usage

## Accessing a Dataset

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

All the currently implemented datasets are accessible this way.

We can also access these tabular datasets as either numpy arrays or pandas dataframes:
```python
from tabben.datasets import OpenTabularDataset

# load the training set as numpy arrays (these are NOT copies) 
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

## Evaluating the Results of a Model

There are standard metrics available (either from scikit-metrics or compatible with autogluon). For most cases,

```python
from tabben.evaluators import get_metrics

eval_metrics = get_metrics('classification', classes=2)


```

