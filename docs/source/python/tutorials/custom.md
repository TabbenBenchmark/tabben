
# Custom Datasets

It's possible to add private or other datasets in the benchmark (at least locally). The data needs to be in a NPZ file in a particular format (see the developer's documentation in the [scripts](https://github.com/umd-otb/OpenTabularDataBenchmark/blob/main/scripts/README.md) directory), and then you can call the `register_dataset` function:
```python
from tabben.datasets import register_dataset

register_dataset(
        'name-of-my-dataset',
        'classification',   # or regression
        data_location='https://url.to/a/npz/file/hosted/somewhere.npz',
        outputs=1,
        classes=5,
)
```

If you want to contribute a dataset to the official benchmark, you could run this with the keyword argument `persist=True` to save this dataset to the data file and open a pull request to the main repository (assuming built from source). See the [contributing guidelines](https://github.com/umd-otb/OpenTabularDataBenchmark/blob/main/CONTRIBUTING.md) for details.

## Helpers

To check that a NPZ dataset file can be loaded by this package (after registering as above), there's a `validate_dataset_file` function to check the contents:
```python
from tabben.datasets import validate_dataset_file

validate_dataset_file('path/or/url/to/file.npz')
```
