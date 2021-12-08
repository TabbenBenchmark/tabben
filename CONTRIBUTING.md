# Contributing Guide

## Contributing in Code
Found a bug or have a suggestion to make things better? Go ahead and open an issue on the GitHub repo.

## Contributing a Dataset

We do want to balance having more breadth/coverage in the benchmark with the computational burden of running the benchmark for a new (and existing!) models. It's probably best to **open a new issue first to discuss**. Eventually, each new dataset should have the following items (although you don't have to have all these things at the start):

- A script (ideally in python) that takes the raw dataset/url and converts it to our standardized data format (placed in [scripts](/scripts)): this is for documenting the data transformation from the raw data
- An entry in [data.toml](/python/src/tabben/datasets/data.toml) (this can be generated using the `register_dataset` function with the keyword argument `persist=True`)
- At least a basic test in [test_datasets.py](python/tests/test_datasets.py), can be based on the tests for other datasets

The pipeline should be able to be recreated by running something like
```shell
python3 path/to/data/generation/script.py output/directory --source data/source 
```
And then in a python shell:
```python-repl
>>> from tabben.datasets import validate_dataset_file, register_dataset
>>> validate_dataset_file('path/to/dataset_file')
>>> register_dataset(
    'new-name',
    'classification',
    data_location='https://link/to/asset/on/GitHub/releases.npz',
    num_classes=5,
    num_outputs=2,
    persist=True,  # this one is important to have so the file is modified
)
```

The dataset file itself shouldn't be committed to git, but added to a GitHub release (and then the URL goes in [data.toml](python/src/tabben/datasets/data.toml) or is passed to `register_dataset` above).

## Repository Organization
At the top-level,
- [python](/python) contains the Python package `tabben`
- [scripts](/scripts) contains scripts used for downloading & processing the original/raw data

### Dependencies

There are a bunch of `requirements.txt` files scattered throughout the directories. They describe the python dependencies for
- [running any of the scripts in the scripts directory](/scripts/requirements.txt)
- [building the documentation](/docs/requirements.txt)
- [running the tests](/python/tests)

(Build dependencies for the python package are in a [pyproject.toml](/python/pyproject.toml) file and the package dependencies are specified in both a [setup.cfg](/python/setup.cfg) and [setup.py](/python/setup.py).)

