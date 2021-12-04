# Contributing Guide

## Contributing

### Contributing in Code
Found a bug or have a suggestion to make things better? Go ahead and open an issue on the GitHub repo.

### Contributing a Dataset

We do want to balance having more breadth/coverage in the benchmark with the computational burden of running the benchmark for a new (and existing!) models. It's probably best to open a new issue first to discuss. Eventually, each new dataset should have the following items:

- A script (ideally in python) that takes the raw dataset/url and converts it to our standardized data format (placed in [scripts](/scripts))
- A description of the dataset
- An entry in [data.toml](/python/src/tabben/datasets/data.toml) (this can be generated using the `register_dataset` function with the keyword argument `persist=True`)


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

