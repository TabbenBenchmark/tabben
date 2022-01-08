# A Python package for the Open Benchmark for Tabular Data

[![Package Unit Tests (Latest)](https://github.com/umd-otb/OpenTabularDataBenchmark/actions/workflows/test.yml/badge.svg)](https://github.com/umd-otb/OpenTabularDataBenchmark/actions/workflows/test.yml)

This is a Python package for working with the tabben benchmark for tabular data (for machine learning tasks):
- loading, processing, or inspecting datasets,
- evaluating models on test sets using consistent evaluation metrics, and
- examining collections of datasets from the benchmark.

See the [tabben website](https://www.tabben.org/) for more info about the project.

## Set Up

For the most recent *stable* release, you can install the `tabben` package from PyPI:
```shell
pip3 install tabben --upgrade
```

If testing/using locally from source, you can install the `tabben` package locally; first clone this repository locally, and then install from the python subdirectory:
```shell
pip3 install -e .
```

## Documentation

See the [package docs](https://umd-otb.github.io/OpenTabularDataBenchmark/) for tutorials, API references, and details about each of the datasets included in the benchmark.

## Testing

After installing `pytest` and the required test dependencies, all the tests can be run by just running
```shell
pytest
```
from the package directory.

There is also a "large" pytest mark on tests that test with large datasets that can be used to exclude running on those datasets (as those can take a while to run, upwards of a minute depending on the network and computer).
```shell
pytest -m "not large"
```
