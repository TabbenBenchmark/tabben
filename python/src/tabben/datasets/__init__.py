"""
Provides Dataset objects for datasets in this benchmark, and various functions for
working with the datasets or collections of datasets.
"""

from .cifar import TabularCIFAR10Dataset
from .dataset import *


def list_datasets():
    """List the non-image tabular datasets available."""
    return list(metadata.keys())

