"""
Provides Dataset objects for datasets in this benchmark, and various functions for
working with the datasets.
"""

from .image_datasets import TabularCIFAR10Dataset
from .pytorch_datasets import *


def list_datasets():
    """List the non-image tabular datasets available."""
    return list(metadata.keys())

