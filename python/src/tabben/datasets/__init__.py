"""
Provides Dataset objects for datasets in this benchmark, and various functions for
working with the datasets or collections of datasets.
"""

from .dataset import *
from .collection import *

__all__ = [
    # functions
    'ensure_downloaded',
    'register_dataset',
    'validate_dataset_file',
    'list_datasets',
    
    # classes
    'OpenTabularDataset',
    'DatasetFormatError',
    'DatasetCollection',
    
    # data/variables/constants
    'metadata',
    'allowed_tasks',
]


def list_datasets():
    """List the non-image tabular datasets available."""
    return list(metadata.keys())

