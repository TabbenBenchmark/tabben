"""
Test the functions used for maintaining the state of the datasets (handling of
metadata, registering new datasets, etc.).
"""

from tabben.datasets import register_dataset, list_datasets
import pytest


def test_register_dataset_failures():
    # invalid dataset names
    with pytest.raises(ValueError) as exc_info:
        register_dataset('all')
    assert 'special name' in str(exc_info.value) and 'all' in str(exc_info.value)
    
    with pytest.raises(ValueError) as exc_info:
        with pytest.warns(UserWarning, match='lowercase'):
            register_dataset('AlL')
    assert 'special name' in str(exc_info.value) and 'all' in str(exc_info.value)
    
    for dataset_name in list_datasets():
        with pytest.raises(ValueError) as exc_info:
            register_dataset(dataset_name)
        assert 'already registered' in str(exc_info.value)
    
    # tasks
    with pytest.raises(ValueError) as exc_info:
        register_dataset('placeholder', 'class')
    assert 'unknown task' in str(exc_info.value).lower()
