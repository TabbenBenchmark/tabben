from importlib import resources

import pytest
import toml
from tabben.datasets import allowed_tasks


################################################################################
# General utilities
################################################################################

@pytest.fixture
def data_toml():
    with resources.open_text('tabben.datasets', 'data.toml') as metadata_file:
        metadata = toml.load(metadata_file)
    
    return metadata


################################################################################
# Individual unit tests
################################################################################

def test_required_fields(data_toml):
    for metadata in data_toml.values():
        assert 'data_location' in metadata
        assert 'task' in metadata
        assert metadata['task'] in allowed_tasks


def test_dataset_identifiers(data_toml):
    for name in data_toml.keys():
        assert name != 'all'
        assert name.lower() == name
