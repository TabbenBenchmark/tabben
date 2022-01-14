import pytest


################################################################################
# Shared pytest fixtures (never need to be imported anywhere)
################################################################################

@pytest.fixture(scope='session')
def data_dir(tmp_path_factory):
    return tmp_path_factory.mktemp('data')
