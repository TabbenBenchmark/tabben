from .image_datasets import TabularCIFAR10Dataset
from .pytorch_datasets import data_urls, OpenTabularDataset


def list_datasets():
    """List the non-image tabular datasets available."""
    return list(data_urls.keys())

