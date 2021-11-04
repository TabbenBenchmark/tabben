from .image_datasets import TabularCIFAR10Dataset
from .pytorch_datasets import data_urls, OpenTabularDataset


def list_datasets():
    return list(data_urls.keys())

