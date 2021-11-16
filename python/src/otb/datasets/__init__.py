from .image_datasets import TabularCIFAR10Dataset
from .pytorch_datasets import metadata, OpenTabularDataset, ensure_downloaded


def list_datasets():
    """List the non-image tabular datasets available."""
    return list(metadata.keys())

