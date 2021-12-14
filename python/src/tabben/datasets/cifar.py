"""
Implements a dataset for the CIFAR10 dataset with cleaned test labels (training
labels are not cleaned).
"""

from typing import Iterable, Union

from torch.utils.data import Dataset

from ..utils import has_package_installed, PathLike


class TabularCIFAR10Dataset(Dataset):
    """
    This is a wrapper dataset class around the CIFAR10 dataset
    provided by torchvision, which treats the input as a tabular-like
    vector of values.
    """
    
    def __init__(self, data_dir: PathLike, split: Union[str, Iterable[str]] = 'train', download=True):
        # allow torchvision as an optional dependency
        if not has_package_installed('torchvision'):
            raise ImportError('torchvision is not installed, but is required for the CIFAR10 dataset')
        
        import torchvision
        import torchvision.transforms as transforms
        
        if isinstance(split, str):
            split = [split]
        
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )
        
        self.ds = torchvision.datasets.CIFAR10(
            data_dir,
            train='train' in split,
            download=download,
            transform=transform,
        )
    
    def __len__(self):
        return self.ds.__len__()
    
    def __getitem__(self, idx):
        im_data, im_label = self.ds.__getitem__(idx)
        
        return im_data.flatten(), im_label
