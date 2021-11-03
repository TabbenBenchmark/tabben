import numpy as np
import os
import requests
import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
# TODO: lazy loading of some dependencies


# TODO: replace this with an actual data file instead of in code
data_urls = {
    'arcene': 'https://drive.google.com/uc?id=1cnuQwVtQ-FsJ_En9_ln2KU0n30wJ4ffe',
    'covertype': 'https://drive.google.com/uc?id=1ixC-jAgdAgPnCL37uaTEnBep7q43liNP',
    'poker': 'https://drive.google.com/uc?id=1yVdp4pHSmrFasHhX4j4vtxVHYHUvciun',
    'sarcos': 'https://drive.google.com/uc?id=1Nr7MIWogLo0aY_uQdSCSfGysMr5Wswq5',
}


def list_datasets():
    return list(data_urls.keys())


def download_datafile(source_url, dest_path, download=True):
    if os.path.exists(dest_path):
        print(f'Data already exists at {dest_path}')
    elif download:
        r = requests.get(source_url)
        if r.status_code == 200:
            with open(dest_path, 'wb') as output_file:
                output_file.write(r.content)
        else:
            raise RuntimeError(f'unable to download file from {source_url}')
    else:
        raise ValueError('Data files don\'t exist but not instructed to download')


def extract_splits(filenames):
    return {filename.partition('-')[0] for filename in filenames if '-' in filename}


class OpenTabularDataset(Dataset):
    """
    A tabular dataset from the benchmark (except for the CIFAR10, which is
    accessible in tabular form using `TabularCIFAR10Dataset`).
    """
    
    # TODO: preprocessing?
    def __init__(self, data_dir, name, split='train', download=True, transform=None):
        name = name.lower()
        if name not in data_urls:
            raise ValueError(f'dataset with name `{name}` not recognized')
        
        # load data files (download if not present)
        data_filename = os.path.join(data_dir, f'{name}.npz')
        download_datafile(data_urls[name], data_filename, download)

        data = np.load(data_filename)

        # check that the requested split exists
        if split not in extract_splits(data.files):
            raise ValueError(f'dataset `{name}` does not have a `{split}`')

        # convert data to torch tensors
        self.X = torch.from_numpy(data[f'{split}-data'])
        self.y = torch.from_numpy(data[f'{split}-labels'])
        
        self.transform = transform

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        inputs = self.X[idx, :]
        outputs = self.y[idx].item() if self.y[idx].numel() == 1 else self.y[idx]
        example_pair = (inputs, outputs)
        
        return self.transform(example_pair) if self.transform else example_pair


class TabularCIFAR10Dataset(Dataset):
    """
    This is a wrapper dataset class around the CIFAR10 dataset
    provided by torchvision, which treats the input as a tabular-like
    vector of values.
    """
    
    def __init__(self, data_dir, split='train', download=True):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        self.ds = torchvision.datasets.CIFAR10(
                data_dir,
                train=split == 'train',
                download=download,
                transform=transform,
        )
    
    def __len__(self):
        return self.ds.__len__()
    
    def __getitem__(self, idx):
        im_data, im_label = self.ds.__getitem__(idx)
        
        return im_data.flatten(), im_label

