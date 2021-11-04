import os

import numpy as np
import requests
import torch
from torch.utils.data import Dataset

from .utils import google_drive_download_link

# TODO: replace this with an actual data file instead of in code
"""
URLs (and, later, possibly other metadata) for each non-CIFAR dataset.

Note: these do not need to be updated if a new version is uploaded to the drive,
only if the file is completely "changed" (i.e. Google is treating it like a different
file).
"""
data_urls = {
    'arcene': google_drive_download_link('1cnuQwVtQ-FsJ_En9_ln2KU0n30wJ4ffe'),
    'covertype': google_drive_download_link('1ixC-jAgdAgPnCL37uaTEnBep7q43liNP'),
    'higgs': google_drive_download_link('1mz6E-5eV5ThnzdbimvTTeTTGSoJTjS_I'),
    'poker': google_drive_download_link('1yVdp4pHSmrFasHhX4j4vtxVHYHUvciun'),
    'sarcos': google_drive_download_link('1Nr7MIWogLo0aY_uQdSCSfGysMr5Wswq5'),
}


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
    # TODO: factor non-pytorch sections into its own thing (for non-pytorch users)
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