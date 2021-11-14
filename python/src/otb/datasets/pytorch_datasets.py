import os
import shutil
from functools import cached_property, partial
from pathlib import Path
from typing import Iterable, Union

import numpy as np
import requests
import toml
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from .utils import google_drive_download_link, PathLike

"""
URLs (and, later, possibly other metadata) for each non-CIFAR dataset.

Note: these ids do not need to be updated if a new version is uploaded to the drive,
only if the file is completely "changed" (i.e. Google is treating it like a different
file).
"""
metadata = toml.load(Path(__file__).parent / 'data.toml')


def _download_datafile(source_url: PathLike, dest_path: PathLike, download=True):
    """
    Ensures that the file (the NPZ archive) exists (will download if the destination
    file does not exist and `download` is True).
    
    Args:
        source_url: download url (should be a google drive download link)
        dest_path: full path of the destination file
        download: whether to download if not present (will error if data is not already present)
    """
    
    if dest_path.exists():
        print(f'Data already available at `{dest_path}`')
    elif download:
        print(f'Downloading data from `{source_url}` into `{dest_path}`')
        r = requests.get(source_url, stream=True)
        
        if r.status_code != requests.codes.ok:
            r.raise_for_status()
            raise RuntimeError(f'Unable to download file from `{source_url}`')

        declared_file_size = int(r.headers.get('Content-Length', 0))
        desc = '(Unknown file size)' if declared_file_size == 0 else ''
        
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        r.raw.read = partial(r.raw.read, decode_content=True)
        with tqdm.wrapattr(r.raw, "read", total=declared_file_size, desc=desc) as progressed_data:
            with dest_path.open('wb') as output_file:
                shutil.copyfileobj(progressed_data, output_file)
    else:
        raise ValueError('Data files don\'t exist but not instructed to download')


class OpenTabularDataset(Dataset):
    """
    A tabular dataset from the benchmark (except for the CIFAR10, which is
    accessible in tabular form using `TabularCIFAR10Dataset`).
    """
    
    def __init__(self, data_dir: PathLike, name: str,
                 split: Union[str, Iterable[str]] = 'train',
                 download=True,
                 transform=None):
        
        self.data_dir = Path(data_dir)
        self.name = name.lower()
        self.split = split
        self.transform = transform

        if self.name not in metadata:
            raise ValueError(f'dataset with name `{self.name}` not recognized')

        # download data if not yet already
        data_filename = self.data_dir / f'{self.name}.npz'
        _download_datafile(google_drive_download_link(metadata[name]['doc_id']), data_filename, download)
        
        # load the full np arrays + input/output arrays for this split
        self.data = np.load(data_filename)
        self.inputs, self.outputs = self._extract_split(self.data, split)

        # convert data to torch tensors
        self.X = torch.from_numpy(self.inputs)
        self.y = torch.from_numpy(self.outputs)

    def _extract_split(self, data, split: str):
        if split not in self.splits:
            raise ValueError(f'dataset `{self.name}` does not have a `{split}` split')
    
        # return requested split
        return data[f'{split}-data'], data[f'{split}-labels']

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        inputs = self.X[idx, :]
        outputs = self.y[idx].item() if self.y[idx].numel() == 1 else self.y[idx]
        example_pair = (inputs, outputs)
        
        # apply transforms if there are any to the input-output pair
        return self.transform(example_pair) if self.transform else example_pair

    def __eq__(self, other):
        if not isinstance(OpenTabularDataset, other):
            return False
        
        return self.name == other.name and self.split == other.split and self.transform == other.transform
    
    def __repr__(self):
        attributes = {
            'data_dir': repr(self.data_dir),
            'name': repr(self.name),
            'split': repr(self.split),
            'transform': repr(self.transform),
        }
        attributes_string = ', '.join(
                '='.join(pair) for pair in attributes.items()
        )
        return f'OpenTabularDataset({attributes_string})'

    @cached_property
    def splits(self):
        return {filename.partition('-')[0] for filename in self.data.files
                if '-' in filename and not filename.startswith('_')}

    @cached_property
    def input_attributes(self):
        return self.data['_columns-data']
    
    @cached_property
    def output_attributes(self):
        return self.data['_columns-labels']
    
    def dataframe(self):
        import pandas as pd

        combined = np.hstack((
            self.inputs,
            np.expand_dims(self.outputs, -1) if self.inputs.ndim == self.outputs.ndim + 1 else self.outputs
        ))
        all_columns = np.hstack((self.data['_columns-data'], self.data['_columns-labels']))

        return pd.DataFrame(data=combined, columns=all_columns)

    def numpy(self):
        return self.inputs, self.outputs

