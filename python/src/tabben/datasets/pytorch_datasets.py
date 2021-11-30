import shutil
from functools import cached_property, partial
from importlib import resources
from pathlib import Path
from typing import Iterable, Union
from warnings import warn

import numpy as np
import requests
import toml
from requests import HTTPError
from torch import from_numpy
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from .utils import has_package_installed, PathLike

__all__ = [
    # functions
    'ensure_downloaded',
    'register_dataset',
    
    # classes
    'OpenTabularDataset',
    
    # data/variables/constants
    'metadata',
    'allowed_tasks',
]

"""
URLs (and, later, possibly other metadata) for each non-CIFAR dataset.
"""
with resources.open_text('tabben.datasets', 'data.toml') as metadata_file:
    metadata = toml.load(metadata_file)


allowed_tasks = {
    'classification',
    'regression',
}


def register_dataset(name, task='classification', *, persist=False, **kwargs):
    """
    Add new datasets to the benchmark at runtime (after package loading).
    
    Args:
        name: the name of the dataset (used as a primary index, cannot be 'all')
        task: which task (see `allowed_tasks`)
    
    Keyword Args:
        persist:
            whether to also save this dataset into the data file (only for this installation)
        data_location (required):
            URL pointing to the NPZ file for this dataset
        outputs (recommended, defaults to 1):
            number of output variables
        classes (recommended for classification tasks, defaults to 2):
            number of classes (if multiple output variables, must all have the same number)
        **kwargs:
            all other keyword arguments will be added as additional metadata
    """
    
    if name != name.lower():
        warn(f'Non-lowercased name `{name}` will be converted to `{name.lower()}`')
        name = name.lower()
    
    if name in metadata:
        raise ValueError(f'Dataset with name `{name}` already registered')
    if name == 'all':
        raise ValueError('Cannot create a dataset with the special name `all`')
    
    if task not in allowed_tasks:
        raise ValueError(f'Unknown task, must be one of {allowed_tasks}')
    kwargs['task'] = task
    
    if 'data_location' not in kwargs:
        raise ValueError('A `data_location` needs to specified (remote url)')
    
    if 'outputs' not in kwargs:
        warn('The number of outputs was not specified using `outputs`, assuming 1 output variable')
        kwargs['outputs'] = 1
        
    if 'classes' not in kwargs and task == 'classification':
        warn('The number of classes was not specified using `classes`, assuming 2 classes')
        kwargs['classes'] = 2
    if 'classes' in kwargs and task == 'regression':
        raise ValueError('`classes` should be not specified if the task is regression')
    
    metadata[name] = kwargs
    if persist:
        with resources.path('tabben.datasets', 'data.toml') as p:
            with p.open('w') as f:
                toml.dump(metadata, f)


def _download_datafile(source_url: PathLike, dest_path: PathLike, download=True):
    """
    Ensures that the file (the NPZ archive) exists (will download if the destination
    file does not exist and `download` is True).
    
    Args:
        source_url: download url (should be a google drive download link)
        dest_path: full path of the destination file
        download: whether to download if not present (will error if data is not already present)
    """
    dest_path = Path(dest_path)
    
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


def ensure_downloaded(data_dir: PathLike, *datasets: str):
    """
    Downloads the specified datasets (all available datasets if none specified)
    into the data directory. This is useful e.g. using this package in an environment
    without Internet access or establishing local caches.
    
    Args:
        data_dir: directory to save the dataset files into
        *datasets: list of dataset names to load (if empty, as if all datasets)
    """
    # TODO: allow downloading the CIFAR dataset?
    
    data_dir = Path(data_dir)
    datasets = metadata.keys() if len(datasets) == 0 else set(datasets)
    
    succeeded = []
    for name, dataset_metadata in metadata.items():
        if name in datasets:
            dest_filename = data_dir / f'{name}.npz'
            try:
                _download_datafile(dataset_metadata['data_location'], dest_filename)
                succeeded.append(name)
            except Union[HTTPError, RuntimeError]:
                print(f'Unable to download the `{name}` dataset')
    
    succeeded_list = f' ({", ".join(succeeded)})' if len(succeeded) > 0 else ''
    print(f'Successfully found {len(succeeded)}/{len(datasets)} datasets{succeeded_list}.')


class OpenTabularDataset(Dataset):
    """
    A tabular dataset from the benchmark (except for CIFAR10, which is
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
        _download_datafile(metadata[name]['data_location'], data_filename, download)
        
        # load the full np arrays + input/output arrays for this split
        self.data = np.load(str(data_filename))
        self.inputs, self.outputs = self._extract_split(self.data, split)

        # convert data to torch tensors
        self.X = from_numpy(self.inputs)
        self.y = from_numpy(self.outputs)

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
        if not has_package_installed('pandas'):
            raise ImportError('Install pandas to load a dataset as a pandas dataframe')
        
        import pandas as pd

        combined = np.hstack((
            self.inputs,
            np.expand_dims(self.outputs, -1) if self.inputs.ndim == self.outputs.ndim + 1 else self.outputs
        ))
        all_columns = np.hstack((self.data['_columns-data'], self.data['_columns-labels']))

        return pd.DataFrame(data=combined, columns=all_columns)

    def numpy(self):
        return self.inputs, self.outputs
