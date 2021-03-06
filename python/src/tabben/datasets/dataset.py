"""
Implementation of a dataset class for benchmarks, also providing access methods
via numpy arrays or pandas dataframes (should not access this module directly). If
PyTorch is installed, the dataset class can be used as a PyTorch-compatible dataset.
"""

import pathlib
import shutil
import warnings
from collections import defaultdict
from functools import cached_property, partial
from importlib import resources
from pathlib import Path
from typing import Iterable, Optional, Sequence, Set, Union
from warnings import warn

import numpy as np
import requests
import toml
from numpy.lib.npyio import NpzFile
from requests import HTTPError
from tqdm.auto import tqdm

from ..utils import has_package_installed, PathLike

if has_package_installed('simplejson'):
    import simplejson as json
else:
    import json

if not has_package_installed('torch'):
    # since torch is a heavy requirement for those who don't use it, make it optional
    warn('Install PyTorch to use the dataset as a PyTorch-compatible dataset')
    Dataset = object
else:
    from torch.utils.data import Dataset

__all__ = [
    # functions
    'ensure_downloaded',
    'register_dataset',
    'validate_dataset_file',
    
    # classes
    'OpenTabularDataset',
    'DatasetFormatError',
    
    # data/variables/constants
    'metadata',
    'allowed_tasks',
]

# metadata for each (non-CIFAR) dataset.
with resources.open_text('tabben.datasets', 'data.toml') as metadata_file:
    metadata = toml.load(metadata_file)


allowed_tasks = {
    'classification',
    'regression',
}


################################################################################
#      Functional Interface: working with dataset metadata/benchmark sets      #
################################################################################
def register_dataset(name: str, task: str = 'classification', *,
                     data_location: str,
                     persist=False, **kwargs) -> None:
    """
    Add new datasets to the benchmark at runtime (after package loading).
    
    Parameters
    ----------
    name : str
        Name of the dataset (used as a primary index, cannot be 'all`)
    task : str
        Which task is associated with this dataset (see `allowed_tasks`)
    persist : bool
        Whether to save this dataset so that it persists between restarts (only
        for this installation)
    data_location : str
        URI string pointing to the NPZ file for this dataset
    outputs : int, recommended, default=1
        Number of output variables
    classes : int, recommended for classification tasks, default=2
        Number of classification classes
    extras_location : str
        URI string pointing to a JSON file of "extras" metadata for this dataset
    **kwargs
        All other keyword arguments are stored as additional metadata in the TOML file

    See Also
    --------
    validate_dataset_file: Validate the NPZ file before adding as a new dataset
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
    
    if 'outputs' not in kwargs:
        warn('The number of outputs was not specified using `outputs`, assuming 1 output variable')
    
    if 'classes' not in kwargs and task == 'classification':
        warn('The number of classes was not specified using `classes`, assuming 2 classes')
        kwargs['classes'] = 2
    if 'classes' in kwargs and task == 'regression':
        raise ValueError('`classes` should be not specified if the task is regression')

    kwargs['task'] = task
    kwargs['data_location'] = data_location
    metadata[name] = kwargs
    
    if persist:
        with resources.path('tabben.datasets', 'data.toml') as p:
            with p.open('w') as f:
                toml.dump(metadata, f, encoder=toml.TomlNumpyEncoder())


class DatasetFormatError(Exception):
    """
    An exception due to an NPZ dataset file having an unexpected format (in addition
    to the usual NPZ file format requirements).
    """


def validate_dataset_file(filepath: PathLike) -> None:
    """
    Validate a NPZ dataset file at a local path to make sure that the dataset it
    contains can be read as a valid dataset using this package. This function is
    needed for interactive use at the REPL.
    
    Parameters
    ----------
    filepath : str or path-like
        Filepath of the NPZ dataset file

    Raises
    ------
    FileNotFoundError
        If the filepath does not exist
    IOError
        If the file cannot be read at all
    DatasetFormatError
        If there is an error with the format of the NPZ dataset file
    """
    if not isinstance(filepath, pathlib.Path) and filepath.startswith('file://'):
        filepath = filepath[7:]
    else:
        filepath = Path(filepath)
    
    # check file itself
    if not filepath.exists():
        raise FileNotFoundError('Dataset file does not exist')
    if not filepath.is_file():
        raise IOError('Dataset path does not point to a file')
    
    # check numpy file format
    try:
        data: NpzFile = np.load(str(filepath))
    except IOError:
        raise DatasetFormatError('Dataset file is not in the correct numpy format')
    except ValueError:
        raise DatasetFormatError('Dataset file contains object arrays, which require pickle (not allowed)')
    
    try:
        names = data.files
    except AttributeError:
        raise DatasetFormatError('Dataset file is not an NPZ file of multiple arrays')
    
    # check just the "filenames"
    split_names = {name for name in names if not name.startswith('_')}
    metadata_names = {name for name in names if name not in split_names}
    
    splits_parts = defaultdict(set)
    for split_name in split_names:
        if split_name.count('-') != 1:
            raise DatasetFormatError(
                'Dataset arrays not starting with underscore need to have exactly 1 '
                'dash in the name'
            )
        
        split, _, part = split_name.partition('-')
        
        splits_parts[split].add(part)
    
    for split, parts in splits_parts.items():
        missing_parts = {'data', 'labels'} - parts
        extra_parts = parts - {'data', 'labels'}
        
        if len(missing_parts) > 0:
            raise DatasetFormatError(
                f'Dataset has missing parts for `{split}` split:'
                f' {", ".join(missing_parts)}'
            )
        if len(extra_parts) > 0:
            raise DatasetFormatError(f'Dataset has extra parts for `{split}` split: {", ".join(extra_parts)}')
    
    if len(splits_parts) == 0:
        raise DatasetFormatError('The dataset file does not contain data for any splits')
    
    if '_columns-data' not in metadata_names:
        raise DatasetFormatError('Dataset needs to have a `_columns-data` array')
    if '_columns-labels' not in metadata_names:
        raise DatasetFormatError('Dataset needs to have a `_columns-labels` array')
    if '_version' not in metadata_names:
        warn('This dataset file does not have a `_version` array, you may run into compatibility issues')
    
    # check sizes
    num_inputs = []
    num_outputs = []
    
    for split in splits_parts:
        try:
            input_data = data[f'{split}-data']
        except ValueError:
            raise DatasetFormatError(f'Input array for `{split}` split requires pickle')
        try:
            output_data = data[f'{split}-labels']
        except ValueError:
            raise DatasetFormatError(f'Output array for `{split}` split requires pickle')
        
        if len(input_data.shape) != 2:
            raise DatasetFormatError('Dataset input data must be 2-dimensional')
        if len(output_data.shape) not in (1, 2):
            raise DatasetFormatError('Dataset output data must be either 1- or 2-dimensional')
        
        if input_data.shape[0] != output_data.shape[0]:
            raise DatasetFormatError(f'The number of examples is not the same in the `{split}` split')
        
        num_inputs.append(input_data.shape[1])
        num_outputs.append(output_data.shape[1] if len(output_data.shape) == 2 else 1)
    
    if any(num != num_inputs[0] for num in num_inputs):
        raise DatasetFormatError('Number of input attributes does not match between splits')
    if any(num != num_outputs[0] for num in num_outputs):
        raise DatasetFormatError('Number of output labels does not match between splits')
    
    print(f'Verified the dataset file located at `{filepath}`')


def check_version(dataset_filepath: PathLike, min_version: str):
    min_version = tuple(min_version.split('.'))
    with np.load(dataset_filepath) as data:
        # if `_version` not in the file, treat as incompatible with any version
        if '_version' not in data.files:
            return False
        if not np.issubdtype(data['_version'].dtype, np.integer):
            return False
        
        actual_version = tuple(data['_version'])
    
    # this assertion may not actually be wanted
    assert len(min_version) == len(actual_version) == 3
    
    for actual_part, min_part in zip(actual_version, min_version):
        if int(actual_part) < int(min_part):
            return False
    
    return True


def _download_datafile(source_url: PathLike, dest_path: PathLike,
                       download: bool = True, ignore_version: bool = False) -> None:
    """
    Ensures that the specified resource file exists (will download if the destination
    file does not exist and `download` is True).
    
    Note that the source "URL" can be a local file (if it starts with 'file://'
    instead of 'https://') and the file will be copied from the source to the destination
    path, even if `download=False`.
    
    Args:
        source_url: download uri
        dest_path: full path of the destination file
        download: whether to download if not present (will error if data is not already present)
        ignore_version: whether to disable the version check (e.g. for non-npz datafiles)
    """
    dest_path = Path(dest_path)
    
    if dest_path.exists() and (ignore_version or check_version(dest_path, '1.1.0')):
        print(f'Data already available at `{dest_path}`')
        
        # in any other case, the version in the datafile will *not* be checked:
        # this should be fine, as long as we make sure that any version of the code
        # cannot download dataset files with incompatible formats (assume risk with
        # local file path)
    elif source_url.startswith('file://'):
        source_path = source_url[7:]
        print(f'Copying locally from `{source_path}` to `{dest_path}`')
        shutil.copyfile(source_path, dest_path)
    elif download:
        print(f'Downloading from `{source_url}` to `{dest_path}`')
        r = requests.get(source_url, stream=True)
        
        if r.status_code != requests.codes.ok:
            r.raise_for_status()
            raise IOError(f'Unable to download file from `{source_url}`')
        
        # determine total file size for the progress bar
        declared_file_size = int(r.headers.get('Content-Length', 0))
        desc = '(Unknown file size)' if declared_file_size == 0 else ''
        
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        r.raw.read = partial(r.raw.read, decode_content=True)
        with tqdm.wrapattr(r.raw, "read", total=declared_file_size, desc=desc) as progressed_data:
            with dest_path.open('wb') as output_file:
                shutil.copyfileobj(progressed_data, output_file)
    else:
        raise ValueError('File does not exist but not instructed to download')


def ensure_downloaded(data_dir: PathLike, *datasets: str) -> None:
    """
    Downloads the specified datasets (all available datasets if none specified)
    into the data directory if they are not already present. This is useful in
    situations where this package is used in an environment without Internet
    access or for establishing local shared caches.
    
    Parameters
    ----------
    data_dir : path-like
        Directory to save the dataset files in
    *datasets : str
        Names of datasets to download (if empty, all datasets will be downloaded)

    """
    
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


################################################################################
#                Object-Oriented Interface to a single dataset                 #
################################################################################
class OpenTabularDataset(Dataset):
    """
    A tabular dataset from the benchmark.
    """
    
    def __init__(self, data_dir: PathLike, name: str,
                 split: Union[str, Iterable[str]] = 'train', *,
                 download=True, lazy=False,
                 transform=None, target_transform=None):
        """
        Load and create a dataset with the given `name` (storing the dataset files
        in the `data_dir`) for the particular subset given by `split`.
        
        Parameters
        ----------
        data_dir : path-like
            Directory to load/store the dataset files
        name : str
            Name (primary key) of the dataset
        split : str or iterable of str, default='train'
            Subset split of the dataset to load
        download : bool, default=True
            Whether to download the dataset files if not already present in `data_dir`
        lazy : bool, default=False
            Whether to postpone loading the data into memory until the first access
            
            Not implemented yet!
        transform : callable, optional
            Transform or function that will be applied to the input attributes vector
        target_transform : callable, optional
            Transform or function that will be applied to the target variables
        """
        
        self.data_dir = Path(data_dir)
        self.name = name.lower()
        self.split = [split] if isinstance(split, str) else split
        self.transform = transform
        self.target_transform = target_transform
        
        if self.name not in metadata:
            raise ValueError(f'dataset with name `{self.name}` not recognized')
        
        if 'test' in self.split:
            warnings.warn('Test set being loaded: make sure you only use this once! '
                          'Do not use this for statistics.')
        
        # download data if not yet already
        data_filename = self.data_dir / f'{self.name}.npz'
        _download_datafile(metadata[self.name]['data_location'], data_filename, download)
        
        self.extras = None
        if 'extras_location' in metadata[self.name]:
            extras_filename = self.data_dir / f'{self.name}.json'
            
            try:
                _download_datafile(metadata[self.name]['extras_location'], extras_filename, download, True)
                with extras_filename.open() as f:
                    self.extras = json.load(f)
            except Union[HTTPError, IOError, ValueError]:
                warn(f'Downloading extras for `{self.name}` failed, proceeding without extras data')
        
        # load the full np arrays + input/output arrays for this split
        with np.load(str(data_filename)) as data:
            self.splits = {name.partition('-')[0] for name in data.files
                           if '-' in name and not name.startswith('_')}
            self.inputs, self.outputs = self._extract_split(data, self.split)
            self.input_attributes = data['_columns-data']
            self.output_attributes = data['_columns-labels']
    
    def _extract_split(self, data: NpzFile, splits: Sequence[str]) -> (np.ndarray, np.ndarray):
        nonexistent_splits = set(splits) - set(self.splits)
        if len(nonexistent_splits) != 0:
            raise ValueError(f'dataset `{self.name}` does not have splits: {", ".join(nonexistent_splits)}')
        
        # return requested splits
        inputs = np.vstack([data[f'{split}-data'] for split in splits])
        stack_func = np.hstack if data[f'{splits[0]}-labels'].ndim == 1 else np.vstack
        outputs = stack_func([data[f'{split}-labels'] for split in splits])
        return inputs, outputs
    
    def __len__(self) -> int:
        """
        Returns
        -------
        int
            The number of examples in this subset of the dataset
        """
        return self.inputs.shape[0]
    
    def __getitem__(self, idx) -> (np.ndarray, np.ndarray):
        inputs = self.inputs[idx, :]
        outputs = self.outputs[idx].item() if self.outputs[idx].size == 1 else self.outputs[idx]
        
        if self.transform:
            inputs = self.transform(inputs)
        if self.target_transform:
            outputs = self.target_transform(outputs)
        
        return inputs, outputs
    
    def __eq__(self, other) -> bool:
        if not isinstance(OpenTabularDataset, other):
            return False
        
        return self.name == other.name and self.split == other.split and self.transform == other.transform \
               and self.target_transform == self.target_transform
    
    def __repr__(self) -> str:
        attributes = {
            'data_dir': repr(self.data_dir),
            'name': repr(self.name),
            'split': repr(self.split),
            'transform': repr(self.transform),
            'target_transform': repr(self.target_transform),
        }
        attributes_string = ', '.join(
            '='.join(pair) for pair in attributes.items()
        )
        return f'OpenTabularDataset({attributes_string})'
    
    @property
    def has_extras(self) -> bool:
        """
        Whether this dataset has "extras" metadata, which typically contains the
        mappings for categories from numbers to labels, license information, bibtex,
        data profiles, etc.
        
        Returns
        -------
        bool
            Whether this dataset has extras
        """
        return self.extras is not None
    
    def has_extra(self, extra_name) -> bool:
        """
        Check whether this dataset has a specific extra.
        
        Parameters
        ----------
        extra_name : str
            Name of the extra to check

        Returns
        -------
        bool
            True if this dataset contains an extra with this name, otherwise False
        """
        return self.has_extras and extra_name in self.extras
    
    @property
    def license(self) -> Optional[str]:
        """
        License text for the dataset itself. (The tabben package is MIT-licensed,
        but the datasets themselves may not be as permissive. Particularly if you
        intend to use the datasets in a commercial setting, make sure to check the
        license of the datasets used.)
        
        Returns
        -------
        str or None
            License text if available, otherwise None
        """
        
        # TODO split this into 'license' and 'license_info'
        return self.extras['license'] if self.has_extra('license') else None
    
    @property
    def bibtex(self) -> Optional[str]:
        """
        Bibtex for the dataset and any associated papers that the original dataset
        providers have asked to be cited. This is useful if you are doing research
        with this benchmark and want to cite the original datasets.
        
        Returns
        -------
        str or None
            Bibtex if available, otherwise None
        """
        
        return self.extras['bibtex'] if self.has_extra('bibtex') else None
    
    @property
    def categorical_attributes(self) -> Optional[Sequence[str]]:
        """
        Labels/names of the categorical attributes of this dataset if available.
        
        Returns
        -------
        sequence of str or None
            List of names of categorical attributes if available, otherwise None
        """
        
        return self.extras['categories'].keys() if self.has_extra('categories') else None
    
    @property
    def num_inputs(self) -> int:
        """
        Number of input attributes for this dataset.
        
        Returns
        -------
        int
            Number of raw input attributes (without preprocessing or transforms)
        """
        
        return len(self.input_attributes)
    
    @property
    def num_outputs(self) -> int:
        """
        Number of output/target variables for this dataset.
        
        Returns
        -------
        int
            Number of raw output variables (without preprocessing or transforms)
        """
        return len(self.output_attributes)
    
    @property
    def num_classes(self) -> int:
        """
        Number of classes for this dataset if it is a classification task.
        
        Returns
        -------
        int
            Number of classification classes
            
        Raises
        ------
        AttributeError
            If called on a non-classification dataset
        """
        
        if metadata[self.name]['task'] == 'classification':
            return metadata[self.name]['classes']
        else:
            raise AttributeError('Non-classification task datasets don\'t have a number of classes')
    
    @property
    def task(self) -> str:
        """
        Task associated with this dataset.
        
        Returns
        -------

        See Also
        --------
        allowed_tasks
            List of allowed/currently supported tasks for the benchmark
        """
        
        return metadata[self.name]['task']
    
    def dataframe(self) -> 'pandas.DataFrame':
        """
        Create a pandas DataFrame consisting of both input attributes and output labels
        for this dataset (for this specific split).
        
        Since pandas is not a required dependency, make sure you already have pandas
        installed before you call this method.
        
        Returns
        -------
        pandas.DataFrame
            Dataframe containing the complete dataset for this split
        """
        
        if not has_package_installed('pandas'):
            raise ImportError('Install pandas to load a dataset as a pandas dataframe')
        
        import pandas as pd
        
        # put input attributes and the output labels in the same dataframe
        combined = np.hstack(
            (
                self.inputs,
                np.expand_dims(
                    self.outputs, -1
                ) if self.inputs.ndim == self.outputs.ndim + 1 else self.outputs
            )
        )
        all_columns = np.hstack((self.input_attributes, self.output_attributes))
        
        return pd.DataFrame(data=combined, columns=all_columns)
    
    def numpy(self) -> (np.ndarray, np.ndarray):
        """
        Return the input and output attributes as numpy arrays in the standard
        scikit-learn format of (inputs, outputs).
        
        Returns
        -------
        tuple of numpy.ndarray
            2-tuple of inputs and outputs as matrices/vectors
        """
        return self.inputs, self.outputs
