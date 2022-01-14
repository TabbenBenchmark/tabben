from os import PathLike
from typing import Iterable, Union

from .dataset import metadata, OpenTabularDataset

__all__ = [
    'DatasetCollection',
]


def _extract_transform(transform, name, index, max_index):
    if transform is not None:
        if isinstance(transform, dict):
            return transform.get(name, None)
        elif isinstance(transform, list):
            return transform[index] if index < max_index else None
        else:
            return transform
    else:
        return None


class DatasetCollection:
    """
    A collection of tabular datasets, providing some convenience methods
    to bulk load, evaluate, or extract metadata/extras from a set of datasets.
    
    Many of the same attributes and methods for OpenTabularDataset are also
    available for DatasetCollection, although some of them are pluralized
    (e.g. `task` -> `tasks`, `dataframe` -> `dataframes`).
    """

    @classmethod
    def match(cls, location: PathLike, *,
              task: Union[str, Iterable[str]] = None,
              outputs: Union[int, Iterable[int]] = None,
              classes: Union[int, Iterable[int]] = None,
              **kwargs):
        """
        Create a dataset collection consisting of all benchmark datasets that match
        all given conditions. This can be used, for example, to get a collection
        of all binary classification datasets.
        
        Parameters
        ----------
        location: path-like
            Path to where datasets are stored/downloaded to
        task : str or iterable of str, optional
            Task(s) that must be associated with the datasets
        outputs : int or range, optional
            Number of outputs that datasets must have
        classes : int or range, optional
            Number of classes that classification datasets must have
            
            Note: this will only filter out *classification* datasets that don't
            have the correct number of classes. That is, if there are other tasks
            selected, they will not be filtered out by the `classes` filter.
        **kwargs
            All other keyword arguments are passed to the constructor.

        Returns
        -------
        DatasetCollection
            Collection of datasets matching all specified conditions

        Raises
        ------
        ValueError
            If `classes` is specified but classification datasets are excluded
            using `task`

        Notes
        -----
        To do this without requiring that datasets be available/already downloaded,
        this class method only supports filtering based on metadata that is located
        in the TOML metadata file, which does not include dataset extras.
        """
        
        datasets = set(metadata.keys())
        
        if task is not None:
            if isinstance(task, str):
                task = (task,)
            datasets.difference_update(name for name in datasets if metadata[name]['task'] not in task)

            if 'classification' in task and classes is not None:
                if isinstance(classes, int):
                    classes = (classes,)
    
                datasets.difference_update(
                    name for name in datasets
                    if metadata[name]['task'] == 'classification' and
                    metadata[name].get('classes', 2) in classes
                )
        
        if outputs is not None:
            if isinstance(outputs, int):
                outputs = (outputs,)
            datasets.difference_update(
                name for name in datasets
                if metadata[name].get('outputs', 1) not in outputs
            )
        
        return cls(location, *datasets, lazy=True, **kwargs)

    def __init__(self, location, *names,
                 split='train',
                 download=True, lazy=True,
                 transform=None, target_transform=None):
        """
        Load and create a collection of datasets stored/downloaded into `location`
        for all the datasets with names `names` and for the same subset `split`.
        
        Parameters
        ----------
        location : path-like
            Path to a directory where the dataset files are stored
        *names : str
            Names (primary keys) of the datasets to include in this collection
        split : str, default='train'
            Name of the dataset subset
        download : bool, default=True
            Whether to download the dataset files if not already present
        transform : callable or list of callable or dict of callable, optional
            Transforms/functions to apply to the input attribute vectors (see below)
        target_transform : callable or list of callable or dict of callable, optional
            Transforms/functions to apply to the output variables (see below)
            
        Notes
        -----
        The parameters `transform` and `target_transform` are optional,
        but can be specified as
        a single callable object,
        a sequence of callable objects, or
        a mapping from dataset names to callable objects.
        In each of these cases:
        
        callable
            The single callable object will be applied to all datasets.
        sequence of callable
            Based on the sequential order of the datasets, transforms are assigned
            to datasets starting at the beginning of sequence until there are either
            no more datasets or no more transforms. Datasets not matched with an
            element in the sequence (i.e. the number of datasets > the length of
            the sequence) are not transformed.
        mapping from name to callable
            For each dataset in the collection, if it is a key in the mapping, then
            the corresponding callable will be applied to the examples for that
            dataset. Otherwise (if the name is not present as a key), no transform
            is applied.
        """
        
        self.location = location
        
        names = [name.lower() for name in names]
        self.datasets = {}
        
        for index, name in enumerate(names):
            iter_transform = _extract_transform(
                transform, name, index, len(names))
            iter_target_transform = _extract_transform(
                target_transform, name, index, len(names))
            
            self.datasets[name] = OpenTabularDataset(
                location,
                name,
                split=split,
                download=download,
                transform=iter_transform,
                target_transform=iter_target_transform,
                lazy=lazy
            )

    def items(self):
        return self.datasets.items()
    
    def __iter__(self):
        return iter(self.datasets.values())

    def extra(self, extra_name):
        return {
            name: dataset.extras[extra_name]
            for name, dataset in self.datasets.items()
            if dataset.has_extra(extra_name)
        }

    @property
    def tasks(self):
        return {
            name: dataset.task
            for name, dataset in self.datasets.items()
        }

    @property
    def licenses(self):
        return {
            name: dataset.license
            for name, dataset in self.datasets.items()
            if dataset.license is not None
        }

    @property
    def bibtex(self):
        return '\n\n'.join(
            dataset.bibtex
            for dataset in self.datasets.values()
            if dataset.bibtex is not None
        )
    
    def table(self, *columns):
        """
        Returns select attributes of the datasets in this collection in a pandas
        dataframe (note, this does *not* return the data attributes, but the
        meta-attributes of the datasets themselves, like task, number of examples,
        types of attributes, etc.).
        
        Because pandas is an optional dependency, make sure you have the pandas
        package installed before calling this method.
        
        Parameters
        ----------
        *columns : str
            Names of meta-attributes to include (see Notes below for a list of options)

        Returns
        -------
        pandas.DataFrame
            Dataframe of meta attributes about the datasets in this collection
        
        Notes
        -----
        
        These are the currently supported meta-attribute names: None. (Work in progress)
        """
    
    def dataframes(self):
        return (
            (name, dataset.dataframe())
            for name, dataset in self.datasets.items()
        )
    
    def numpy(self):
        return (
            (name, dataset.inputs, dataset.outputs)
            for name, dataset in self.datasets.items()
        )
