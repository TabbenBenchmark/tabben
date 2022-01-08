from python.src.tabben.datasets import OpenTabularDataset

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
    Represents a collection of tabular datasets, providing some convenience methods
    to bulk load, evaluate, or extract metadata/extras from a set of datasets.
    """

    @classmethod
    def match(cls, location, *, task=None, outputs=None, classes=None):
        """
        Create a dataset collection where all the datasets match all given
        conditions. This can be used, for example, to get a collection of all
        binary classification datasets.
        """
        
        datasets = []
        
        if task is not None:
            pass
        
        if outputs is not None:
            pass
        
        if task == 'classification' or 'classification' in task and classes is not None:
            pass
        
        return cls(location, *datasets)

    def __init__(self, location, *names,
                 split='train', download=True,
                 transform=None, target_transform=None):
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
        
        Args:
            *columns: names of meta-attributes to be included

        Returns:
        
        """
        pass
    
    def dataframes(self):
        return (
            name, dataset.dataframe()
            for name, dataset in self.datasets.items()
        )
    
    def numpy(self):
        return (
            name, dataset.inputs, dataset.outputs
            for name, dataset in self.datasets.items()
        )
