
# `tabben.datasets`

## Dataset Class

The basic unit of work for this package is the [OpenTabularDataset](tabben.datasets.OpenTabularDataset):

```{eval-rst}
.. autoclass:: tabben.datasets.OpenTabularDataset
    :show-inheritance:
    :members:
```

## Metadata

```{eval-rst}
.. autodata:: tabben.datasets.metadata
    :no-value:

.. autodata:: tabben.datasets.allowed_tasks
```

## Collection of Datasets 
```{eval-rst}
.. autoclass:: tabben.datasets.DatasetCollection
    :members:
```

## Other Utilities

```{eval-rst}
.. autofunction:: tabben.datasets.ensure_downloaded

.. autofunction:: tabben.datasets.register_dataset

.. autofunction:: tabben.datasets.validate_dataset_file

.. autoexception:: tabben.datasets.DatasetFormatError
```
