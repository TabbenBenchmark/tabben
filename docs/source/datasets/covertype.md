
# Forest Cover Type Dataset

The forest cover type dataset is a typical multiclass classification dataset with a mix of categorical and continuous attributes.

You can read more about the description of this dataset from its [UCI ML Repo page](https://archive.ics.uci.edu/ml/datasets/covertype).

```{admonition} This dataset has a validation split
:class: note

While most of our included datasets have only a train-test split, this dataset has a validation split (`valid`). Use of this split is not required, and it can be included as part of your training data.
```

## Data preprocessing

The 'train', 'valid', and 'test' splits are as they are provided in the original dataset.

We don't do any further preprocessing of the data. In fact, the "soil type" attributes are one-hot encoded in the original dataset, and we keep that layout in our version (in most other datasets, they're kept as the single categorical/ordinal format).
