+++
title = "Working with Scikit-Learn"
+++

This guide goes through how to use this package with the Scikit-Learn package.

[Work in progress]


## Load the train and test datasets

We'll first get the train and test splits for the `musk` dataset (completely unrelated to Elon Musk).

```python
from tabben.datasets import OpenTabularDataset

train = OpenTabularDataset('./temp', 'musk')  # train split by default
test = OpenTabularDataset('./temp', 'musk', split='test')  # should only be used ONCE!

print(f'The {train.name} dataset is a {train.task} task with {train.num_classes} classes.')
```

## Create and train a model

Next, we'll create a $k$-Nearest Neighbors model and train it on our train split.

```python
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier()
```

```python
model.fit()
```
