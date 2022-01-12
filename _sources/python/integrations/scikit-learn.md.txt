# Working with Scikit-Learn

This guide goes through how to use this package with the Scikit-Learn package.


## Load the train and test datasets

We'll first get the train and test splits for the `musk` dataset (completely unrelated to Elon Musk).

```python
from tabben.datasets import OpenTabularDataset

train = OpenTabularDataset('./temp', 'musk')  # train split by default
test = OpenTabularDataset('./temp', 'musk', split='test')  # should only be used ONCE!

print(f'The {train.name} dataset is a {train.task} task with {train.num_classes} classes.')
```

```python
X_fulltrain, y_fulltrain = train.numpy()
```

In order to tune some hyperparameters, we'll need our own validation split (not the test set). We'll do an 80-20 split and stratify on the class.

```python
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(
    X_fulltrain, y_fulltrain, 
    train_size=0.8, 
    stratify=y_fulltrain
)
```

## Create and train a model

Next, we'll create a $k$-Nearest Neighbors model and train it on our train split.

```python
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier()
```

```python
model.fit(X_train, y_train)
```

And we'll evaluate it on our *validation* set, using a simple accuracy metric.

```python
model.score(X_valid, y_valid)
```

## In a larger data processing pipeline

However, it might be the case that we want to use a sklearn pipeline to do some data preprocessing like feature normalization, one-hot encoding, etc. or explore the effect of, say, turning continuous attributes into binary ones.

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Binarizer, StandardScaler

pipeline = make_pipeline(
    StandardScaler(with_std=False),
    Binarizer(),
    KNeighborsClassifier(),
)
```

```python
pipeline.fit(X_train, y_train)
```

```python
pipeline.score(X_valid, y_valid)
```

---

This code was last run using the following package versions (if you're looking at the webpage which doesn't have the output, see the notebook for versions):

```python
from importlib.metadata import version

packages = ['scikit-learn', 'tabben']

for pkg in packages:
    print(f'{pkg}: {version(pkg)}')
```
