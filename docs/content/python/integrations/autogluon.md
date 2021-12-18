+++
title = "Working with Autogluon"
+++

This guide goes through how to use this package with autogluon hyperparameter tuning package.

## Load the train and test datasets

First, we'll go ahead and grab the train and test sets for the [arcene](https://archive.ics.uci.edu/ml/datasets/Arcene) data set using the `tabben` package.

```python
from tabben.datasets import OpenTabularDataset

train_ds = OpenTabularDataset('./data/', 'arcene')
test_ds = OpenTabularDataset('./data/', 'arcene', split='test')
```

This dataset has a large number of features, some of which are intentionally meaningless. (The attributes are not assigned to meaningful concepts either.)

```python
print(f'Number of Attributes: {train_ds.num_inputs}')
print(f'Attributes: {train_ds.input_attributes}')
```

For this dataset, we can get the metric functions that we should use (for consistency across everyone's runs) for evaluating on the test set. Autogluon will only use 1 metric (that it tests on its validation data set), so we just choose one of them.

```python
from tabben.autogluon import get_metrics

eval_metrics = get_metrics(train_ds.task, classes=train_ds.num_classes)

print(eval_metrics)
```

## Train the set of models


Now we can use [autogluon](https://auto.gluon.ai/stable/index.html) to automatically train a large set of different models and evaluate on all of them. We'll use the `TabularPredictor` class from autogluon.

```python
from autogluon.tabular import TabularPredictor

predictor = TabularPredictor(
    eval_metric=eval_metrics[0],
    label=train_ds.output_attributes[0], 
    path='ag-covertype')
```

```python
predictor.fit(
    train_ds.dataframe().head(300),  # artificially reduce the size of the dataset for faster demo
    presets='medium_quality_faster_train')
```

We can check to make sure that autogluon inferred the correct task (binary classification for this dataset).

```python
print(predictor.problem_type)
print(predictor.feature_metadata)
```

## Evaluate the model


Now, we're ready to evaluate our dataset. We can evaluate using autogluon's `leaderboard` method and supply our extra metrics that we want to compare by.

```python
X_test = test_ds.dataframe().drop(columns=test_ds.output_attributes)

y_pred = predictor.predict(X_test)
```

```python
predictor.leaderboard(test_ds.dataframe(), silent=True, extra_metrics=eval_metrics[1:])
```

(If you're looking at the leaderboard in the notebook, the 'score_test' column represents the auroc metric that was passed to the `TabularPredictor` constructor.)


---

This code was last run with the following versions (if you're looking at the no-output webpage, see the notebook in the repository for versions):

```python
from importlib.metadata import version

packages = ['autogluon', 'tabben']

for pkg in packages:
    print(f'{pkg}: {version(pkg)}')
```
