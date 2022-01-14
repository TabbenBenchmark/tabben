
# Rossman Store Sales Dataset

This dataset is a set of categorical characteristics about each employee and their role at the company.
The goal is to be able to predict whether they will be granted access to a resource.

You can read more about the dataset from its [Kaggle competition page](https://www.kaggle.com/c/rossmann-store-sales/data) (Kaggle login required).

## Data Preprocessing

For the train-test split, we split the original train set into an 80-20 train-test split
such that stores only ever appear in one of the sets.

```{admonition} Train-test split
:class: caution

We may reconsider how the train-test split is done in a future update to also/instead
split the data by transaction time.
```

We also join the store information table into the main dataset 
(which is the same for all transactions occurring at the same store).
The store table also includes a large number of missing values, encoded as NaNs.
Doing this _excludes_ the store id from the final set of input attributes.
