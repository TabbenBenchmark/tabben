
# Amazon Employee Access Dataset

This dataset is a set of categorical characteristics about each employee and their role at the company.
The goal is to be able to predict whether they will be granted access to a resource.

You can read more about the dataset from its [Kaggle competition page](https://www.kaggle.com/c/amazon-employee-access-challenge/data) (Kaggle login required).

## Data Preprocessing

For the train-test split, we split the original train set into an 80-20 train-test split
stratified by the target class (the action taken: `approved` or `denied`).

We don't do any further preprocessing of the data.

