+++
title = 'Musk Dataset'
+++

The musk dataset is a chemical molecule dataset trying to classify whether certain molecules (using only measurements from a particular conformation) smell like [musk](https://en.wikipedia.org/wiki/Musk). (This is using version *2* of the dataset.)

You can read more of the description of the dataset from its [UCI ML Repo page](https://archive.ics.uci.edu/ml/datasets/Musk+(Version+2)).

## Data Preprocessing

The dataset contains measurements for multiple conformations of multiple molecules, where the identity of the molecule (not the conformation) determines whether it's musk or not. For the train-test split, we split the dataset into a custom 80-20 train-test split, with any given molecule occurring in *only* one split. The train and test splits also have approximately the same class imbalance.

We don't do any further preprocessing of the data itself.
