
# Parkinson's Telemonitoring Dataset

The parkinson's telemonitoring dataset contains measurement records from a telemonitoring device for several individuals at multiple time points. There are 2 regression tasks associated with this dataset.

You can read more of the description of the dataset from its [UCI ML Repo page](https://archive.ics.uci.edu/ml/datasets/parkinsons+telemonitoring).

## Data Preprocessing

As in the original setup, we include only the 16 measurements and *exclude* the instance-level data such as the subject # or timestamp. For the train-test split, we use a custom 80-20 train-test split of the original dataset, where individuals appear in only one of the splits.

We don't do any further preprocessing of the data.
