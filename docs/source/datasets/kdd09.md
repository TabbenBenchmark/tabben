
# KDD Cup 2009 Dataset

Our `kdd09` dataset is the full set of datasets (large version) used for the 2009 KDD Cup
(appetency, churn, upselling).

You can read more of the description of the dataset from its [KDD Cup archive page](https://kdd.org/kdd-cup/view/kdd-cup-2009/Intro).

## Data Preprocessing

For the train-test split, we take the original use the original train subset and 
perform an 80-20 train-test split of the dataset.

Categorical attributes are mapping from symbols/strings to ordinal numbers. 
Missing values are encoded as NaNs.
We don't do any further preprocessing of the data.
