# `tabben` v0.0.5

## New Features
- new datasets:
  - [musk (version 2)](https://archive.ics.uci.edu/ml/datasets/Musk+(Version+2))
  - [rossman store sales prediction](https://www.kaggle.com/c/rossmann-store-sales/data?select=sample_submission.csv) (this dataset is somewhat preprocessed, splits are between stores, and the store id is removed)
  - [amazon employee access](https://www.kaggle.com/c/amazon-employee-access-challenge/overview) (note that the data for this dataset is mostly integer ids)

## Breaking Changes

## Non-Breaking Changes
- latest dataset files all have versions; versions are now required in the dataset files (will redownload if not present)
- update `parkinsons` dataset: different train/test split (splits by subject---no subject appears in both train and test)
- `covertype` labels now range from 0-6 (previously were 1-7, which can sometimes cause issues)

## Bugfixes

## Non-Code Updates

