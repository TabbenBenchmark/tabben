# `tabben` v0.0.7

## New Features
- added the CIFAR10 dataset as a standardized dataset (i.e. can be accessed directly using OpenTabularDataset)
- added a DatasetCollection class for bulk processing of datasets

## Breaking Changes
- removed the `TabularCIFAR10Dataset` as it is no longer needed
- categorical columns for all datasets start counting at 0

## Non-Breaking Changes
- many datasets have additional "extras" (every current dataset has extras) available, such as
  - training data profiles (and full-data profiles that should *not* be used for model selection)
  - bibtex, licenses when available
- as a result of the many changes to several datasets, the version for all datasets has been incremented
  - version numbers are now 3 element integer arrays

## Bugfixes


## Non-Code Updates
- the documentation website is now generated with sphinx, and has API reference info as well
- there is now also a Julia package [Tabben.jl](https://umd-otb.github.io/Tabben.jl/latest/) for loading datasets and evaluating models (not at feature parity yet)
