# `tabben` v0.0.7

## New Features
- added the CIFAR10 dataset as a standardized dataset (i.e. can be accessed directly using OpenTabularDataset)
- added a DatasetCollection class for bulk processing of datasets

## Breaking Changes
- removed the `TabularCIFAR10Dataset` as it is no longer needed

## Non-Breaking Changes


## Bugfixes


## Non-Code Updates
- the documentation website is now generated with sphinx, and has API reference info as well
- there is now also a Julia package [Tabben.jl](https://umd-otb.github.io/Tabben.jl/latest/) for loading datasets and evaluating models (not at feature parity yet)
