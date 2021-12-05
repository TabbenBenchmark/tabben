# `tabben` v0.0.2

## New Features
- the location of the source for each dataset can be a file URI (e.g. `file:///abs/path/to/dataset/file.npz`); this is useful locally
- new function `validate_dataset_file` to verify that a dataset file is in the correct format and loadable
- additional `OpenTabularDataset` properties: `num_inputs`, `num_outputs`, `num_classes`, `task`

## Breaking Changes
- dataset file format
  - now includes a version number (versions for already downloaded files are checked and redownloaded if possible)
- the `valid` split for the 'arcene' dataset renamed to `test`
- `OpenTabularDataset`
  - now generates numpy arrays instead of torch tensors, if no transform is passed in (not breaking if PyTorch DataLoader is used)
  - `transform` argument of the constructor has been split into `transform` and `target_transform`

## Non-Breaking Changes


## Non-Code Updates
- additions of formal documentation
  - Markdown source files for basic and some advanced uses
  - synced, runnable Jupyter notebooks containing the same content
