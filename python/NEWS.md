# `tabben` v0.0.3

## New Features
- new dataset: adult income prediction (#2)
- `get_metrics` function
  - standard set of evaluation metrics available via `tabben.evaluators` (currently, only works for binary classification tasks)
  - same set of available evaluation metrics compatible with autogluon (optional dependency) in `tabben.autogluon`

## Bugfixes
- fixed critical bug that limits the length of Datasets to the number of input attributes

## Breaking Changes


## Non-Breaking Changes


## Non-Code Updates
- quick guide for working with autogluon added (notebook + Markdown)
