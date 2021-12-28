# Data Preprocessing Scripts

This directory contains cli scripts for pre-processing the various datasets in the benchmark to a standard format.

The scripts use a standard calling convention:
```shell
python3 [script name] [destination directory] --source [source directory or url parent] 
```

For example, if the raw data is already downloaded in the `data` folder,
```shell
python3 arcene.py ./ --source ./data
```
However, there is also a convenience script that can run all or a subset of the dataset scripts:
```shell
python3 all.py -d data/cache/dir dataset names separated by spaces
```

## Data Format
Each dataset is an [NPZ (compressed NPZ)](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html#module-numpy.lib.format) file.
- `_columns-data` (1D np.ndarray with unicode dtype): input attribute names
- `_columns-labels` (1D np.ndarray with unicode dtype): 1D array of output attribute names
- `{split}-data` (2D np.ndarray with numeric dtype): input data, e.g. `train-data`
- `{split}-labels` (1D or 2D np.ndarray with numeric dtype): output labels, e.g. `train-labels`
- `_version`: version number for the data format (automatically added)

The standard split names are `train`, `valid`, and `test` (having a `valid` split is optional, but a `test` set is required).

Other metadata can be included in this file with keys that start with `_`, but it's probably better to use other formats for the majority of metadata (otherwise it requires having the data file in order to read metadata).

(Files should *not* require `pickle`.)

## Metadata (partial work in progress)
"Basic" metadata is stored in [data.toml](/python/src/tabben/datasets/data.toml), things like the task (e.g. classification), number of classes, number of outputs, etc.

Complex metadata (referred to as "extras") is saved in json format.

Fields that may be at the top-level of the json object:
- '`profile'`: result of running the dataprofiler (some datasets have not been run on this because of having too many attributes)
- `'categories'`: mappings for categorical variables
- `'license'`: licensing information for the dataset
- `'bibtex'`: bibtex string (if available) that should be cited if the dataset is used for a research publication

## Data Processing Checklist
- [ ] labels for classification problems
  - binary: "positive" class should be `1`, "negative" class should be `0`
  - multiclass: *not* one-hot encoded, consecutive integers starting at `0`
- [ ] conform to a standard dtype? (not done currently)
- [ ] produce unicode np arrays of the attribute names (one for the input attributes and one for the output attributes, see [format](#Data Format))
  - if no meaningful names, can just use numeric names (e.g. `'0'`, `'1'`, ... or `'A1'`, `'A2'`, ...)
  - prefer `label` for the output attribute name (not as important)
- [ ] form numpy arrays for each training split based on [the above format](#Data Format)
- [ ] generate metadata
- [ ] upload data files and add entry to [data.toml](/python/src/tabben/datasets/data.toml) (can use `tabben.datasets.register_dataset(..., persist=True, ...)`)
- [ ] if not excessively large dataset, add unit tests

