# Data Preprocessing Scripts

This directory contains cmdline scripts for pre-processing the various datasets in the benchmark to a standard format.

At least right now, the scripts use a (mostly) standard calling convention:
```shell
python3 [script name] [destination directory] --source [source directory or url parent] 
```

For example, if the raw data is already downloaded in the `data` folder,
```shell
python3 arcene.py ./ --source ./data
```

## Data Format
Each dataset is an [NPZ (compressed NPZ)](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html#module-numpy.lib.format) file.
- `_columns-data` (1D np.ndarray with unicode dtype): input attribute names
- `_columns-labels` (1D np.ndarray with unicode dtype): 1D array of output attribute names
- `{split}-data` (2D np.ndarray with numeric dtype): input data, e.g. `train-data`
- `{split}-labels` (1D or 2D np.ndarray with numeric dtype): output labels, e.g. `train-labels`

The standard split names are `train`, `valid`, and `test`.

Other metadata can be included in this file with keys that start with `_`, but it's probably better to use other formats for the majority of metadata.

(Files should not require `pickle`.)

## Metadata
Metadata is saved as json, starting with the output of the dataprofiler.

Additional fields that need/can be added at the top-level of the json object:
- task (required, used to determine appropriate metrics)
  - `binary-classification`
  - `multiclass-classification`
  - `regression`
- license
- bibtex

## Data Processing Checklist
- [ ] labels for classification problems
  - binary: positive class should be `1`, negative class should be `0`
  - multiclass: *not* one-hot encoded, consecutive integers starting at `0`
- [ ] conform to a standard dtype? (not done currently)
- [ ] produce unicode np arrays of the attribute names (one for the input attributes and one for the output attributes, see [format](#Data Format))
  - if no meaningful names, can just use numeric names (e.g. `'0'`, `'1'`)
  - prefer `label` for the output attribute name
- [ ] form numpy arrays for each training split based on [the above format](#Data Format)
- [ ] generate metadata
- [ ] upload data files to the public Google drive and put document ids in [data.toml](/python/src/otb/datasets/data.toml)
- [ ] if not excessively large dataset, add unit tests

