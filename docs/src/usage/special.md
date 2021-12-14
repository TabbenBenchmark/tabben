# How to Handle Special Cases

## Downloading all (or some) of the datasets explicitly

This is useful if you're trying to import the package into an environment that is usually without an Internet connection. In this case, you can make sure that the datasets that you want are downloaded to your data directory using the `ensure_downloaded` function.

```python
from tabben.datasets import ensure_downloaded

ensure_downloaded('path/to/directory', 
                  'arcene', 
                  'higgs')
```

Note that this doesn't handle the CIFAR10 dataset automatically (yet).
