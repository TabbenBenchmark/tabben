
# Special Situations

## Downloading all (or some) of the datasets explicitly

This is useful if you're trying to import the package into an environment that is usually without an Internet connection. In this case, you can make sure that the datasets that you want are downloaded to your data directory using the `ensure_downloaded` function.

```python
from tabben.datasets import ensure_downloaded

# download just the arcene and higgs datasets
ensure_downloaded('path/to/directory', 
                  'arcene', 
                  'higgs')

# download all available datasets
ensure_downloaded('path/to/directory')
```

