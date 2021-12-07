"""
A set of utility functions related to loading datasets.
"""

import os
from typing import Union

PathLike = Union[str, bytes, os.PathLike]


def has_package_installed(*package_names: str):
    """Detect whether all the specified packages are installed."""
    
    import importlib
    return all(importlib.util.find_spec(package_name) for package_name in package_names)

