import os
from pathlib import Path
from typing import Union

PathLike = Union[str, bytes, os.PathLike, Path]


def has_package_installed(*package_names: str):
    """Detect whether all the specified packages are installed."""
    
    import importlib
    return all(importlib.util.find_spec(package_name) for package_name in package_names)


def google_drive_download_link(identifier: str):
    """Convert a Google Drive document id to a direct download link."""
    
    return f'https://drive.google.com/uc?id={identifier}&export=download'

