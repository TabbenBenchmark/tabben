"""
Set of utilities for creating standard dataset files for this benchmark.
"""

import argparse
import contextlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import tarfile
from dataprofiler import Profiler

try:
    import simplejson as json
except ImportError:
    import json

uci_bibtex = """\
@misc{Dua:2019 ,
    author = "Dua, Dheeru and Graff, Casey",
    year = "2017",
    title = "{UCI} Machine Learning Repository",
    url = "http://archive.ics.uci.edu/ml",
    institution = "University of California, Irvine, School of Information and Computer Sciences"
}"""


def create_csv_reader(root, **configs):
    def reader(filename, use_root=True, **kwargs):
        filepath = root / filename if use_root else filename
        return pd.read_csv(filepath, **{**configs, **kwargs})
    
    return reader


def default_config(name, *,
                   description='Download or convert files into a standard format',
                   download_root=None):
    parser = argparse.ArgumentParser(description=description)
    
    parser.add_argument(
        'outputdirectory', dest='output_directory', type=Path,
        help='local directory to save outputs'
    )
    parser.add_argument(
        '--source', '-s', default=None, type=Path,
        help='local directory of (cached) source files',
    )
    parser.add_argument(
        '--download-sources', '-ds', action='store_true',
        help='Download files from the original source if possible',
    )
    
    parser.add_argument(
        '--dataset-file', action='store_true',
        help='Generate the NPZ dataset file',
    )
    parser.add_argument(
        '--extras-file', action='store_true',
        help='Generate the metadata data if needed',
    )
    parser.add_argument(
        '--no-profile', action='store_true',
        help='Do not generate the profile',
    )
    
    args = parser.parse_args()
    args.name = name
    
    args.download_root = download_root
    if download_root is None:
        args.download_sources = False
    
    if not any([args.dataset_file, args.extras_file]):
        args.dataset_file = True
        args.extras_file = True
    
    return args


def column_name_array(df):
    return np.array(df.columns, dtype=np.str_)


def hvcat(arrays):
    return pd.concat(
        [
            pd.concat(row, axis=1) for row in arrays
        ], axis=0
    )


def save_npz(config, df_dict):
    filename = (config.output_directory / config.name).with_suffix('.npz')
    comp_filename = filename.with_suffix('.npz.tar.gz')
    
    arr_dict = {
        key: value.to_numpy().squeeze() if not isinstance(value, np.ndarray) else value
        for key, value in df_dict.items()
    }
    arr_dict['_version'] = '1.0.0'
    
    np.savez_compressed(str(filename), **arr_dict)
    print(f'\u001b[32mData saved in NPZ format at `{filename}`\u001b[0m')
    
    with tarfile.open(comp_filename, 'w:gz') as f:
        f.add(filename, arcname=filename.name)
    print(f'\u001b[32mData tarred and gzipped at `{comp_filename}`\u001b[0m')


def split_by_label(df, col_name='label'):
    if isinstance(col_name, str):
        col_name = [col_name]
    
    return df.drop(col_name, axis=1), df[col_name]


class JSONNumpyEncoder(json.JSONEncoder):
    
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        
        return json.JSONEncoder.default(self, obj)


def save_json(config, data):
    filename = (config.output_directory / config.name).with_suffix('.json')
    comp_filename = filename.with_suffix('.json.tar.gz')
    
    with filename.open('w') as f:
        json.dump(data, f, cls=JSONNumpyEncoder)
    
    print(f'\u001b[32mExtras saved in JSON format at `{filename}`\u001b[0m')
    
    with tarfile.open(comp_filename, 'w:gz') as f:
        f.add(filename, arcname=filename.name)
        
    print(f'\u001b[32mExtras tarred and gzipped at `{comp_filename}`\u001b[0m')


def generate_profile(df):
    profile = Profiler(df)
    return profile.report(report_options={"output_format": "serializable"})


def convert_categorical(df):
    categorical_columns = df.select_dtypes(['category']).columns
    
    categories = {}
    
    for column in categorical_columns:
        categories[column] = list(df[column].cat.categories)
    
    df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
    
    return categories


@contextlib.contextmanager
def stage(action):
    print(f'Begin {action}...', end='', flush=True)
    yield
    print('finished.')
    
