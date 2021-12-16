"""
Set of utilities for creating standard dataset files for this benchmark.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dataprofiler import Profiler

try:
    import simplejson as json
except ImportError:
    import json


def create_csv_reader(*roots, **configs):
    def reader(filename, use_roots=True, **kwargs):
        filepath = os.path.join(*roots, filename) if use_roots else filename
        return pd.read_csv(filepath, **{**configs, **kwargs})
    
    return reader


def default_config(*,
                   description='Download and convert files into a standard format',
                   source_default='./'):
    parser = argparse.ArgumentParser(description=description)
    
    parser.add_argument('outputdirectory', help='directory to save outputs')
    parser.add_argument(
        '--source', '-s',
        help='root of the source urls to download from, or the directory of source files',
        default=source_default
    )
    
    parser.add_argument('--dataset-file', action='store_true',
                        help='Generate the NPZ dataset file')
    parser.add_argument('--extras-file', action='store_true',
                        help='Generate the metadata data if needed')
    
    parser.add_argument('--no-profile', action='store_true')
    
    args = parser.parse_args()
    
    if not any([args.dataset_file, args.metadata_file]):
        args.dataset_file = True
        args.metadata_file = True
    
    return args


def column_name_array(df):
    return np.array(df.columns, dtype=np.str_)


def hvcat(arrays):
    return pd.concat(
        [
            pd.concat(row, axis=1) for row in arrays
        ], axis=0
    )


def save_to_numpy_array(filename, df_dict):
    filename = Path(filename).with_suffix('.npz')
    
    arr_dict = {
        key: value.to_numpy().squeeze() if not isinstance(value, np.ndarray) else value
        for key, value in df_dict.items()
    }
    arr_dict['_version'] = '1.0.0'
    
    np.savez_compressed(str(filename), **arr_dict)
    print(f'Data saved in NPZ format at `{filename}`')


def split_by_label(df, col_name='label'):
    if isinstance(col_name, str):
        col_name = [col_name]
    
    return df.drop(col_name, axis=1), df[col_name]


class JSONNumpyEncoder(json.JSONEncoder):
    
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        
        return json.JSONEncoder.default(self, obj)


def save_json(data, filename):
    with Path(filename).open('w') as f:
        json.dump(data, f, cls=JSONNumpyEncoder)


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