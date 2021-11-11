import argparse
from pathlib import Path

import numpy as np
import os
import pandas as pd


def create_csv_reader(*roots, **configs):
    def reader(filename, use_roots=True):
        filepath = os.path.join(*roots, filename) if use_roots else filename
        return pd.read_csv(filepath, **configs)
    
    return reader


def default_arg_parser(*,
                       description='Download and convert files into a standard format',
                       source_default='./'):
    
    parser = argparse.ArgumentParser(description=description)
    
    parser.add_argument('outputdirectory', help='directory to save outputs')
    parser.add_argument('--source', '-s',
                        help='root of the source urls to download from, or the directory of source files',
                        default=source_default)
    
    return parser


def column_name_array(df):
    return np.array(df.columns, dtype=np.str_)


def save_to_numpy_array(filename, df_dict):
    filename = Path(filename).with_suffix('.npz')
    
    arr_dict = {
        key: value.to_numpy().squeeze() if not isinstance(value, np.ndarray) else value
        for key, value in df_dict.items()
    }
    np.savez_compressed(str(filename), **arr_dict)
    print(f'Data saved in NPZ format at `{filename}`')


def split_by_label(df, col_name='label'):
    return df.drop(col_name, axis=1), df[[col_name]]

