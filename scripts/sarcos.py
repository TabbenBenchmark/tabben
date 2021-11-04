"""


Note: this dataset should be downloaded locally first (don't supply a URL source).
"""

import os

import numpy as np
from scipy.io import loadmat

from utils import save_to_numpy_array, create_csv_reader, default_arg_parser, split_by_label

input_column_names = [
    *[f'P{num}' for num in range(1, 7+1)],
    *[f'V{num}' for num in range(1, 7+1)],
    *[f'A{num}' for num in range(1, 7+1)],
]

output_column_names = [f'T{num}' for num in range(1, 7+1)]


def input_output_split(mat):
    return mat[:, :21], mat[:, 21:]


def convert_format(config):
    train_mat = loadmat(os.path.join(config.source, 'sarcos_inv.mat'))['sarcos_inv']
    test_mat = loadmat(os.path.join(config.source, 'sarcos_inv_test.mat'))['sarcos_inv_test']
    
    train_data, train_labels = input_output_split(train_mat)
    test_data, test_labels = input_output_split(test_mat)

    save_to_numpy_array(os.path.join(config.outputdirectory, 'sarcos'), {
        'train-data'  : train_data,
        'train-labels': train_labels,
        'test-data'   : test_data,
        'test-labels' : test_labels,
        '_columns-data': np.array(input_column_names, dtype=np.str_),
        '_columns-labels': np.array(output_column_names, dtype=np.str_),
    })


if __name__ == '__main__':
    args = default_arg_parser().parse_args()
    
    convert_format(args)

