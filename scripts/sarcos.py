"""


Note: this dataset should be downloaded locally first (don't supply a URL source).
"""

import os
from scipy.io import loadmat

from utils import save_to_numpy_array, create_csv_reader, default_arg_parser, split_by_label


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
    })


if __name__ == '__main__':
    args = default_arg_parser().parse_args()
    
    convert_format(args)

