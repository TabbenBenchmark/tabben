"""
Script to convert HIGGS CSV data to standardized PyTorch data. The file should be 
already downloaded from the website (should work by downloading automatically, but
it's a large file).

Note: this script assumes that the host computer has sufficient memory to load
all of the HIGGS data into memory comfortably.
"""

import os
import pandas as pd

from utils import column_name_array, save_to_numpy_array, default_arg_parser

column_names = [
    'label',
    'lepton pT',
    'lepton eta',
    'lepton phi',
    'missing energy magnitude',
    'missing energy phi',
    'jet 1 pt',
    'jet 1 eta',
    'jet 1 phi',
    'jet 1 b-tag',
    'jet 2 pt',
    'jet 2 eta',
    'jet 2 phi',
    'jet 2 b-tag',
    'jet 3 pt',
    'jet 3 eta',
    'jet 3 phi',
    'jet 3 b-tag',
    'jet 4 pt',
    'jet 4 eta',
    'jet 4 phi',
    'jet 4 b-tag',
    'm_jj',
    'm_jjj',
    'm_lv',
    'm_jlv',
    'm_bb',
    'm_wbb',
    'm_wwbb',
]


def convert_format(config):
    # load full CSV file into pandas
    df = pd.read_csv(
            config.source,
            header=None,
            index_col=None,
            names=column_names)

    # split into their standard train/test sets
    train_df = df[:-500_000]
    test_df = df[-500_000:]
    
    train_data_df = train_df.drop('label', axis=1)
    train_label_df = train_df['label']
    test_data_df = test_df.drop('label', axis=1)
    test_label_df = test_df['label']

    save_to_numpy_array(os.path.join(config.outputdirectory, 'higgs'), {
        'train-data': train_data_df,
        'train-labels': train_label_df,
        'test-data': test_data_df,
        'test-labels': test_label_df,
        '_columns-data': column_name_array(train_data_df),
        '_columns-labels': column_name_array(train_label_df),
    })


if __name__ == '__main__':
    config = default_arg_parser(
            source_default='https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz',
    ).parse_args()
    convert_format(config)


