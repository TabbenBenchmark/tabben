"""
Script to convert HIGGS CSV data to standardized PyTorch data. The file should be 
already downloaded from the website (should work by downloading automatically, but
it's a large file).

Note: this script assumes that the host computer has sufficient memory to load
all of the HIGGS data into memory comfortably.
"""

import os
import pandas as pd

from utils import save_to_numpy_array, default_arg_parser

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

    # rename columns
    df.columns = [f'A{cname}' if cname != 0 else 'label' for cname in df.columns]
    
    # split into their standard train/test sets
    train_df = df[:-500_000]
    test_df = df[-500_000:]

    save_to_numpy_array(os.path.join(config.outputdirectory, 'higgs'), {
        'train-data': train_df.drop('label', axis=1),
        'train-labels': train_df['label'],
        'test-data': test_df.drop('label', axis=1),
        'test-labels': test_df['label'],
    })


if __name__ == '__main__':
    config = default_arg_parser(
            source_default='https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz',
    ).parse_args()
    convert_format(config)


