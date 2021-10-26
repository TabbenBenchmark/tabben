"""
Script to convert HIGGS CSV data to standardized PyTorch data.

Note: this script assumes that the host computer has sufficient memory to load
all of the HIGGS data into memory.
"""

import argparse
import os
import pandas as pd
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Convert HIGGS CSV to PyTorch')
    parser.add_argument('inputfile', help='path to source CSV file (can be compressed)')
    parser.add_argument('outputdirectory', help='directory to place output')

    return parser.parse_args()


def convert_format(config):
    # load full CSV file into pandas
    df = pd.read_csv(config.inputfile, header=None, index_col=None, dtype={0: int})

    # rename columns
    df.columns = [f'A{cname}' if cname != 0 else 'label' for cname in df.columns]
    
    # split into their standard train/test sets
    train_df = df[:-500_000]
    test_df = df[-500_000:]

    train_df_X = train_df.drop('label', axis=1)
    train_df_y = train_df['label']
    test_df_X = test_df.drop('label', axis=1)
    test_df_y = test_df['label']

    # convert to torch tensors and save to files
    train_X = torch.tensor(train_df_X.values)
    train_y = torch.tensor(train_df_y.values)
    test_X = torch.tensor(test_df_X.values)
    test_y = torch.tensor(test_df_y.values)

    torch.save(train_X, os.path.join(config.outputdirectory, 'train_data.pt'))
    torch.save(train_y, os.path.join(config.outputdirectory, 'train_labels.pt'))
    torch.save(test_X, os.path.join(config.outputdirectory, 'test_data.pt'))
    torch.save(test_y, os.path.join(config.outputdirectory, 'test_labels.pt'))


if __name__ == '__main__':
    config = parse_args()
    convert_format(config)


