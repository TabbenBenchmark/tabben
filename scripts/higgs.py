"""
Convert HIGGS CSV data to the standardized format. The file should be
already downloaded from the website (should work by downloading automatically, but
it's a large uncompressed text file and sometimes has failed).

Note: this script assumes that the host computer has sufficient memory to load
all the HIGGS data into memory *comfortably*.
"""

import os

import numpy as np
import pandas as pd

from utils import column_name_array, default_config, generate_profile, save_json, save_npz, \
    split_by_label, uci_license

bibtex = """\
@article{baldiSearchingExoticParticles2014,
  title = {Searching for Exotic Particles in High-Energy Physics with Deep Learning},
  author = {Baldi, P. and Sadowski, P. and Whiteson, D.},
  date = {2014-07-02},
  journaltitle = {Nature Communications},
  shortjournal = {Nat Commun},
  volume = {5},
  number = {1},
  pages = {4308},
  publisher = {{Nature Publishing Group}},
  issn = {2041-1723},
  doi = {10.1038/ncomms5308},
  url = {https://www.nature.com/articles/ncomms5308},
  urldate = {2021-12-13},
  issue = {1},
}"""

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
        os.path.join(config.source, 'HIGGS.csv.gz'),
        header=None,
        index_col=None,
        names=column_names
    )

    # split into their standard train/test sets
    train_df = df[:-500_000]
    test_df = df[-500_000:]
    
    if config.dataset_file:
        train_data_df, train_labels_df = split_by_label(train_df)
        test_data_df, test_labels_df = split_by_label(test_df)
        
        save_npz(
            config,
            {
                'train-data': train_data_df,
                'train-labels': train_labels_df,
                'test-data': test_data_df,
                'test-labels': test_labels_df,
                '_columns-data': column_name_array(train_data_df),
                '_columns-labels': np.array(['label'], dtype=np.str_),
            }
        )
    
    if config.extras_file:
        save_json(
            config,
            {
                'train-profile': generate_profile(train_df, config.no_profile),
                'profile': generate_profile(df, config.no_profile),
                'bibtex': bibtex,
                'license': uci_license,
                'categories': {
                    'label': ['background', 'signal'],
                },
                'column-names-attributes': column_names[1:],
                'column-names-target': column_names[:1],
            }
        )


if __name__ == '__main__':
    config = default_config(
        'higgs',
        download_root='https://archive.ics.uci.edu/ml/machine-learning-databases/00280/',
    )
    convert_format(config)
