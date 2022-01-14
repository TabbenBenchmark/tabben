"""
Convert the poker hand classification dataset to the standardized format, reusing
its train-test split.
"""

import os

from utils import column_name_array, create_csv_reader, default_config, generate_profile, hvcat, save_json, \
    save_npz, \
    split_by_label, uci_bibtex, uci_license

column_names = [
    'S1',
    'C1',
    'S2',
    'C2',
    'S3',
    'C3',
    'S4',
    'C4',
    'S5',
    'C5',
    'hand',
]

suits = ['hearts', 'spades', 'diamonds', 'clubs']
ranks = ['ace', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'jack', 'queen', 'king']

categories = {
    'hand': ['nothing', 'one pair', 'two pairs', 'three of a kind', 'straight',
             'flush', 'full house', 'four of a kind', 'straight flush', 'royal flush'],
    **{f'S{c}': suits for c in range(1, 5+1)},
    **{f'C{c}': ranks for c in range(1, 5+1)},
}


def convert_format(config):
    read_csv = create_csv_reader(
        config.source,
        header=None,
        index_col=None,
        names=column_names
    )
    
    train_df = read_csv('poker-hand-training-true.data')
    test_df = read_csv('poker-hand-testing.data')
    
    train_df[column_names[:-1]] = train_df[column_names[:-1]] - 1
    test_df[column_names[:-1]] = test_df[column_names[:-1]] - 1
    
    train_data_df, train_labels_df = split_by_label(train_df, 'hand')
    test_data_df, test_labels_df = split_by_label(test_df, 'hand')
    
    if config.dataset_file:
        save_npz(
            config,
            {
                'train-data': train_data_df,
                'train-labels': train_labels_df,
                'test-data': test_data_df,
                'test-labels': test_labels_df,
                '_columns-data': column_name_array(train_data_df),
                '_columns-labels': column_name_array(train_labels_df),
            }
        )
    
    if config.extras_file:
        save_json(
            config,
            {
                'train-profile': generate_profile(train_df, config.no_profile),
                'profile': generate_profile(hvcat([[train_df], [test_df]]), config.no_profile),
                'categories': categories,
                'license': uci_license,
                'bibtex': uci_bibtex,
                'column-names-attributes': column_names[:-1],
                'column-names-target': column_names[-1:],
            }
        )


if __name__ == '__main__':
    args = default_config(
        'poker',
        download_root='https://archive.ics.uci.edu/ml/machine-learning-databases/poker',
    )
    
    convert_format(args)
