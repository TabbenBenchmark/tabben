"""
Convert the poker hand classification dataset to the standardized format, reusing
its train-test split.
"""

import os

from utils import column_name_array, create_csv_reader, default_config, save_to_numpy_array, \
    split_by_label

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
    'label',
]


def convert_format(config):
    read_csv = create_csv_reader(
        config.source,
        header=None,
        index_col=None,
        names=column_names
    )
    
    train_df = read_csv('poker-hand-training-true.data')
    test_df = read_csv('poker-hand-testing.data')
    
    train_data_df, train_labels_df = split_by_label(train_df)
    test_data_df, test_labels_df = split_by_label(test_df)
    
    save_to_numpy_array(
        os.path.join(config.outputdirectory, 'poker'), {
            'train-data': train_data_df,
            'train-labels': train_labels_df,
            'test-data': test_data_df,
            'test-labels': test_labels_df,
            '_columns-data': column_name_array(train_data_df),
            '_columns-labels': column_name_array(train_labels_df),
        }
    )


if __name__ == '__main__':
    args = default_config(
        source_default='https://archive.ics.uci.edu/ml/machine-learning-databases/poker',
    )
    
    convert_format(args)
