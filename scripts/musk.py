"""
Convert the musk (version 2) molecule/conformation classification dataset.
"""

import os

from sklearn.model_selection import train_test_split

from utils import column_name_array, create_csv_reader, default_config, save_to_numpy_array, \
    split_by_label

column_names = [
    'molecule_name',  # NOT used as input
    'conformation_name',  # NOT used as input
    *[f'f{num}' for num in range(1, 166 + 1)],
    'musk',
]


def convert_format(config):
    read_csv = create_csv_reader(
        config.source,
        header=None,
        index_col=None,
        names=column_names,
        usecols=column_names[2:],
        sep=',',
    )
    
    df = read_csv('clean2.data')
    train_df, test_df = train_test_split(df, train_size=0.8, random_state=171_234)
    
    train_data_df, train_labels_df = split_by_label(train_df, 'musk')
    test_data_df, test_labels_df = split_by_label(test_df, 'musk')
    
    save_to_numpy_array(
        os.path.join(config.outputdirectory, 'musk'), {
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
        source_default='https://archive.ics.uci.edu/ml/machine-learning-databases/musk/',
    )
    
    convert_format(args)