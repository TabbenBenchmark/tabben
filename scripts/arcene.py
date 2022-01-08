"""
Convert the arcene dataset to the standardized format.
"""

import os

from utils import column_name_array, create_csv_reader, default_config, save_npz


def convert_format(config):
    read_csv = create_csv_reader(
        config.source,
        header=None,
        index_col=None,
        sep='\\s+',
    )
    
    if config.download_sources:
        pass
    
    # labels for one of the splits is stored in the parent directory of where everything else is stored,
    # but this is just the organization on
    if config.source == 'https://archive.ics.uci.edu/ml/machine-learning-databases/arcene/ARCENE/':
        valid_labels_source = 'https://archive.ics.uci.edu/ml/machine-learning-databases/arcene' \
                              '/arcene_valid.labels'
    else:
        valid_labels_source = os.path.join(config.source, 'arcene_valid.labels')
    
    train_data_df = read_csv('arcene_train.data')
    train_labels_df = read_csv('arcene_train.labels', names=['label'])
    valid_data_df = read_csv('arcene_valid.data')
    valid_labels_df = read_csv(valid_labels_source, use_root=False, names=['label'])
    
    save_npz(
        config,
        {
            'train-data': train_data_df,
            'train-labels': train_labels_df['label'],
            'test-data': valid_data_df,
            'test-labels': valid_labels_df['label'],
            '_columns-data': column_name_array(train_data_df),
            '_columns-labels': column_name_array(train_labels_df),
        }
    )


if __name__ == '__main__':
    args = default_config(
        'arcene',
        download_root='https://archive.ics.uci.edu/ml/machine-learning-databases/arcene/ARCENE/',
    )
    
    convert_format(args)
