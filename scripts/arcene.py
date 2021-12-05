import os

from utils import save_to_numpy_array, create_csv_reader, default_arg_parser, column_name_array


def convert_format(config):
    read_csv = create_csv_reader(
            config.source,
            header=None,
            index_col=None,
            sep='\\s+',
    )
    
    # labels for one of the splits is stored in the parent directory of where everything else is stored
    if config.source == 'https://archive.ics.uci.edu/ml/machine-learning-databases/arcene/ARCENE/':
        valid_labels_source = 'https://archive.ics.uci.edu/ml/machine-learning-databases/arcene/arcene_valid.labels'
    else:
        valid_labels_source = os.path.join(config.source, 'arcene_valid.labels')
    
    train_data_df = read_csv('arcene_train.data')
    train_labels_df = read_csv('arcene_train.labels')
    valid_data_df = read_csv('arcene_valid.data')
    valid_labels_df = read_csv(valid_labels_source, use_roots=False)
    
    train_labels_df.columns = ['label']
    valid_labels_df.columns = ['label']
    
    save_to_numpy_array(os.path.join(config.outputdirectory, 'arcene'), {
        'train-data': train_data_df,
        'train-labels': train_labels_df['label'],
        'test-data': valid_data_df,
        'test-labels': valid_labels_df['label'],
        '_columns-data': column_name_array(train_data_df),
        '_columns-labels': column_name_array(train_labels_df),
    })


if __name__ == '__main__':
    args = default_arg_parser(
            source_default='https://archive.ics.uci.edu/ml/machine-learning-databases/arcene/ARCENE/',
    ).parse_args()
    
    convert_format(args)

