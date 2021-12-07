import os

from utils import save_to_numpy_array, create_csv_reader, default_arg_parser, column_name_array, \
    split_by_label


def convert_format(config):
    read_csv = create_csv_reader(
            config.source,
            header=0,
            index_col=None,
            usecols=range(4, 22),
            sep=',',
            skipinitialspace=True,
    )
    
    df = read_csv('parkinsons_updrs.data')
    df = df.sample(frac=1, random_state=171_234)  # shuffle rows
    
    train_examples = int(0.80 * df.shape[0])
    train_df = df.iloc[:train_examples]
    test_df = df.iloc[train_examples:]
    
    input_columns = df.columns[2:]
    output_columns = df.columns[:2]
    
    train_data_df, train_labels_df = train_df[input_columns], train_df[output_columns]
    test_data_df, test_labels_df = test_df[input_columns], test_df[output_columns]
    
    save_to_numpy_array(os.path.join(config.outputdirectory, 'parkinsons'), {
        'train-data'     : train_data_df,
        'train-labels'   : train_labels_df,
        'test-data'      : test_data_df,
        'test-labels'    : test_labels_df,
        '_columns-data'  : column_name_array(train_data_df),
        '_columns-labels': column_name_array(train_labels_df),
    })


if __name__ == '__main__':
    args = default_arg_parser(
            source_default='https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/',
    ).parse_args()
    
    convert_format(args)

