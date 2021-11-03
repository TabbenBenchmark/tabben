import os
import pandas as pd

from utils import save_to_numpy_array, default_arg_parser, split_by_label


def convert_format(config):
    df = pd.read_csv(
            os.path.join(config.source, 'covtype.data.gz'),
            header=None,
            index_col=None,
    )
    
    # TODO rename to slightly more descriptive column names?
    df.columns = [f'A{cname}' if cname != len(df.columns) - 1 else 'label' for cname in df.columns]
    
    train_data_df, train_labels_df = split_by_label(df[:11_340])
    valid_data_df, valid_labels_df = split_by_label(df[11_340:11_340+3_780])
    test_data_df, test_labels_df = split_by_label(df[-565_892:])
    
    save_to_numpy_array(os.path.join(config.outputdirectory, 'covertype'), {
        'train-data': train_data_df,
        'train-labels': train_labels_df,
        'valid-data': valid_data_df,
        'valid-labels': valid_labels_df,
        'test-data': test_data_df,
        'test-labels': test_labels_df,
    })


if __name__ == '__main__':
    args = default_arg_parser(
            source_default='https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/',
    ).parse_args()
    
    convert_format(args)

