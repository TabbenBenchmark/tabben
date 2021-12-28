"""
Convert the source files for the Parkinson's Tele-monitoring dataset (available
from UCI Dataset Archive).
"""

import os

from sklearn.model_selection import train_test_split

from utils import column_name_array, create_csv_reader, default_config, save_npz


def convert_format(config):
    read_csv = create_csv_reader(
        config.source,
        header=0,
        index_col=None,
        sep=',',
        skipinitialspace=True,
    )
    
    df = read_csv('parkinsons_updrs.data')
    df = df.sample(frac=1, random_state=171_234)  # shuffle rows
    
    train_subjects, test_subjects = train_test_split(
        df['subject#'].unique(), train_size=0.8,
        random_state=171_234, )
    
    train_df = df[df['subject#'].isin(train_subjects)]
    test_df = df[df['subject#'].isin(test_subjects)]
    
    input_columns = df.columns[6:]
    output_columns = df.columns[4:5 + 1]
    
    train_data_df, train_labels_df = train_df[input_columns], train_df[output_columns]
    test_data_df, test_labels_df = test_df[input_columns], test_df[output_columns]
    
    save_npz(
        os.path.join(config.outputdirectory, 'parkinsons'), {
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
        source_default='https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/',
    )
    
    convert_format(args)
