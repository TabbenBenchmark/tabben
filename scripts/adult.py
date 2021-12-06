import os

import pandas as pd
import simplejson as json

from utils import save_to_numpy_array, create_csv_reader, default_arg_parser, column_name_array, \
    split_by_label

column_names = [
    'age',
    'workclass',  # categorical
    'fnlwgt',
    'education',  # categorical
    'education-num',
    'marital-status',  # categorical
    'occupation',  # categorical
    'relationship',  # categorical
    'race',  # categorical
    'sex',  # categorical
    'capital-gain',
    'capital-loss',
    'hours-per-week',
    'native-country',  # categorical
    'income',  # categorical
]

categorical_indices = [1, 3, 5, 6, 7, 8, 9, 13, 14]
categorical_column_names = [column_names[idx] for idx in categorical_indices]

def convert_categorical(df):
    categorical_columns = df.select_dtypes(['category']).columns
    
    categories = {}
    
    for column in categorical_columns:
        categories[column] = list(df[column].cat.categories)
    
    df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
    
    return categories


def convert_format(config):
    read_csv = create_csv_reader(
            config.source,
            header=None,
            index_col=None,
            names=column_names,
            sep=',',
            skipinitialspace=True,
    )
    
    combined_df = pd.concat([
        read_csv('adult.data'),
        read_csv('adult.test', skiprows=1),
    ], sort=False)
    
    combined_df['income'] = combined_df['income'].map(lambda i: i.strip('.'))
    
    combined_df[categorical_column_names] = combined_df[categorical_column_names].astype('category')
    categories = convert_categorical(combined_df)
    
    train_df = combined_df.iloc[:32_561]
    test_df = combined_df.iloc[32_561:]
    train_data_df, train_labels_df = split_by_label(train_df, 'income')
    test_data_df, test_labels_df = split_by_label(test_df, 'income')

    save_to_numpy_array(os.path.join(config.outputdirectory, 'adult'), {
        'train-data'     : train_data_df,
        'train-labels'   : train_labels_df,
        'test-data'      : test_data_df,
        'test-labels'    : test_labels_df,
        '_columns-data'  : column_name_array(train_data_df),
        '_columns-labels': column_name_array(train_labels_df),
    })
    
    with open(os.path.join(config.outputdirectory, 'adult.json'), 'w') as f:
        json.dump({
            'categories': categories,
        }, f)


if __name__ == '__main__':
    args = default_arg_parser(
            source_default='https://archive.ics.uci.edu/ml/machine-learning-databases/adult/',
    ).parse_args()
    
    convert_format(args)

