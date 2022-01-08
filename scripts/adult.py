"""
Convert the adult census income prediction dataset to the standardized format.
"""

import pandas as pd

from utils import column_name_array, convert_categorical, create_csv_reader, default_config, \
    generate_profile, save_json, save_npz, split_by_label, uci_bibtex

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


def convert_format(config):
    read_csv = create_csv_reader(
        config.source,
        header=None,
        index_col=None,
        names=column_names,
        sep=',',
        skipinitialspace=True,
    )
    
    if config.download_sources:
        pass
    
    combined_df = pd.concat(
        [
            read_csv('adult.data'),
            read_csv('adult.test', skiprows=1),
        ], sort=False
    )
    
    combined_df['income'] = combined_df['income'].map(lambda i: i.strip('.'))
    
    combined_df[categorical_column_names] = combined_df[categorical_column_names].astype('category')
    categories = convert_categorical(combined_df)
    
    if config.dataset_file:
        train_df = combined_df.iloc[:32_561]
        test_df = combined_df.iloc[32_561:]
        train_data_df, train_labels_df = split_by_label(train_df, 'income')
        test_data_df, test_labels_df = split_by_label(test_df, 'income')
        
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
                'profile': generate_profile(combined_df),
                'bibtex': uci_bibtex,
                'categories': categories,
            }
        )


if __name__ == '__main__':
    args = default_config(
        'adult',
        download_root='https://archive.ics.uci.edu/ml/machine-learning-databases/adult/',
    )
    
    convert_format(args)
