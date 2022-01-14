"""
Convert the forest cover type classification dataset to the standardized format.
"""

import os

import pandas as pd

from utils import column_name_array, convert_categorical, default_config, generate_profile, \
    hvcat, save_json, save_npz, split_by_label, uci_bibtex

column_names = [
    'Elevation',
    'Aspect',
    'Slope',
    'Horizontal_Distance_To_Hydrology',
    'Vertical_Distance_To_Hydrology',
    'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
    'Horizontal_Distance_To_Fire_Points',
    *[f'Wilderness_Area{rep}' for rep in range(1, 4 + 1)],
    *[f'Soil_Type{rep}' for rep in range(1, 40 + 1)],
    'Cover_Type',
]

license_info = 'Reuse of this database is unlimited with retention of copyright notice for ' \
               'Jock A. Blackard and Colorado State University.'


def convert_format(config):
    df = pd.read_csv(
        config.source / 'covtype.data.gz',
        header=None,
        index_col=None,
        names=column_names,
    )
    
    df['Cover_Type'] = df['Cover_Type'].astype('category')
    df, categories = convert_categorical(df)

    train_df = df[:11_340]
    valid_df = df[11_340:11_340 + 3_780]
    test_df = df[-565_892:]

    train_data_df, train_labels_df = split_by_label(train_df, col_name='Cover_Type')
    valid_data_df, valid_labels_df = split_by_label(valid_df, col_name='Cover_Type')
    test_data_df, test_labels_df = split_by_label(test_df, col_name='Cover_Type')
    
    if config.dataset_file:
        save_npz(
            config,
            {
                'train-data': train_data_df,
                'train-labels': train_labels_df,
                'valid-data': valid_data_df,
                'valid-labels': valid_labels_df,
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
                'train-profile': generate_profile(hvcat([[train_df], [valid_df]]), config.no_profile),
                'profile': generate_profile(df, config.no_profile),
                'categories': categories,
                'bibtex': uci_bibtex,
                'license': license_info,
                'column-names-attributes': list(train_data_df.columns),
                'column-names-target': list(train_labels_df.columns),
            }
        )


if __name__ == '__main__':
    args = default_config(
        'covertype',
        download_root='https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/',
    )
    
    convert_format(args)
