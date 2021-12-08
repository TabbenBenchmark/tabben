"""
Convert the forest cover type classification dataset to the standardized format.
"""

import os

import pandas as pd
import simplejson as json

from utils import column_name_array, convert_categorical, default_config, save_to_numpy_array, split_by_label

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


def convert_format(config):
    df = pd.read_csv(
        os.path.join(config.source, 'covtype.data.gz'),
        header=None,
        index_col=None,
        names=column_names,
    )
    
    df['Cover_Type'] = df['Cover_Type'].astype('category')
    categories = convert_categorical(df)
    
    train_data_df, train_labels_df = split_by_label(df[:11_340], col_name='Cover_Type')
    valid_data_df, valid_labels_df = split_by_label(df[11_340:11_340 + 3_780], col_name='Cover_Type')
    test_data_df, test_labels_df = split_by_label(df[-565_892:], col_name='Cover_Type')
    
    save_to_numpy_array(
        os.path.join(config.outputdirectory, 'covertype'), {
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
    
    with open(os.path.join(config.outputdirectory, 'covertype.json'), 'w') as f:
        json.dump(
            {
                'categories': categories,
            }, f
        )


if __name__ == '__main__':
    args = default_config(
        source_default='https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/',
    )
    
    convert_format(args)
