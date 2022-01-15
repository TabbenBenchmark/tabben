"""
Convert the KDD Cup data (all 3 tasks) into the standardized format.
"""

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import column_name_array, convert_categorical, create_csv_reader, default_config, \
    generate_profile, \
    hvcat, save_json, \
    save_npz, split_by_label, stage

bibtex = """\
@book{KDDCUP09Proceedings2009,
  title = {{{KDD-CUP}}'09: {{Proceedings}} of the 2009 {{International Conference}} on {{KDD-Cup}} 2009 - {
  {Volume}} 7},
  date = {2009},
  publisher = {{JMLR.org}},
  location = {{Paris, France}},
  keywords = {cmsc673}
}
"""


def convert_format(config):
    read_csv = create_csv_reader(
        config.source,
        header=0,
        index_col=None,
        sep='\t',
        low_memory=False,
        dtype={
            **{
                f'Var{index + 1}': np.float32
                for index in range(14_740)
            },
            **{
                f'Var{index + 1}': 'category'
                for index in range(14_740, 15_000)
            }
        }
    )
    
    with stage('reading first data subset'):
        first_data_df = read_csv('orange_large_train.data.chunk1.zip')
    
    with stage('reading and constructing full dataset'):
        df = hvcat(
            [
                [first_data_df],
                [read_csv('orange_large_train.data.chunk2.zip', header=None, names=first_data_df.columns)],
                [read_csv('orange_large_train.data.chunk3.zip', header=None, names=first_data_df.columns)],
                [read_csv('orange_large_train.data.chunk4.zip', header=None, names=first_data_df.columns)],
                [read_csv('orange_large_train.data.chunk5.zip', header=None, names=first_data_df.columns)],
            ]
        )
        
        def get_single_label(filename):
            return (read_csv(filename, header=None, squeeze=True, dtype=np.int8) + 1) // 2
        
        df['appetency'] = get_single_label('orange_large_train_appetency.labels')
        df['churn'] = get_single_label('orange_large_train_churn.labels')
        df['upselling'] = get_single_label('orange_large_train_upselling.labels')
    
    with stage('extracting and converting categorical attributes'):
        df, categories = convert_categorical(df)
        df = df.copy()
        
        df['labels'] = 100 * df['appetency'] + 10 * df['churn'] + df['upselling']
    
    with stage('begin splitting dataset'):
        train_df, test_df = train_test_split(
            df, train_size=0.8, random_state=171_234,
            stratify=df['labels']
        )
        train_df = train_df.drop(['labels'], axis=1).copy()
        test_df = test_df.drop(['labels'], axis=1).copy()
        
        train_data_df, train_labels_df = split_by_label(train_df, ['appetency', 'churn', 'upselling'])
        test_data_df, test_labels_df = split_by_label(test_df, ['appetency', 'churn', 'upselling'])
    
    if config.dataset_file:
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
                'train-profile': generate_profile(train_df, config.no_profile),
                'profile': generate_profile(df, config.no_profile),
                'categories': categories,
                'bibtex': bibtex,
                'column-names-attributes': list(train_data_df.columns),
                'column-names-target': list(train_labels_df.columns),
            }
        )


if __name__ == '__main__':
    args = default_config('kdd09')
    
    convert_format(args)
