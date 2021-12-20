"""
Convert the duolingo spaced learning dataset to the standard format, including some
preprocessing of the data.

Because of the way that the source data is distributed, the data file needs be
downloaded first into some local directory (cannot specify a URL source).
"""

import os
import re
from itertools import chain

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import column_name_array, convert_categorical, default_config, generate_profile, \
    save_json, save_to_numpy_array, split_by_label

bibtex = """\
@inproceedings{settles.acl16,
    Author = {B. Settles and B. Meeder},
    Booktitle = {Proceedings of the Association for Computational Linguistics (ACL)},
    Pages = {1848--1858},
    Publisher = {ACL},
    Title = {A Trainable Spaced Repetition Model for Language Learning},
    Year = {2016}
    DOI = {10.18653/v1/P16-1174},
    URL = {http://www.aclweb.org/anthology/P16-1174}
}"""

categorical_columns = ['user_id', 'learning_language', 'ui_language', 'lexeme_id']


def convert_format(config):
    df = pd.read_csv(
        os.path.join(config.source, 'settles.acl16.learning_traces.13m.csv'),
        header=0,
        index_col=None,
        #nrows=10000
    )
    df['lexeme_string'] = df['lexeme_string'].astype('string')
    df[categorical_columns] = df[categorical_columns].astype('category')
    
    # prepare the dataframe for "original"
    orig_df = df.drop(['lexeme_string'], axis=1)
    orig_categories = convert_categorical(orig_df)
    
    # prepare the dataframe for "categorical"
    cat_df = df.drop(['lexeme_id'], axis=1)
    cat_df[['surface_form', 'lemma', 'part_of_speech', 'modifiers']] = \
        df['lexeme_string'].str.extract(r'(\w+)/(\w+)<([^>]+)>(.*)').fillna('')
    
    all_tags = set(
        chain.from_iterable(
            re.split('>[^<>]*<', tag_str.partition('<')[-1].rpartition('>')[0])
            for tag_str in cat_df['modifiers'].to_list()
            if isinstance(tag_str, str) and len(tag_str.strip()) > 0
        )
    )
    for tag in all_tags:
        cat_df[tag] = cat_df['modifiers'].str.contains(tag, regex=False).astype(int)
    
    cat_df[['surface_form', 'lemma', 'part_of_speech']] = \
        cat_df[['surface_form', 'lemma', 'part_of_speech']].astype('category')
    cat_df = cat_df.drop(['lexeme_string', 'modifiers'], axis=1)
    cat_categories = convert_categorical(cat_df)
    
    if config.dataset_file:
        train_users, test_users = train_test_split(
            orig_df['user_id'].unique(),
            train_size=0.8,
            random_state=17_123,
        )
        
        orig_train_df = orig_df[orig_df['user_id'].isin(train_users)]
        orig_test_df = orig_df[orig_df['user_id'].isin(test_users)]
        
        train_data_df, train_labels_df = split_by_label(orig_train_df, col_name='p_recall')
        test_data_df, test_labels_df = split_by_label(orig_test_df, col_name='p_recall')
        
        save_to_numpy_array(
            os.path.join(config.outputdirectory, 'duolingo-original'), {
                'train-data': train_data_df,
                'train-labels': train_labels_df,
                'test-data': test_data_df,
                'test-labels': test_labels_df,
                '_columns-data': column_name_array(train_data_df),
                '_columns-labels': column_name_array(train_labels_df),
            }
        )
        
        # "categorical" version
        cat_train_df = cat_df[cat_df['user_id'].isin(train_users)]
        cat_test_df = cat_df[cat_df['user_id'].isin(test_users)]
        
        train_data_df, train_labels_df = split_by_label(cat_train_df, col_name='p_recall')
        test_data_df, test_labels_df = split_by_label(cat_test_df, col_name='p_recall')
        
        save_to_numpy_array(
            os.path.join(config.outputdirectory, 'duolingo-categorical'), {
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
            {
                'profile': generate_profile(orig_df),
                'categories': orig_categories,
                'bibtex': bibtex,
                'license': 'CC BY-NC 4.0',
            }, os.path.join(config.outputdirectory, 'duolingo-original.json')
        )
        
        save_json(
            {
                'profile': generate_profile(cat_df),
                'categories': cat_categories,
                'bibtex': bibtex,
                'license': 'CC BY-NC 4.0',
            }, os.path.join(config.outputdirectory, 'duolingo-categorical.json')
        )


if __name__ == '__main__':
    args = default_config(
        source_default='https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/',
    )
    
    convert_format(args)
