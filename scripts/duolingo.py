"""
Convert the duolingo spaced learning dataset to the standard format, including some
preprocessing of the data.

Because of the way that the source data is distributed, the data file needs be
downloaded first into some local directory (cannot specify a URL source).

Additional note if you actually run this script: as written, this script requires
a substantial amount of memory available.
"""
import contextlib
import os
import re
from copy import deepcopy
from itertools import chain

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import column_name_array, convert_categorical, default_config, generate_profile, \
    save_json, save_npz, split_by_label, stage
from tqdm import tqdm

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
    with stage('reading CSV file'):
        df = pd.read_csv(
            config.source / 'settles.acl16.learning_traces.13m.csv.gz',
            header=0,
            index_col=None,
            #nrows=10_000,
        )
    
    with stage('converting types'):
        df['lexeme_string'] = df['lexeme_string'].astype('string')
        df[categorical_columns] = df[categorical_columns].astype('category')
    
    # prepare the dataframe for "original"
    with stage('creating original dataset'):
        orig_df = df.drop(['lexeme_string'], axis=1)
        orig_df, orig_categories = convert_categorical(orig_df)
    
    # prepare the dataframe for "categorical"
    with stage('extracting surface forms, lemmas, parts of speech, and other tags'):
        cat_df = df.drop(['lexeme_id'], axis=1)
        cat_df[['surface_form', 'lemma', 'part_of_speech', 'modifiers']] = \
            df['lexeme_string'].str.extract(r'(\w+)/(\w+)<([^>]+)>(.*)').fillna('')
    
    with stage('finding the set of all modifiers'):
        all_tags = set(
            chain.from_iterable(
                re.split('>[^<>]*<', tag_str.partition('<')[-1].rpartition('>')[0])
                for tag_str in cat_df['modifiers'].to_list()
                if isinstance(tag_str, str) and len(tag_str.strip()) > 0
            )
        )
    
    with stage('adding new columns for each modifier/tag'):
        for idx, tag in enumerate(tqdm(all_tags, leave=False)):
            cat_df[tag] = cat_df['modifiers'].str.contains(tag, regex=False).astype(np.bool_)
            
            if (idx + 1) % 10 == 0:
                # defragment dataframe
                cat_df = cat_df.copy()
    
    with stage('converting types for categorical dataset'):
        cat_df[['surface_form', 'lemma', 'part_of_speech']] = \
            cat_df[['surface_form', 'lemma', 'part_of_speech']].astype('category')
        cat_df = cat_df.drop(['lexeme_string', 'modifiers'], axis=1)
        cat_df, cat_categories = convert_categorical(cat_df)
    
    # split original dataset
    train_users, test_users = train_test_split(
        orig_df['user_id'].unique(),
        train_size=0.8,
        random_state=17_123,
    )
    
    orig_train_df = orig_df[orig_df['user_id'].isin(train_users)]
    orig_test_df = orig_df[orig_df['user_id'].isin(test_users)]
    
    train_data_df, train_labels_df = split_by_label(orig_train_df, col_name='p_recall')
    test_data_df, test_labels_df = split_by_label(orig_test_df, col_name='p_recall')
    
    # "categorical" version
    cat_train_df = cat_df[cat_df['user_id'].isin(train_users)]
    cat_test_df = cat_df[cat_df['user_id'].isin(test_users)]
    
    cat_train_data_df, cat_train_labels_df = split_by_label(cat_train_df, col_name='p_recall')
    cat_test_data_df, cat_test_labels_df = split_by_label(cat_test_df, col_name='p_recall')
    
    cat_config = deepcopy(config)
    cat_config.name = 'duolingo-categorical'
    
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
        
        save_npz(
            cat_config,
            {
                'train-data': cat_train_data_df,
                'train-labels': cat_train_labels_df,
                'test-data': cat_test_data_df,
                'test-labels': cat_test_labels_df,
                '_columns-data': column_name_array(cat_train_data_df),
                '_columns-labels': column_name_array(cat_train_labels_df),
            }
        )
    
    if config.extras_file:
        common_extras = {
            'bibtex': bibtex,
            'license': 'CC BY-NC 4.0',
        }
        
        save_json(
            config,
            {
                'train-profile': generate_profile(orig_train_df, config.no_profile),
                'profile': generate_profile(orig_df, config.no_profile),
                'categories': orig_categories,
                'column-names-attributes': list(train_data_df.columns),
                'column-names-target': list(train_labels_df.columns),
                **common_extras,
            }
        )
        
        save_json(
            cat_config,
            {
                'train-profile': generate_profile(cat_train_df, config.no_profile),
                'profile': generate_profile(cat_df, config.no_profile),
                'categories': cat_categories,
                'column-names-attributes': list(cat_train_data_df.columns),
                'column-names-target': list(cat_train_labels_df.columns),
                **common_extras,
            }
        )


if __name__ == '__main__':
    args = default_config(
        'duolingo-original',
        download_root='https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/',
    )
    
    convert_format(args)
