"""
Convert the arcene dataset to the standardized format.
"""

import os

from utils import column_name_array, create_csv_reader, default_config, generate_profile, hvcat, save_json, \
    save_npz, \
    uci_license

bibtex = """\
@inproceedings{guyonResultAnalysisNIPS2005,
    title = {Result {{Analysis}} of the {{NIPS}} 2003 {{Feature Selection Challenge}}},
    booktitle = {Advances in {{Neural Information Processing Systems}}},
    author = {Guyon, Isabelle and Gunn, Steve and Ben-Hur, Asa and Dror, Gideon},
    date = {2005},
    volume = {17},
    publisher = {{MIT Press}},
    url = {https://papers.nips.cc/paper/2004/hash/5e751896e527c862bf67251a474b3819-Abstract.html},
}"""


def convert_format(config):
    read_csv = create_csv_reader(
        config.source,
        header=None,
        index_col=None,
        sep='\\s+',
    )
    
    if config.download_sources:
        pass
    
    # labels for one of the splits is stored in the parent directory of where everything else is stored,
    # but this is just the organization on
    if config.source == 'https://archive.ics.uci.edu/ml/machine-learning-databases/arcene/ARCENE/':
        valid_labels_source = 'https://archive.ics.uci.edu/ml/machine-learning-databases/arcene' \
                              '/arcene_valid.labels'
    else:
        valid_labels_source = os.path.join(config.source, 'arcene_valid.labels')
    
    train_data_df = read_csv('arcene_train.data')
    train_labels_df = read_csv('arcene_train.labels', names=['label'])
    valid_data_df = read_csv('arcene_valid.data')
    valid_labels_df = read_csv(valid_labels_source, use_root=False, names=['label'])
    
    train_labels_df['label'] = train_labels_df['label'].map(lambda x: (x + 1) // 2)
    valid_labels_df['label'] = valid_labels_df['label'].map(lambda x: (x + 1) // 2)
    
    if config.dataset_file:
        save_npz(
            config,
            {
                'train-data': train_data_df,
                'train-labels': train_labels_df['label'],
                'test-data': valid_data_df,
                'test-labels': valid_labels_df['label'],
                '_columns-data': column_name_array(train_data_df),
                '_columns-labels': column_name_array(train_labels_df),
            }
        )
    
    if config.extras_file:
        train_df = hvcat([[train_data_df, train_labels_df]])
        full_df = hvcat([[train_df],
                         [valid_data_df, valid_labels_df]])
        
        save_json(
            config,
            {
                'train-profile': generate_profile(train_df, config.no_profile),
                'profile': generate_profile(full_df, config.no_profile),
                'bibtex': bibtex,
                'license': uci_license,
                'column-names-attributes': list(train_data_df.columns),
                'column-names-target': list(train_labels_df.columns),
            }
        )


if __name__ == '__main__':
    args = default_config(
        'arcene',
        download_root='https://archive.ics.uci.edu/ml/machine-learning-databases/arcene/ARCENE/',
    )
    
    convert_format(args)
