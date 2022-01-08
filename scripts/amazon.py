"""
Convert the Kaggle Amazon Employee Access Prediction dataset to the standardized
format.
"""

from sklearn.model_selection import train_test_split

from utils import column_name_array, create_csv_reader, default_config, generate_profile, save_json, \
    save_npz, split_by_label


def convert_format(config):
    read_csv = create_csv_reader(
        config.source,
        header=0,
        index_col=None,
        sep=',',
    )
    
    if config.download_sources:
        pass
    
    df = read_csv('train.csv')
    
    if config.dataset_file:
        train_df, test_df = train_test_split(df, train_size=0.8, random_state=171_234, stratify=df['ACTION'])
        
        train_data_df, train_labels_df = split_by_label(train_df, 'ACTION')
        test_data_df, test_labels_df = split_by_label(test_df, 'ACTION')
        
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
                'profile': generate_profile(df),
            }
        )


if __name__ == '__main__':
    args = default_config('amazon')
    
    convert_format(args)
