import os

import simplejson as json
from sklearn.model_selection import train_test_split

from utils import save_to_numpy_array, create_csv_reader, default_arg_parser, column_name_array, \
    split_by_label


def convert_categorical(df):
    categorical_columns = df.select_dtypes(['category']).columns
    
    categories = {}
    
    for column in categorical_columns:
        categories[column] = list(df[column].cat.categories)
    
    df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
    
    return categories


state_holidays = {
    'a': 'public holiday',
    'b': 'Easter holiday',
    'c': 'Christmas',
    '0': 'None',
}

assortments = {
    'a': 'basic',
    'b': 'extra',
    'c': 'extended',
}


def convert_format(config):
    read_csv = create_csv_reader(
            config.source,
            header=0,
            index_col=None,
            sep=',',
            low_memory=False,
    )
    
    train_df = read_csv('train.csv')
    train_df['StateHoliday'] = train_df['StateHoliday'].map(lambda h: state_holidays[h]).astype('category')
    train_df[['Year', 'Month', 'Day']] = train_df['Date'].str.split('-', 3, expand=True).astype(int)
    
    store_df = read_csv('store.csv')
    store_df['StoreType'] = store_df['StoreType'].astype('category')
    store_df['Assortment'] = store_df['Assortment'].map(lambda h: assortments[h]).astype('category')
    store_df['PromoInterval'] = store_df['PromoInterval'].astype('category')
    
    df = train_df.join(store_df.set_index('Store'), 'Store')
    categories = convert_categorical(df)
    
    train_stores, test_stores = train_test_split(train_df['Store'].unique(), train_size=0.8,
                                                 random_state=171_234)
    
    train_df = df[df['Store'].isin(train_stores)].drop(['Store', 'Date'], axis=1)
    test_df = df[df['Store'].isin(test_stores)].drop(['Store', 'Date'], axis=1)
    
    train_data_df, train_labels_df = split_by_label(train_df, 'Sales')
    test_data_df, test_labels_df = split_by_label(test_df, 'Sales')
    
    save_to_numpy_array(os.path.join(config.outputdirectory, 'rossman'), {
        'train-data'     : train_data_df,
        'train-labels'   : train_labels_df,
        'test-data'      : test_data_df,
        'test-labels'    : test_labels_df,
        '_columns-data'  : column_name_array(train_data_df),
        '_columns-labels': column_name_array(train_labels_df),
    })
    
    with open(os.path.join(config.outputdirectory, 'rossman.json'), 'w') as f:
        json.dump({
            'categories': categories,
        }, f)


if __name__ == '__main__':
    args = default_arg_parser().parse_args()
    
    convert_format(args)

