import openml
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from torch.utils.data import Dataset
from tabben.datasets import OpenTabularDataset, TabularCIFAR10Dataset
from torch.utils.data import DataLoader



def simple_lapsed_time(text, lapsed):
    hours, rem = divmod(lapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print(text+": {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))


def task_dset_ids(task):
    dataset_ids = {
        'binary': [1487,44,1590,42178,1111,31,42733,1494,1017,4134],
        'multiclass': [188, 1596, 4541, 40664, 40685, 40687, 40975, 41166, 41169, 42734],
        'regression':[541, 42726, 42727, 422, 42571, 42705, 42728, 42563, 42724, 42729]
    }

    return dataset_ids[task]

def concat_data(X,y):
    # import ipdb; ipdb.set_trace()
    return pd.concat([pd.DataFrame(X['data']), pd.DataFrame(y['data'][:,0].tolist(),columns=['target'])], axis=1)


def data_split(X,y,nan_mask,indices):
    x_d = {
        'data': X.values[indices],
        'mask': nan_mask.values[indices]
    }
    
    if x_d['data'].shape != x_d['mask'].shape:
        raise'Shape of data not same as that of nan mask!'
        
    y_d = {
        'data': y[indices].reshape(-1, 1)
    } 
    return x_d, y_d


def data_prep_openml(dataset_name, task, datasplit, target_task = 1):
    if dataset_name == "cifar10":
        ds = TabularCIFAR10Dataset('./')
        ds_test = TabularCIFAR10Dataset('./', split='test')
        for inputs, labels in DataLoader(ds, batch_size=50000):
            a = inputs.numpy()
            b = labels.numpy()
        for inputs, labels in DataLoader(ds_test, batch_size=50000):
            a_test = inputs.numpy()
            b_test = labels.numpy()
        X = pd.DataFrame(a)
        y = pd.DataFrame(b)
        X_test = pd.DataFrame(a_test)
        y_test = pd.DataFrame(b_test)
    else:
        ds = OpenTabularDataset('./', dataset_name)
        df = ds.dataframe()
        X = df[ds.input_attributes]
        y = df[ds.output_attributes]
        ds_test = OpenTabularDataset('./', dataset_name, split='test')
        df_test = ds_test.dataframe()
        X_test = df_test[ds_test.input_attributes]
        y_test = df_test[ds_test.output_attributes]
  

    
    if dataset_name ==  "covertype":
        categorical_indicator = ([False] * 10) + ([True]*44)
    elif dataset_name == "arcene":
        categorical_indicator = ([False] * 10000)
    elif dataset_name == "higgs":
        categorical_indicator = ([False] * 28)
    elif dataset_name == "poker":
        categorical_indicator = ([True] * 10)
    elif dataset_name == "sarcos":
        categorical_indicator = ([False] * 21)
        y = y.iloc[:,target_task - 1]
        y_test = y_test.iloc[:,target_task-1]
    elif dataset_name == "cifar10":
        categorical_indicator = ([False] * 3072)
    elif dataset_name == "adult":
        categorical_indicator = [False, True, False, True , False, True, True, True, True, True, False, False, False, True]
    elif dataset_name == "parkinsons":
        print(X.columns)
        X = X.rename(columns={a:str(i) for i,a in enumerate(X.columns)})
        X_test = X_test.rename(columns={a:str(i) for i,a in enumerate(X.columns)})
        print(X)
        categorical_indicator = [False] * 16
        y = y.iloc[:, target_task - 1]
        y_test = y_test.iloc[:,target_task-1]
    elif dataset_name == 'amazon':
        categorical_indicator = [True] * 9

        
    X["Set"] = np.random.choice(["train", "valid"], p=datasplit, size=(X.shape[0],))
    X_test["Set"] = "test"


        
    X_frames = [X, X_test]
    y_frames = [y, y_test]
    
    
    X = pd.concat(X_frames)
    y = pd.concat(y_frames)
    
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    
  
    train_indices = X[X.Set=="train"].index
    valid_indices = X[X.Set=="valid"].index
    test_indices = X[X.Set=="test"].index
   
    X = X.drop(columns=['Set'])

    categorical_columns = X.columns[list(np.where(np.array(categorical_indicator)==True)[0])].tolist()
    cont_columns = list(set(X.columns.tolist()) - set(categorical_columns))

    cat_idxs = list(np.where(np.array(categorical_indicator)==True)[0])
    con_idxs = list(set(range(len(X.columns))) - set(cat_idxs))

    for col in categorical_columns:
        X[col] = X[col].astype("object")     
     

    
    
    temp = X.fillna("MissingValue")
    nan_mask = temp.ne("MissingValue").astype(int)
    cat_dims = []
    for col in categorical_columns:
    #     X[col] = X[col].cat.add_categories("MissingValue")
        X[col] = X[col].fillna("MissingValue")
        l_enc = LabelEncoder() 
        X[col] = l_enc.fit_transform(X[col].values)
        cat_dims.append(len(l_enc.classes_))
    for col in cont_columns:
    #     X[col].fillna("MissingValue",inplace=True)
        X.fillna(X.loc[train_indices, col].mean(), inplace=True)
    y = y.values
    if task != 'regression':
        l_enc = LabelEncoder() 
        y = l_enc.fit_transform(y)
    X_train, y_train = data_split(X,y,nan_mask,train_indices)
    X_valid, y_valid = data_split(X,y,nan_mask,valid_indices)
    X_test, y_test = data_split(X,y,nan_mask,test_indices)

    train_mean, train_std = np.array(X_train['data'][:,con_idxs],dtype=np.float32).mean(0), np.array(X_train['data'][:,con_idxs],dtype=np.float32).std(0)
    train_std = np.where(train_std < 1e-6, 1e-6, train_std)
    # import ipdb; ipdb.set_trace()
    return cat_dims, cat_idxs, con_idxs, X_train, y_train, X_valid, y_valid, X_test, y_test, train_mean, train_std




class DataSetCatCon(Dataset):
    def __init__(self, X, Y, cat_cols,task='clf',continuous_mean_std=None):
        
        cat_cols = list(cat_cols)
        X_mask =  X['mask'].copy()
        X = X['data'].copy()
        con_cols = list(set(np.arange(X.shape[1])) - set(cat_cols))
        self.X1 = X[:,cat_cols].copy().astype(np.int64) #categorical columns
        self.X2 = X[:,con_cols].copy().astype(np.float32) #numerical columns
        self.X1_mask = X_mask[:,cat_cols].copy().astype(np.int64) #categorical columns
        self.X2_mask = X_mask[:,con_cols].copy().astype(np.int64) #numerical columns
        if task == 'clf':
            self.y = Y['data']#.astype(np.float32)
        else:
            self.y = Y['data'].astype(np.float32)
        self.cls = np.zeros_like(self.y,dtype=int)
        self.cls_mask = np.ones_like(self.y,dtype=int)
        if continuous_mean_std is not None:
            mean, std = continuous_mean_std
            self.X2 = (self.X2 - mean) / std

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        # X1 has categorical data, X2 has continuous
        return np.concatenate((self.cls[idx], self.X1[idx])), self.X2[idx],self.y[idx], np.concatenate((self.cls_mask[idx], self.X1_mask[idx])), self.X2_mask[idx]

