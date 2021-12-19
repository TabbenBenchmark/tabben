from pytorch_tabnet.tab_model import TabNetClassifier
from tabben.datasets import OpenTabularDataset
from itertools import compress
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import os
import wget
from pathlib import Path
from matplotlib import pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dset_id', required=True, type=str, choices=['arcene', 'poker', 'higgs', 'sarcos' ,'covertype', 'cifar10', 'adult', 'parkinsons', 'adult', 'amazon'])
parser.add_argument('--target_task', default=1, type=int)
parser.add_argument('--valid_split', default=.2, type=float)
parser.add_argument('--task', required=True, type=str,choices = ['binary','multiclass','regression'])
parser.add_argument('--embedding_size', default=32, type=int)
parser.add_argument('--transformer_depth', default=6, type=int)
parser.add_argument('--n_d', default=64, type=int)
parser.add_argument('--n_a', default=64, type=int)
parser.add_argument('--n_steps', default=5, type=int)
parser.add_argument('--gamma', default=1.5, type=float)
parser.add_argument('--n_independent', default=2, type=int)
parser.add_argument('--n_shared', default=2, type=int)
parser.add_argument('--cat_emb_dim', default=1, type=int)
parser.add_argument('--lambda_sparse', default=1e-4, type=float)
parser.add_argument('--momentum', default=.7, type=float)
parser.add_argument('--lr', default=2e-2, type=float)
parser.add_argument('--gamma_step', default=.95, type=float)
parser.add_argument('--step_size', default=20, type=int)
parser.add_argument('--clip_value', default=2., type=float)
parser.add_argument('--max_epochs', default=1000, type=int)
parser.add_argument('--patience', default=100, type=int)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--virtual_batch_size', default=128, type=int)
opt = parser.parse_args()
ds = OpenTabularDataset('./', opt.dset_id)
test_ds = OpenTabularDataset('./', opt.dset_id, split='test')
train = ds.dataframe()
test = test_ds.dataframe()

train["Set"] = np.random.choice(["train", "valid"], p =[1-opt.valid_split, opt.valid_split], size=(train.shape[0],))
test['Set'] = 'test'




n_total = len(train)

train = pd.concat([train, test])
train = train.reset_index(drop=True)

columns = list(train.columns)
unused_features = []

train_indices = train[train.Set=="train"].index
valid_indices = train[train.Set=="valid"].index
test_indices = train[train.Set=="test"].index

dataset_name = opt.dset_id
if dataset_name ==  "covertype":
    categorical_indicator = ([False] * 10) + ([True]*44)
    target = 'Cover_Type'
elif dataset_name == "arcene":
    target = "label"
    categorical_indicator = ([False] * 10000)
elif dataset_name == "higgs":
    target = "label"
    categorical_indicator = ([False] * 28)
elif dataset_name == "poker":
    target = 'label'
    categorical_indicator = ([True] * 10)
elif dataset_name == "sarcos":
    target = "T" + str(opt.target_task)
    unused_features = ["T" + str(i) for i in range(1,8)]
    categorical_indicator = ([False] * 21)
    test = test.iloc[:,target_task - 1]
    y_test = y_test.iloc[:,target_task-1]
elif dataset_name == "cifar10":
    categorical_indicator = ([False] * 3072)
elif dataset_name == "adult":
    target = 'income'
    categorical_indicator = [False, True, False, True , False, True, True, True, True, True, False, False, False, True]
elif dataset_name == "parkinsons":
    target = "motor_UPDRS" if opt.target_task == 1 else "total_UPDRS"
    unused_features = ['motor_UPDRS', 'total_UPDRS']
    categorical_indicator = [False] * 16
elif dataset_name == 'amazon':
    target = "ACTION"
    categorical_indicator = [True] * 9

cat_columns = list(compress(train.columns, categorical_indicator))


nunique = train.nunique()
types = train.dtypes

categorical_columns = []
categorical_dims =  {}
for col in train.columns:
    if types[col] == 'object' or nunique[col] < 200:
        print(col, train[col].nunique())
        l_enc = LabelEncoder()
        train[col] = train[col].fillna("VV_likely")
        train[col] = l_enc.fit_transform(train[col].values)
        categorical_columns.append(col)
        categorical_dims[col] = len(l_enc.classes_)
    else:
        train.fillna(train.loc[train_indices, col].mean(), inplace=True)
        
unused_feat = [target] + unused_features

features = [ col for col in train.columns if col not in unused_feat+[target]] 

cat_idxs = [ i for i, f in enumerate(features) if f in categorical_columns]

cat_dims = [ categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]


X_train = train[features].values[train_indices]
y_train = train[target].values[train_indices]

X_valid = train[features].values[valid_indices]
y_valid = train[target].values[valid_indices]

X_test = train[features].values[test_indices]
y_test = train[target].values[test_indices]

if opt.task == 'regression':
    eval_metric = ['rmse']
    clf = TabNetRegressor(
        n_d=opt.n_d, n_a=opt.n_a, n_steps=opt.n_steps,
        gamma=opt.gamma, n_independent=opt.n_independent, n_shared=opt.n_shared,
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        cat_emb_dim=opt.cat_emb_dim,
        lambda_sparse=opt.lambda_sparse, momentum=opt.momentum, clip_value=opt.clip_value,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=opt.lr),
        scheduler_params = {"gamma": opt.gamma,
                         "step_size": opt.step_size},
        scheduler_fn=torch.optim.lr_scheduler.StepLR, epsilon=1e-15
    )
else:
    clf = TabNetClassifier(
        n_d=opt.n_d, n_a=opt.n_a, n_steps=opt.n_steps,
        gamma=opt.gamma, n_independent=opt.n_independent, n_shared=opt.n_shared,
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        cat_emb_dim=opt.cat_emb_dim,
        lambda_sparse=opt.lambda_sparse, momentum=opt.momentum, clip_value=opt.clip_value,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=opt.lr),
        scheduler_params = {"gamma": opt.gamma_step,
                         "step_size": opt.step_size},
        scheduler_fn=torch.optim.lr_scheduler.StepLR, epsilon=1e-15
    )

    if opt.task == 'binary':
        eval_metric = ['accuracy', 'auc']
    else:
        eval_metric = ['accuracy']


clf.fit(
    X_train=X_train, y_train=y_train,
    eval_set=[(X_train, y_train),(X_test, y_test),(X_valid, y_valid)],
    eval_name=['train', 'test', 'valid'],
    eval_metric=eval_metric,
    max_epochs=opt.max_epochs , patience=opt.patience,
    batch_size=opt.batch_size, virtual_batch_size=opt.virtual_batch_size,
)

preds = clf.predict_proba(X_test)
if opt.task == 'binary':
    test_auc = roc_auc_score(y_score=preds[:,1], y_true=y_test)

    print(f"FINAL TEST SCORE FOR {opt.dset_id} : {test_auc}")
elif opt.task =='regression':
    y_true = y_test
    test_score = mean_squared_error(y_pred=preds, y_true=y_true)
    print(f"FINAL TEST SCORE FOR {opt.dset_id} : {test_score}")
else:
    preds = clf.predict_proba(X_train)
    y_pred = np.vectorize(preds_mapper.get)(np.argmax(preds, axis=1))


    test_acc = accuracy_score(y_pred=y_pred, y_true=y_train)

    print(f"FINAL TEST SCORE FOR {opt.dset_id} : {test_acc}")