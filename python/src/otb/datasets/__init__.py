import os
import torch
from torch.utils.data import Dataset


# TODO: ogb uses a general dataset to load any dataset, should we try something similar?
class HIGGSDataset(Dataset):
    def __init__(self, data_dir, train=True, download=False):
        split = 'train' if train else 'test'

        # TODO download data files if not already present

        data_file = os.path.join(data_dir, f'{split}_data.pt')
        label_file = os.path.join(data_dir, f'{split}_labels.pt')

        self.X = torch.load(data_file)
        self.y = torch.load(label_file)

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return self.X[idx, :], self.y[idx]


