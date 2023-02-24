import os
import numpy as np
import pandas as pd
import torch

class MCMLICDataset:
    def __init__(self, data_dir, label_dir,train_rate,device, split='train'):
        self.device = device
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.filenames = [f for f in os.listdir(data_dir) if f.endswith('.pt')]
        self.filenames.sort()
        self.labels = []
        for f in self.filenames:
            label_file = f.replace('.pt', '.csv')
            label = pd.read_csv(os.path.join(label_dir, label_file), header=None)
            self.labels.append(torch.tensor(label.values).flatten())

        # 全データをインデックスで指定して、無作為に8:2の割合で分割する
        indices = np.arange(len(self.filenames))
        np.random.shuffle(indices)
        split_num = int(len(indices) * train_rate)

        if split == 'train':
            self.filenames = [self.filenames[i] for i in indices[:split_num]]
            self.labels = [self.labels[i].float() for i in indices[:split_num]]
        elif split == 'validation':
            self.filenames = [self.filenames[i] for i in indices[split_num:]]
            self.labels = [self.labels[i].float() for i in indices[split_num:]]
        else:
            raise ValueError("Invalid value for split. Choose 'train' or 'validataion'.")

    def __getitem__(self, idx):
        data = torch.load(os.path.join(self.data_dir, self.filenames[idx]), map_location=self.device)
        return data, self.labels[idx]

    def __len__(self):
        return len(self.filenames)