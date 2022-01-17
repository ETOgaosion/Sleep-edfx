import os
from torch.utils.data import Dataset, DataLoader
from scipy import io
import numpy as np


class MyDataset(Dataset):
    def __init__(self, path_dir, transform=None, target_transform=None):

        self.path_dir = path_dir  # 文件路径,如'.\data\cat-dog'
        self.transform = transform  # 对图形进行处理，如标准化、截取、转换等
        self.target_transform = target_transform  # 对标签进行处理
        self.seq_list = os.listdir(self.path_dir)  # 把路径下的所有文件放在一个列表中

        bispectra = []
        labels = []
        for mat_idx in self.seq_list:
            mat_name = self.path_dir + mat_idx
            fea_loader = io.loadmat(mat_name)
            temp1 = np.array(fea_loader['bs_amp'])
            temp2 = np.array(fea_loader['bs_pha'])
            temp = np.append(temp1, temp2, axis=2)
            # bispectra = np.append(np.append(temp1, temp2, axis=2))

            bispectra = np.append(bispectra, temp, axis=0)
            labels = np.append(labels, np.array(fea_loader['labels']))
            labels.append()

        self.bispectra = bispectra
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        bispecturm = self.bispectra[index]
        label = self.labels[index]

        return bispecturm, label