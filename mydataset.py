import numpy as np
import os
import scipy.io as scio
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose


def normalize_to_255(data):  # 归一化到0-255
    rawdata_max = max(map(max, data))
    rawdata_min = min(map(min, data))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i][j] = round(((255 - 0) * (data[i][j] - rawdata_min) / (rawdata_max - rawdata_min)) + 0)
    return data


# def normalize_to_255(data):
#     # 将数据归一化到 [0, 255] 范围
#     rawdata_max = data.max()
#     rawdata_min = data.min()
#     data = ((255 - 0) * (data - rawdata_min) / (rawdata_max - rawdata_min)) + 0
#     return data

def normalize_to_1(data):
    # 将数据归一化到 [0, 1] 范围
    data = data / 255.0
    return data

def standardize(data, mean=0.5, std=0.5):
    # 将数据标准化到 [-1, 1] 范围
    data = (data - mean) / std
    return data

class MyDataset(Dataset):
    def __init__(self, root_dir, names_file, transform=None):
        self.root_dir = root_dir
        self.names_file = names_file
        self.transform = transform
        self.size = 0
        self.names_list = []
        if not os.path.isfile(self.names_file):
            print(self.names_file + ' does not exist!')
        file = open(self.names_file)
        for f in file:
            self.names_list.append(f.strip())
            self.size += 1

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        data_path = self.root_dir + self.names_list[idx].split(' ')[0]
        if not os.path.isfile(data_path):
            print(data_path + ' does not exist!')
            return None
        rawdata = scio.loadmat(data_path)['data']
        rawdata = rawdata.astype(np.float32)  # 确保数据类型为 float32
        # 先归一化到 [0, 255] 范围
        data = normalize_to_255(rawdata)
        # 再归一化到 [0, 1] 范围
        data = normalize_to_1(data)
        # 再标准化到 [-1, 1] 范围
        data = standardize(data)
        label = int(self.names_list[idx].split(' ')[1])
        sample = {'data': data, 'label': label}
        if self.transform:
            sample = self.transform(sample)

        return sample
        # rawdata = scio.loadmat(data_path)['data']  # 10000,12 uint16
        # rawdata = rawdata.astype(int)  # int32
        # data = normalize_to_255(rawdata)
        # label = int(self.names_list[idx].split(' ')[1])
        # sample = {'data': data, 'label': label}
        # if self.transform:
        #     sample = self.transform(sample)
        # return sample