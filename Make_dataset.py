"""
Author: HaoHe
Email: hehao@stu.xmu.edu.cn
Date: 2020-09-13
"""

import numpy as np
import os
import torch
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


class Read_data:
    def __init__(self, path, valid_ratio=20):
        self.path = path
        self.valid_ratio = valid_ratio

    def read_file(self):
        """
        Args:
            valid_ratio: the ratio of the validation sets
            path: the absolute path of the source file
        """

        filenames = os.listdir(self.path)
        train_data, valid_data = [], []
        for filename in filenames:
            # 分别存储单个文件中的训练数据和验证数据及标签
            tmp_train_data, tmp_valid_data = [], []
            if os.path.splitext(filename)[1] == '.mat':
                # 绝对路径+文件名
                name = self.path + '/' + filename
                tmp = sio.loadmat(name)
                for key in tmp.keys():
                    if key == 'noise':
                        data = np.array(tmp[key])
                data = data.T
                num, spec = data.shape
                valid_num = int(np.ceil(self.valid_ratio / 100 * num))
                tmp_valid_data, tmp_train_data = data[0:valid_num], data[valid_num:]
                train_data.append(tmp_train_data)
                valid_data.append(tmp_valid_data)
        train_data_tmp = np.array(train_data[0])
        valid_data_tmp = np.array(valid_data[0])

        # list 拼接成numpy数组
        if len(train_data) > 1:
            for i in range(1, len(train_data)):
                train_data_tmp = np.concatenate((train_data_tmp, train_data[i]), axis=0)
                valid_data_tmp = np.concatenate((valid_data_tmp, valid_data[i]), axis=0)

        train_dataset = train_data_tmp.reshape((-1, 1, spec))
        valid_dataset = valid_data_tmp.reshape((-1, 1, spec))
        return train_dataset, valid_dataset


# 制作数据集
class Make_dataset(Dataset):
    def __init__(self, data):
        super(Make_dataset, self).__init__()
        self.data = data

    def __getitem__(self, index):
        feature = self.data[index]
        return feature

    def __len__(self):
        return np.size(self.data, 0)


if __name__ == '__main__':
    path = r'E:\PAPER\paper writing\Noise learning\Simulate datasets'
    reader = Read_data(path, 20)
    train_set, _ = reader.read_file()
    print(train_set.shape)
    x = np.mean(train_set[0], axis=0)
    print(x.shape)
    plt.plot(x.ravel())
    # test dataloader
    myset = Make_dataset(train_set)
    train_loader = DataLoader(dataset=myset, batch_size=100)
    print(len(train_loader))
    for batch_idx, raw in enumerate(train_loader):
        if batch_idx > 1:
            print(raw.shape)
            x = np.mean(raw.numpy(), axis=0)
            print(x.shape)
            plt.plot(x)
            plt.show()
            break
