import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random

#字典转换
def convert_dict(key, val):
    return { key: val }

class CudaTransform(object):
    def __init__(self):
        pass

    def __call__(self, data):
        for k,v in data.items():
            if hasattr(v, 'cuda'):
                data[k] = v.cuda()
        return data

class MyBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.cat((torch.zeros(1).long(),
                             # torch.ones(1).long(),
                             torch.from_numpy
                             (np.array
                              (random.sample(range(1, self.n_classes), self.n_way - 1))).long()))

class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            # yield torch.cat((torch.zeros(1).long(),
            #                  # torch.ones(1).long(),
            #                  torch.from_numpy
            #                  (np.array
            #                   (random.sample(range(1,self.n_classes),self.n_way-1))).long())) #replace by oyyk, solid first num n_classes是文件总数即上限 n_way是抽取多少个
            # #for binary class test
            # # yield torch.cat((torch.zeros(1).long(),
            # #                  torch.zeros(1).long()))

            yield torch.randperm(self.n_classes)[:self.n_way]