import os
import glob
import pandas as pd
from functools import partial
import torch
import csv
from torchnet.dataset import ListDataset, TransformDataset
from torchnet.transform import compose
from torch.utils.data import Dataset

from MiDiFewNets.dataLoader.base import convert_dict, CudaTransform, EpisodicBatchSampler, MyBatchSampler
import numpy as np
class myDataset(Dataset):
    def __init__(self, data_dir):
        data = np.array(pd.read_csv(data_dir).values[:, :324])
        self.len = data.shape[0]
        self.x = torch.from_numpy(data[:,:])

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.x[index]




Pipeline_CACHE = {}
DATA_DIR = os.path.join(os.path.dirname(__file__), '../../Data/pipeline')


def extract_episode(n_support, n_query, d):
    n_examples = d['data'].size(0)

    if n_query == -1:
        n_query = n_examples - n_support

    example_inds = torch.randperm(n_examples)[:n_support+n_query]
    support_inds = example_inds[:n_support]
    query_inds = example_inds[n_support:]

    xs = d['data'][support_inds]
    xq = d['data'][query_inds]

    return {
        'class': d['class'],
        'xs': xs,
        'xq': xq
    }

def convert_tensor(key, d):
    d[key] = 1.0 - torch.from_numpy(np.array(d[key], np.float32, copy=False)).transpose(0, 1).contiguous().view(d[key].shape[0], d[key].shape[1])
    return d

def scale_data(key, d):
    d[key] = d[key].reshape((1, d[key].shape[0]))
    return d

def load_class_data(d):
    alphabet = d['class']
    data_dir = os.path.join(DATA_DIR, alphabet, '0.csv')

    #读入数据
    class_data = myDataset(data_dir)

    if len(class_data) == 0:
        raise Exception("No data found for omniglot class {} at {}.".format(d['class'], data_dir))


    data_ds = TransformDataset(class_data,
                               compose([partial(convert_dict, 'data'),
                                        partial(scale_data, 'data'),
                                        partial(convert_tensor, 'data')]))

    loader = torch.utils.data.DataLoader(data_ds, batch_size=10, shuffle=False)
    for sample in loader:
        Pipeline_CACHE[d['class']] = sample['data']
        break  # only need one sample because batch size equal to dataset length

    return {'class': d['class'], 'data': Pipeline_CACHE[d['class']]}


def load(opt, splits):
    split_dir = os.path.join(DATA_DIR, 'splits', opt['data.split'])

    ret = {}

    for split in  splits:
        if split in ['val', 'test'] and opt['data.test_way'] != 0:
            n_way = opt['data.test_way']
        else:
            n_way = opt['data.way']

        if split in ['val', 'test'] and opt['data.test_shot'] != 0:
            n_support = opt['data.test_shot']
        else:
            n_support = opt['data.shot']

        if split in ['val', 'test'] and opt['data.test_query'] != 0:
            n_query = opt['data.test_query']
        else:
            n_query = opt['data.query']

        if split in ['val', 'test']:
            n_episodes = opt['data.test_episodes']
        else:
            n_episodes = opt['data.train_episodes']


        transforms = [partial(convert_dict, 'class'),
                      load_class_data,
                      partial(extract_episode, n_support, n_query)]

        if opt['data.cuda']:
            transforms.append(CudaTransform())

        transforms = compose(transforms)

        class_names = []

        with open(os.path.join(split_dir, "{:s}.txt".format(split)), 'r') as f:
            for class_name in f.readlines():
                class_names.append(class_name.rstrip('\n'))
        ds = TransformDataset(ListDataset(class_names), transforms)

        sampler = MyBatchSampler(len(ds), n_way, n_episodes)

        # use num_workers=0, otherwise may receive duplicate episodes
        ret[split] = torch.utils.data.DataLoader(ds, batch_sampler=sampler, num_workers=0)

    return ret