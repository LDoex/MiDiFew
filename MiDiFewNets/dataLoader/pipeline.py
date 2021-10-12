import os
import glob
import pandas as pd
from functools import partial
import torch
import csv
from torchnet.dataset import ListDataset, TransformDataset
from torchnet.transform import compose

from MiDiFewNets.dataLoader.base import convert_dict, CudaTransform, EpisodicBatchSampler, MyBatchSampler
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), '../../Data/pipeline')

def data_prepro(key, out_field, d):
    pass

def extract_episode(n_support, n_query, d):
    pass

def convert_tensor(key, d):
    d[key] = 1.0 - torch.from_numpy(np.array(d[key], np.float32, copy=False)).transpose(0, 1).contiguous().view(1, d[key].shape[0], d[key].shape[1])
    return d

def scale_data(key, height, width, d):
    d[key] = d[key].reshape((height,width))
    return d

def load_class_data(d):
    alphabet = d['class']
    data_dir = os.path.join(DATA_DIR, alphabet, '*.csv')

    #读入数据
    class_data = sorted(glob.glob(os.path.join(data_dir, '*.csv')))

    if len(class_data) == 0:
        raise Exception("No data found for omniglot class {} at {}.".format(d['class'], data_dir))

    data_ds = TransformDataset(ListDataset(class_data),
                               compose([partial(convert_dict, 'file_name'),
                                        partial(data_prepro, 'file_name', 'data'),
                                        partial(scale_data, 'data', 18, 18),
                                        partial(convert_tensor, 'data')]))



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

    sampler = MyBatchSampler(len(ds))

    # use num_workers=0, otherwise may receive duplicate episodes
    ret[split] = torch.utils.data.DataLoader(ds, batch_sampler=sampler, num_workers=0)

    return ret