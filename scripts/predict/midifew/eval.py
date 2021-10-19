import os
import json
import math
import numpy as np
from tqdm import tqdm
import pandas as pd
import torch
import torchnet as tnt
import csv
from MiDiFewNets.utils import filter_opt, merge_dict
import MiDiFewNets.utils.data as data_utils
import MiDiFewNets.utils.model as model_utils

def frames_into_csv(frames):
    name = '../../../data/RawData/output/proto_TestResults_temp.csv'
    frames.to_csv(name, index=None)
    file_name_raw = '../../../data/RawData/output/proto_TestResults.csv'
    reader = csv.DictReader(open(name))
    header = reader.fieldnames
    with open(file_name_raw, 'a', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames = header)
        writer.writerows(reader)

def main(opt):
    # load model
    model = torch.load(opt['model.model_path'])
    model.eval()

    # load opts
    model_opt_file = os.path.join(os.path.dirname(opt['model.model_path']), 'opt.json')
    with open(model_opt_file, 'r') as f:
        model_opt = json.load(f)

    # Postprocess arguments
    model_opt['model.x_dim'] = map(int, model_opt['model.x_dim'].split(','))
    model_opt['log.fields'] = model_opt['log.fields'].split(',')

    # construct data
    data_opt = { 'data.' + k: v for k,v in filter_opt(model_opt, 'data').items() }

    episode_fields = {
        'data.test_way': 'data.way',
        'data.test_shot': 'data.shot',
        'data.test_query': 'data.query',
        'data.test_episodes': 'data.train_episodes'
    }

    for k,v in episode_fields.items():
        if opt[k] != 0:
            data_opt[k] = opt[k]
        elif model_opt[k] != 0:
            data_opt[k] = model_opt[k]
        else:
            data_opt[k] = model_opt[v]

    print("Evaluating {:d}-way, {:d}-shot with {:d} query examples/class over {:d} episodes".format(
        data_opt['data.test_way'], data_opt['data.test_shot'],
        data_opt['data.test_query'], data_opt['data.test_episodes']))

    torch.manual_seed(1234)
    if data_opt['data.cuda']:
        torch.cuda.manual_seed(1234)

    data = data_utils.load(data_opt, ['test'])

    if data_opt['data.cuda']:
        model.cuda()

    meters = { field: tnt.meter.AverageValueMeter() for field in model_opt['log.fields'] }

    model_utils.evaluate(model, data['test'], meters, desc="test")
    # df = pd.DataFrame(data=None, columns=['loss','accuracy','precision','recall','f1-score'])
    # temp = []

    for field,meter in meters.items():
        mean, std = meter.value()
        print("test {:s}: {:0.6f} +/- {:0.6f}".format(field, mean, 1.96 * std / math.sqrt(data_opt['data.test_episodes'])))
    #     temp.append(mean)
    # temp = np.array(temp).reshape(1,-1)
    # temp = pd.DataFrame(temp, columns=df.columns)
    # frames_into_csv(temp)


