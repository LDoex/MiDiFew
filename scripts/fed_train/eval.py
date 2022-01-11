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

def eval_main(classFile_name ,opt, net):
    # load model
    model = net

    model.eval()


    # load opts
    model_opt_file = os.path.join(opt['log.exp_dir'], 'opt.json')
    with open(model_opt_file, 'r') as f:
        model_opt = json.load(f)

    # # Postprocess arguments
    # model_opt['model.x_dim'] = map(int, model_opt['model.x_dim'].split(','))
    # model_opt['log.fields'] = model_opt['log.fields'].split(',')

    # construct data
    data_opt = { 'data.' + k: v for k,v in filter_opt(model_opt, 'data').items() }

    episode_fields = {
        'data.test_way': 'data.way',
        'data.test_shot': 'data.shot',
        'data.test_query': 'data.query',
        'data.test_episodes': 'fed.test_episodes'
    }

    for k,v in episode_fields.items():
        if k == 'data.test_episodes':
            data_opt[k] = model_opt[v]
        elif opt[k] != 0:
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

    data = data_utils.load(data_opt, [classFile_name])

    if data_opt['data.cuda']:
        model.cuda()

    # for plot
    file_path = os.path.join('../../plot/')
    y_cache = {'y_re': [], 'decision_val': [], 'predic_label': []}
    # df = pd.DataFrame(data=None, columns=['loss','accuracy','precision','recall','f1-score'])
    # temp = []
    re_file = os.path.join(file_path, 'y_real_' + opt['file.suffixName'] + '.csv')
    decision_val_file = os.path.join(file_path, 'decision_val_' + opt['file.suffixName'] + '.csv')
    preLabel_file = os.path.join(file_path, 'predLabel_' + opt['file.suffixName'] + '.csv')

    meters = {'val': None}
    meters['val'] = { field: tnt.meter.AverageValueMeter() for field in model_opt['log.fields'] }

    model_utils.evaluate(model, data[classFile_name], meters['val'], desc="test", y_cache=y_cache)
    # df = pd.DataFrame(data=None, columns=['loss','accuracy','precision','recall','f1-score'])
    # temp = []

    for field,meter in meters['val'].items():
        mean, std = meter.value()
        print("test {:s}: {:0.6f} +/- {:0.6f}".format(field, mean, 1.96 * std / math.sqrt(data_opt['data.test_episodes'])))
    #     temp.append(mean)
    # temp = np.array(temp).reshape(1,-1)
    # temp = pd.DataFrame(temp, columns=df.columns)
    # frames_into_csv(temp)
    if os.path.isfile(re_file):
        os.remove(re_file)
        os.remove(decision_val_file)
        os.remove(preLabel_file)

    f1 = open(re_file, 'a', newline='')
    writer1 = csv.writer(f1)
    writer1.writerows(y_cache['y_re'][0].reshape(-1, 1).tolist())
    f3 = open(decision_val_file, 'a', newline='')
    writer3 = csv.writer(f3)
    writer3.writerows(y_cache['decision_val'][0].tolist())
    f4 = open(preLabel_file, 'a', newline='')
    writer4 = csv.writer(f4)
    writer4.writerows(y_cache['predic_label'][0].reshape(-1, 1).tolist())
    f1.close()
    f3.close()
    f4.close()

    return meters


