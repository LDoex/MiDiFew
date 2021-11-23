from fed_train import fed_train_main
import numpy as np
from tqdm import tqdm
import MiDiFewNets.utils.model as model_utils
import MiDiFewNets.utils.data as data_utils
import MiDiFewNets.utils.log as log_utils
import torch
from fed_eval import fed_eval_main
import os
import json
import math

def main(opt):
    client_num = opt['fed.client_num']
    comm_num = opt['fed.comm_num']
    cfraction = opt['fed.cfraction']

    clients_best_model = {'client{}'.format(i): None for i in range(client_num)}

    clients_best_loss = {'client{}'.format(i): np.inf for i in range(client_num)}

    # Postprocess arguments
    opt['model.x_dim'] = list(map(int, opt['model.x_dim'].split(',')))
    opt['log.fields'] = opt['log.fields'].split(',')

    num_in_comm = int(max(client_num * cfraction, 1))
    net = model_utils.load(opt)

    global_trace_file = os.path.join(opt['log.exp_dir'], 'global_trace.txt')

    global_best_parameters = {}
    global_parameters = {}
    for key, var in net.state_dict().items():
        global_parameters[key] = var.clone()

    for key, var in net.state_dict().items():
        global_best_parameters[key] = var.clone()

    best_model_name = 'best_global_model.pt'
    best_acc = 0

    # save opts
    with open(os.path.join(opt['log.exp_dir'], 'opt.json'), 'w') as f:
        json.dump(opt, f)
        f.write('\n')

    for i in range(comm_num):
        print("communicate round {}".format(i + 1))

        order = np.random.permutation(client_num)
        clients_in_comm = ['client{}'.format(i) for i in order[0:num_in_comm]]

        sum_parameters = None

        for client_i, client in enumerate(clients_in_comm):
            print("No.{} client".format(client_i))

            #返回最佳local模型权重参数
            if client_i==0:
                local_parameters = fed_train_main(opt, net, global_best_parameters, client, i, clients_best_model, clients_best_loss)
            else:
                local_parameters = fed_train_main(opt, net, global_parameters, client, i, clients_best_model, clients_best_loss)

            if sum_parameters is None:
                sum_parameters = {}
                for key, var in local_parameters.items():
                    sum_parameters[key] = var.clone()
            else:
                for var in sum_parameters:
                    sum_parameters[var] = sum_parameters[var] + local_parameters[var]

        for var in global_parameters:
            global_parameters[var] = (sum_parameters[var] / num_in_comm)

        with torch.no_grad():
            if (i + 1) % opt['fed.val_freq'] == 0:
                net.load_state_dict(global_parameters, strict=True)

                meters = fed_eval_main(opt, net)

                meter_vals = log_utils.extract_meter_values(meters)

                if os.path.isfile(global_trace_file) and (i+1)/opt['fed.val_freq']==1:
                    os.remove(global_trace_file)

                with open(global_trace_file, 'a') as f:
                    json.dump(meter_vals, f)
                    f.write('\n')

                if meter_vals['val']['acc'] > best_acc:
                    best_acc = meter_vals['val']['acc']
                    print("==> best global model (acc = {:0.6f}), saving model...".format(best_acc))
                    # if meter_vals['val']['loss'] < hook_state['best_loss']:
                    #     hook_state['best_loss'] = meter_vals['val']['loss']
                    #     print("==> best model (loss = {:0.6f}), saving model...".format(hook_state['best_loss']))

                    global_best_parameters = global_parameters
                    net.cpu()

                    torch.save(net, os.path.join(opt['fed.global_save_path'],opt['data.dataset']+'_'+best_model_name))

                    if opt['data.cuda']:
                        net.cuda()