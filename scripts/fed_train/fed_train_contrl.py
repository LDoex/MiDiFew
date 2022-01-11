from fed_train import fed_train_main
import numpy as np
from tqdm import tqdm
import MiDiFewNets.utils.model as model_utils
import MiDiFewNets.utils.data as data_utils
import MiDiFewNets.utils.log as log_utils
import MiDiFewNets.utils.transfer_model as transfer_model
import torch
from fed_eval import fed_eval_main
import os
import json
import math
from torchsummaryX import summary
import copy
from scripts.fed_train.eval import eval_main

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def main(opt):
    torch.manual_seed(1234)
    if opt['data.cuda']:
        torch.cuda.manual_seed(1234)

    client_num = opt['fed.client_num']
    comm_num = opt['fed.comm_num']
    cfraction = opt['fed.cfraction']
    model_dir = os.path.join(opt['fed.global_save_path'], 'best_teacher_kdd.pt')

    clients_best_model = {'client{}'.format(i): None for i in range(client_num)}

    clients_best_loss = {'client{}'.format(i): np.inf for i in range(client_num)}
    clients_best_acc = {'client{}'.format(i): 0.0 for i in range(client_num)}

    # Postprocess arguments
    opt['model.x_dim'] = list(map(int, opt['model.x_dim'].split(',')))
    opt['log.fields'] = opt['log.fields'].split(',')

    num_in_comm = int(max(client_num * cfraction, 1))

    net = torch.load(model_dir)
    net = transfer_model.freezParam(net)
    net = transfer_model.extend_param(net)

    print(get_parameter_number(net))

    # net = model_utils.load(opt)

    global_trace_file = os.path.join(opt['fed.global_save_path'], 'global_trace_{}clients.txt'.format(num_in_comm))


    # copy weights
    global_parameters = net.state_dict()
    global_best_parameters = net.state_dict()
    # for key, var in net.state_dict().items():
    #     global_parameters[key] = var.clone()

    # for key, var in net.state_dict().items():
    #     global_best_parameters[key] = var.clone()

    best_model_name = 'best_global_model_{}clients.pt'.format(num_in_comm)
    best_loss = np.inf
    best_acc = 0

    # save opts
    with open(os.path.join(opt['log.exp_dir'], 'opt.json'), 'w') as f:
        json.dump(opt, f)
        f.write('\n')

    for i in range(comm_num):
        print("communicate round {}".format(i + 1))

        if i != 0 and i % opt['train.decay_every'] == 0:
            opt['train.learning_rate'] = opt['train.learning_rate']*0.5

        order = np.random.permutation(client_num)
        clients_in_comm = ['client{}'.format(i) for i in order[0:num_in_comm]]

        sum_parameters = None

        for client_i, client in enumerate(clients_in_comm):
            print("No.{} client".format(client))

            #返回最佳local模型权重参数
            if client_i==0:
                local_parameters = copy.deepcopy(fed_train_main(opt, copy.deepcopy(net), copy.deepcopy(global_best_parameters), client, i,
                                                  clients_best_model, clients_best_loss, clients_best_acc))
            else:
                local_parameters = copy.deepcopy(fed_train_main(opt, copy.deepcopy(net), copy.deepcopy(global_best_parameters), client, i,
                                                  clients_best_model, clients_best_loss, clients_best_acc))

            if sum_parameters is None:
                sum_parameters = copy.deepcopy(local_parameters)
                # for key, var in local_parameters.items():
                #     sum_parameters[key] = var.clone()
            else:
                for var in sum_parameters:
                    sum_parameters[var] = sum_parameters[var] + local_parameters[var]

        for var in global_parameters:
            global_parameters[var] = torch.div(sum_parameters[var], num_in_comm)
            # global_parameters[var] = (sum_parameters[var] / num_in_comm)

        with torch.no_grad():
            if (i + 1) % opt['fed.val_freq'] == 0:
                net.load_state_dict(global_parameters, strict=True)

                # meters = fed_eval_main('fed_test', opt, net)
                #
                test_meters = eval_main('test', opt, net)

                # meter_vals = log_utils.extract_meter_values(meters)

                test_meter_vals = log_utils.extract_meter_values(test_meters)

                if os.path.isfile(global_trace_file) and (i+1)/opt['fed.val_freq']==1:
                    os.remove(global_trace_file)

                with open(global_trace_file, 'a') as f:
                    json.dump(test_meter_vals, f)
                    f.write('\n')

                # if meter_vals['val']['acc'] > best_acc:
                #     best_acc = meter_vals['val']['acc']
                #     print("==> best global model (acc = {:0.6f}), saving model...".format(best_acc))
                #     # if meter_vals['val']['loss'] < hook_state['best_loss']:
                #     #     hook_state['best_loss'] = meter_vals['val']['loss']
                #     #     print("==> best model (loss = {:0.6f}), saving model...".format(hook_state['best_loss']))
                #
                #     global_best_parameters = copy.deepcopy(global_parameters)
                #     net.cpu()
                #
                #     torch.save(net, os.path.join(opt['fed.global_save_path'],opt['data.dataset']+'_'+best_model_name))
                #
                #     if opt['data.cuda']:
                #         net.cuda()

                global_best_parameters = copy.deepcopy(global_parameters)
                net.cpu()

                torch.save(net, os.path.join(opt['fed.global_save_path'], opt['data.dataset'] + '_' + best_model_name))

                if opt['data.cuda']:
                    net.cuda()