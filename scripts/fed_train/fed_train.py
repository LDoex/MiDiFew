import os
import json
from functools import partial
from tqdm import tqdm
import pandas as pd
import numpy as np

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
import torchnet as tnt

from MiDiFewNets.fed_engine import Engine

import MiDiFewNets.utils.data as data_utils
import MiDiFewNets.utils.model as model_utils
import MiDiFewNets.utils.log as log_utils
import MiDiFewNets.utils.transfer_model as transfer_model


def fed_train_main(opt, net, global_parameters, client_name, comm_num, clients_best_model, clients_best_loss, clients_best_acc):
    if not os.path.isdir(opt['log.exp_dir']):
        os.makedirs(opt['log.exp_dir'])


    # # save opts
    # with open(os.path.join(opt['log.exp_dir'], client_name+'_opt.json'), 'w') as f:
    #     json.dump(opt, f)
    #     f.write('\n')

    trace_file = os.path.join(opt['log.exp_dir'], client_name+'_trace.txt')

    torch.manual_seed(1234)
    if opt['data.cuda']:
        torch.cuda.manual_seed(1234)

    if opt['data.trainval']:
        data = data_utils.fed_load(opt, ['trainval'], client_name)
        train_loader = data['trainval']
        val_loader = None
    else:
        data = data_utils.fed_load(opt, ['train', 'val'], client_name)
        train_loader = data['train']
        val_loader = data['val']

    if 'student' in opt['model.model_name']:
        if opt['train.isDistill'] == False:
            best_model_name = 'best_model.pt'

        else:
            best_model_name = 'best_model_withDistill.pt'

    else:
        best_model_name = 'best_teacher_model.pt'


    best_model_name = 'best_local_model.pt'
    model = net
    model.load_state_dict(global_parameters, strict=True)
    #model = transfer_model.freezParam(model)
    model.train(mode=True)


    # for child in net.children():
    #     for param in child.parameters():
    #         print(param.requires_grad)

    if opt['data.cuda']:
        model.cuda()

    engine = Engine()

    meters = {'train': {field: tnt.meter.AverageValueMeter() for field in opt['log.fields']}}

    if val_loader is not None:
        meters['val'] = {field: tnt.meter.AverageValueMeter() for field in opt['log.fields']}

    def on_start(state):
        if os.path.isfile(trace_file) and state['cur_comm_num']==0:
            os.remove(trace_file)
        state['scheduler'] = lr_scheduler.StepLR(state['optimizer'], opt['train.decay_every'], gamma=0.5)
    engine.hooks['on_start'] = on_start

    def on_start_epoch(state):
        for split, split_meters in meters.items():
            for field, meter in split_meters.items():
                meter.reset()

    engine.hooks['on_start_epoch'] = on_start_epoch

    def on_update(state):
        for field, meter in meters['train'].items():
            meter.add(state['output'][field])
    engine.hooks['on_update'] = on_update

    def on_end_epoch(hook_state, state):
        state['scheduler'].step()
        if val_loader is not None:
            # if 'best_loss' not in hook_state:
            #     hook_state['best_loss'] = np.inf
            if 'best_loss' not in hook_state:
                hook_state['best_loss'] = np.inf
            if 'wait' not in hook_state:
                hook_state['wait'] = 0

        if val_loader is not None:
            model_utils.evaluate(state['model'],
                                 val_loader,
                                 meters['val'],
                                 desc="Epoch {:d} valid".format(state['epoch']))

        meter_vals = log_utils.extract_meter_values(meters)
        print("Epoch {:02d}: {:s}".format(state['epoch'], log_utils.render_meter_values(meter_vals)))
        meter_vals['epoch'] = state['epoch']
        with open(trace_file, 'a') as f:
            json.dump(meter_vals, f)
            f.write('\n')



        if val_loader is not None:
            if meter_vals['val']['loss'] < state['clients_best_loss'][state['client_name']] and meter_vals['val']['acc'] > state['clients_best_acc'][state['client_name']]:
                state['clients_best_loss'][state['client_name']] = meter_vals['val']['loss']
                state['clients_best_acc'][state['client_name']] = meter_vals['val']['acc']
                print("==> best model (loss = {:0.6f}), saving model...".format(state['clients_best_loss'][state['client_name']]))
            # if meter_vals['val']['loss'] < hook_state['best_loss']:
            #     hook_state['best_loss'] = meter_vals['val']['loss']
            #     print("==> best model (loss = {:0.6f}), saving model...".format(hook_state['best_loss']))

                state['clients_best_model'][state['client_name']] = state['model'].state_dict()
                state['model'].cpu()

                torch.save(state['model'], os.path.join(opt['log.exp_dir'], client_name+'_'+best_model_name))

                if opt['data.cuda']:
                    state['model'].cuda()

                hook_state['wait'] = 0
            elif meter_vals['val']['acc'] > state['clients_best_acc'][state['client_name']]:
                state['clients_best_acc'][state['client_name']] = meter_vals['val']['acc']
                print("==> best model (acc = {:0.6f}), saving model...".format(state['clients_best_acc'][state['client_name']]))
            # if meter_vals['val']['loss'] < hook_state['best_loss']:
            #     hook_state['best_loss'] = meter_vals['val']['loss']
            #     print("==> best model (loss = {:0.6f}), saving model...".format(hook_state['best_loss']))

                state['clients_best_model'][state['client_name']] = state['model'].state_dict()
                state['model'].cpu()

                torch.save(state['model'], os.path.join(opt['log.exp_dir'], client_name+'_'+best_model_name))

                if opt['data.cuda']:
                    state['model'].cuda()

                hook_state['wait'] = 0
            else:
                hook_state['wait'] += 1

                if hook_state['wait'] > opt['train.patience']:
                    print("==> patience {:d} exceeded".format(opt['train.patience']))
                    state['stop'] = True
        else:
            state['model'].cpu()

            torch.save(state['model'], os.path.join(opt['log.exp_dir'], client_name+best_model_name))

            if opt['data.cuda']:
                state['model'].cuda()


    engine.hooks['on_end_epoch'] = partial(on_end_epoch, {})

    if not opt['model.midiFew']:
        sec_model = None

    teacher_model = None if 'student' in best_model_name or opt['train.isDistill'] == False else torch.load(os.path.join(opt['log.exp_dir'], 'best_teacher_model.pt'))

    return engine.train(
        teacher_model=teacher_model,
        model=model,
        loader=train_loader,
        optim_method=getattr(optim, opt['train.optim_method']),
        optim_config={'lr': opt['train.learning_rate'],
                      'weight_decay': opt['train.weight_decay']},
        max_epoch=opt['train.epochs'],
        cur_comm_num=comm_num,
        client_name=client_name,
        clients_best_loss=clients_best_loss,
        clients_best_acc=clients_best_acc,
        clients_best_model=clients_best_model
    )
