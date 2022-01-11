from tqdm import tqdm

class Engine(object):
    def __init__(self):
        hook_names = ['on_start', 'on_start_epoch', 'on_sample', 'on_forward',
                      'on_backward', 'on_end_epoch', 'on_update', 'on_end']

        self.hooks = { }
        for hook_name in hook_names:
            self.hooks[hook_name] = lambda state: None

    def train(self, **kwargs):
        state = {
            'teacher_moder': kwargs['teacher_model'],
            'model': kwargs['model'],
            'loader': kwargs['loader'],
            'optim_method': kwargs['optim_method'],
            'optim_config': kwargs['optim_config'],

            'max_epoch': kwargs['max_epoch'],
            'epoch': 0, # epochs done so far
            't': 0, # samples seen so far
            'batch': 0, # samples seen in current epoch
            'stop': False,
            'cur_comm_num': kwargs['cur_comm_num'],
            'client_name': kwargs['client_name'],
            'clients_best_loss': kwargs['clients_best_loss'],
            'clients_best_acc': kwargs['clients_best_acc'],
            'clients_best_model': kwargs['clients_best_model']
        }

        #Hierarchical lr
        # blocks = []
        # for child in state['model'].children():
        #     block_1 = list(map(id, child[0].parameters()))
        #     block_2 = list(map(id, child[1].parameters()))
        #     blocks.append(child[1].parameters())
        #     blocks.append(filter(lambda p: id(p) not in block_1+block_2, child.parameters()))
        #
        # params = [
        #     {"params": filter(lambda p: p.requires_grad, blocks[0]), "lr": state['optim_config']['lr']*0.5},
        #     {"params": filter(lambda p: p.requires_grad, blocks[1])},
        # ]

        params = [
            {"params": filter(lambda p: p.requires_grad, state['model'].parameters())}
        ]
        state['optimizer'] = state['optim_method'](params, **state['optim_config'])

        # state['optimizer'] = state['optim_method'](state['model'].parameters(), **state['optim_config'])

        self.hooks['on_start'](state)
        while state['epoch'] < state['max_epoch'] and not state['stop']:
            state['model'].train()

            self.hooks['on_start_epoch'](state)

            state['epoch_size'] = len(state['loader'])

            for sample in tqdm(state['loader'], desc="Epoch {:d} train".format(state['epoch'] + 1)):
                state['sample'] = sample
                self.hooks['on_sample'](state)

                state['optimizer'].zero_grad()

                loss, state['output'] = state['model'].loss(sample=state['sample'],
                                                            teacher_model=state['teacher_moder'],
                                                            y_cache = None)
                self.hooks['on_forward'](state)

                loss.backward()
                self.hooks['on_backward'](state)

                state['optimizer'].step()

                state['t'] += 1
                state['batch'] += 1
                self.hooks['on_update'](state)

            state['epoch'] += 1
            state['batch'] = 0
            self.hooks['on_end_epoch'](state)

        self.hooks['on_end'](state)

        return state['clients_best_model'][state['client_name']]
