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
            'sec_model': kwargs['sec_model'],
            'loader': kwargs['loader'],
            'optim_method': kwargs['optim_method'],
            'optim_config': kwargs['optim_config'],
            'sec_optim_config': kwargs['sec_optim_config'],
            'max_epoch': kwargs['max_epoch'],
            'epoch': 0, # epochs done so far
            't': 0, # samples seen so far
            'batch': 0, # samples seen in current epoch
            'stop': False
        }

        state['optimizer'] = state['optim_method'](state['model'].parameters(), **state['optim_config'])
        state['sec_optimizer'] = state['optim_method'](state['sec_model'].parameters(), **state['sec_optim_config']) if state['sec_model'] else None

        self.hooks['on_start'](state)
        while state['epoch'] < state['max_epoch'] and not state['stop']:
            state['model'].train()

            self.hooks['on_start_epoch'](state)

            state['epoch_size'] = len(state['loader'])

            for sample in tqdm(state['loader'], desc="Epoch {:d} train".format(state['epoch'] + 1)):
                state['sample'] = sample
                self.hooks['on_sample'](state)

                state['optimizer'].zero_grad()

                if state['sec_optimizer']:
                    state['sec_optimizer'].zero_grad()

                loss, state['output'] = state['model'].loss(sample=state['sample'],
                                                            teacher_model=state['teacher_moder'],
                                                            sec_optimizer=state['sec_optimizer'],
                                                            sec_model=state['sec_model'])
                self.hooks['on_forward'](state)

                loss.backward()
                self.hooks['on_backward'](state)

                state['optimizer'].step()

                if state['sec_optimizer']:
                    state['sec_optimizer'].step()

                state['t'] += 1
                state['batch'] += 1
                self.hooks['on_update'](state)

            state['epoch'] += 1
            state['batch'] = 0
            self.hooks['on_end_epoch'](state)

        self.hooks['on_end'](state)
