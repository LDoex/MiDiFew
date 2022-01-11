import MiDiFewNets.dataLoader

def load(opt, splits):
    if opt['data.dataset'] == 'pipeline' or opt['data.dataset'] == 'kdd' or opt['data.dataset'] == 'watertank':
        ds = MiDiFewNets.dataLoader.pipeline.load(opt, splits)
    else:
        raise ValueError("Unknown dataset: {:s}".format(opt['data.dataset']))

    return ds

def fed_load(opt, splits, client_name):
    if opt['data.dataset'] == 'pipeline' or opt['data.dataset'] == 'kdd' or opt['data.dataset'] == 'watertank':
        ds = MiDiFewNets.dataLoader.fed_pipeline.load(opt, splits, client_name)
    else:
        raise ValueError("Unknown dataset: {:s}".format(opt['data.dataset']))

    return ds