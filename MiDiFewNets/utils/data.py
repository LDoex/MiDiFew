import MiDiFewNets.dataLoader

def load(opt, splits):
    if opt['data.dataset'] == 'pipeline':
        ds = MiDiFewNets.dataLoader.pipeline.load(opt, splits)
    else:
        raise ValueError("Unknown dataset: {:s}".format(opt['data.dataset']))

    return ds