from tqdm import tqdm

from MiDiFewNets.utils import filter_opt
from MiDiFewNets.models import get_model

def load(opt):
    model_opt = filter_opt(opt, 'model')
    model_name = model_opt['model_name']

    del model_opt['model_name']

    return get_model(model_name, model_opt)

def evaluate(model, sec_model, data_loader, meters, desc=None):
    model.eval()
    sec_model.eval()

    for field,meter in meters.items():
        meter.reset()

    if desc is not None:
        data_loader = tqdm(data_loader, desc=desc)

    for sample in data_loader:
        _, output = model.loss(sample=sample, sec_model=sec_model, teacher_model=None)
        for field, meter in meters.items():
            meter.add(output[field])

    return meters
