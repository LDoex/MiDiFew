from MiDiFewNets.models import get_model
import torch
def freezParam(net):
    ct = 0
    for child in net.children():
        while ct < 2:
            for param in child[ct].parameters():
                param.requires_grad = False
            ct += 1
    return net

def extend_param(net):
    model = get_model('extend_model', model_opt={'x_dim': 16})
    for child in net.children():
        del child[-1]
        l = len(child)
        for name,m_child in model.named_children():
            child.add_module(str(int(name)+l),m_child)
    return net