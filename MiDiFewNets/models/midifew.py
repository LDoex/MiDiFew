import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
from MiDiFewNets.models import register_model

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()


    def forward(self, x):
        return x.view(x.size(0),-1)

class midifewNet2d(nn.Module):
    def __init__(self, encoder):
        super(midifewNet2d, self).__init__()
        self.encoder = encoder

    def loss(self, sample):
        #计算最终loss并返回
        pass

class midifewNet1d(nn.Module):
    def __init__(self, encoder):
        super(midifewNet1d, self).__init__()
        self.encoder = encoder

    def loss(self, sample):
        pass

@register_model('protonet_conv1d')
def load_protonet_conv1d(**kwargs):
    x_dim = kwargs['x_dim']
    hid_dim = kwargs['hid_dim']
    z_dim = kwargs['z_dim']

    def conv1d_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

    encoder = nn.Sequential(
        conv1d_block(x_dim[0], 64),
        Flatten()
    )

    return midifewNet1d(encoder)

@register_model('protonet_conv2d')
def load_protonet_conv2d(**kwargs):
    x_dim = kwargs['x_dim']
    hid_dim = kwargs['hid_dim']
    z_dim = kwargs['z_dim']

    def conv2d_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    encoder = nn.Sequential(
        conv2d_block(x_dim[0], 64),
        Flatten()
    )

    return midifewNet2d(encoder)
