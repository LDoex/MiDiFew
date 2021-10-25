import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import csv
from torch.autograd import Variable
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
from MiDiFewNets.models import register_model

from .utils import euclidean_dist
from .utils import CosineMarginLoss

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

class midifewNet1d_teacher(nn.Module):
    def __init__(self, encoder):
        super(midifewNet1d_teacher, self).__init__()
        self.encoder = encoder

    def loss(self, **kwargs):
        sample = kwargs['sample']
        xs = Variable(sample['xs'])  # support
        xq = Variable(sample['xq'])  # query

        n_class = xs.size(0)
        assert xq.size(0) == n_class
        n_support = xs.size(1)
        n_query = xq.size(1)

        target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False)

        if xq.is_cuda:
            target_inds = target_inds.cuda()

        x = torch.cat([xs.view(n_class * n_support, *xs.size()[2:]),
                       xq.view(n_class * n_query, *xq.size()[2:])], 0)

        z = self.encoder.forward(x)
        # save z into txt
        result = np.array(z.cpu().detach().numpy())
        rows = []
        for i in range(result.shape[0]):
            rows.append(result[i])
        file = open("result.csv", 'w')
        writer = csv.writer(file, delimiter=',', lineterminator='\n')
        writer.writerows(rows)
        file.close()
        # np.savetxt('npresult1.csv', result, delimiter=',',  fmt=['%s']*result.shape[1], newline='\n')

        z_dim = z.size(-1)

        z_proto = z[:n_class * n_support].view(n_class, n_support, z_dim).mean(1)
        zq = z[n_class * n_support:]

        dists = euclidean_dist(zq, z_proto)

        log_p_y = F.log_softmax(-dists, dim=1).view(n_class, n_query, -1)

        loss_log = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()

        loss_val = loss_log

        _, y_hat = log_p_y.max(2)
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

        y_re = target_inds.squeeze()

        y_real = np.array(y_re.cpu()).reshape(-1)
        y_pred = np.array(y_hat.cpu()).reshape(-1)
        acc = accuracy_score(y_real, y_pred)  # TP+TN/(TP+FN+FP+TN)
        pre = precision_score(y_real, y_pred, average='binary')  # TP/TP+FP
        rec = recall_score(y_real, y_pred, average='binary')  # TP/TP+FN
        F1s = f1_score(y_real, y_pred, average='binary')  # 2*(pre*recall/(pre+recall))
        # F1s, pre, rec, TP = f_score(y_real, y_pred)

        return loss_val, {
            'loss': loss_val.item(),
            'acc': acc_val.item(),
            'Accuracy': acc,
            'Precision': pre,
            'Recall': rec,
            'F1': F1s
        }

class midifewNet1d_student(nn.Module):
    def __init__(self, encoder):
        super(midifewNet1d_student, self).__init__()
        self.encoder = encoder

    def loss(self, **kwargs):
        sample = kwargs['sample']
        teacher_model = kwargs['teacher_model']
        xs = Variable(sample['xs'])  # support
        xq = Variable(sample['xq'])  # query

        n_class = xs.size(0)
        assert xq.size(0) == n_class
        n_support = xs.size(1)
        n_query = xq.size(1)

        target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False)

        if xq.is_cuda:
            target_inds = target_inds.cuda()

        x = torch.cat([xs.view(n_class * n_support, *xs.size()[2:]),
                       xq.view(n_class * n_query, *xq.size()[2:])], 0)

        z = self.encoder.forward(x)
        # save z into txt
        result = np.array(z.cpu().detach().numpy())
        rows = []
        for i in range(result.shape[0]):
            rows.append(result[i])
        file = open("result.csv", 'w')
        writer = csv.writer(file, delimiter=',', lineterminator='\n')
        writer.writerows(rows)
        file.close()
        # np.savetxt('npresult1.csv', result, delimiter=',',  fmt=['%s']*result.shape[1], newline='\n')

        z_dim = z.size(-1)

        z_proto = z[:n_class * n_support].view(n_class, n_support, z_dim).mean(1)
        zq = z[n_class * n_support:]

        dists = euclidean_dist(zq, z_proto)

        log_p_y = F.log_softmax(-dists, dim=1).view(n_class, n_query, -1)

        loss_log = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()

        loss_val = loss_log

        _, y_hat = log_p_y.max(2)
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

        y_re = target_inds.squeeze()

        y_real = np.array(y_re.cpu()).reshape(-1)
        y_pred = np.array(y_hat.cpu()).reshape(-1)
        acc = accuracy_score(y_real, y_pred)  # TP+TN/(TP+FN+FP+TN)
        pre = precision_score(y_real, y_pred, average='binary')  # TP/TP+FP
        rec = recall_score(y_real, y_pred, average='binary')  # TP/TP+FN
        F1s = f1_score(y_real, y_pred, average='binary')  # 2*(pre*recall/(pre+recall))
        # F1s, pre, rec, TP = f_score(y_real, y_pred)

        return loss_val, {
            'loss': loss_val.item(),
            'acc': acc_val.item(),
            'Accuracy': acc,
            'Precision': pre,
            'Recall': rec,
            'F1': F1s
        }

@register_model('midifew_teacher_conv1d')
def load_protonet_conv1d(**kwargs):
    x_dim = kwargs['x_dim']
    hid_dim = kwargs['hid_dim']
    z_dim = kwargs['z_dim']

    def conv1d_block_3(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
    def conv1d_block_5(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

    encoder = nn.Sequential(
        conv1d_block_5(x_dim[0], 64),
        conv1d_block_3(64, 64),
        conv1d_block_3(64, 32),
        Flatten()
    )

    return midifewNet1d_teacher(encoder)

@register_model('midifew_student_conv1d')
def load_protonet_conv1d(**kwargs):
    x_dim = kwargs['x_dim']
    hid_dim = kwargs['hid_dim']
    z_dim = kwargs['z_dim']

    def conv1d_block_3(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
    def conv1d_block_5(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

    encoder = nn.Sequential(
        conv1d_block_5(x_dim[0], 64),
        conv1d_block_3(64, 64),
        conv1d_block_3(64, 32),
        Flatten()
    )

    return midifewNet1d_student(encoder)

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
