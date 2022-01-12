import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import csv
from torch.autograd import Variable
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
from MiDiFewNets.models import register_model
import math
import os

from .utils import euclidean_dist
from .utils import CosineMarginLoss

class Dropout(nn.Module):
    def __init__(self, param):
        super(Dropout, self).__init__()
        self.param = param


    def forward(self, input):
        m = nn.Dropout(self.param)
        return m(input)

# train the Net
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0),-1)

# train the Net
class Concat(nn.Module):
    def __init__(self, n_support, n_query, n_class):
        super(Concat, self).__init__()
        self.n_support = n_support
        self.n_query = n_query
        self.n_class = n_class

    def forward(self, x):
        x0 = x[:self.n_support].clone()
        x1 = x[self.n_support: self.n_class * self.n_support].clone()
        xq = x[self.n_class * self.n_support:].clone()
        total_x = None
        for i in range(self.n_class * self.n_query):
            dim_0 = xq[i].shape[0]
            for j in range(self.n_support):
                x0_ = torch.cat([x0[j].clone(), xq[i].clone()], -1).view(1, dim_0, -1)
                if total_x is None:
                    total_x = x0_
                else:
                    total_x = torch.cat([total_x, x0_])
            for j in range(self.n_support):
                x1_ = torch.cat([x1[j].clone(), xq[i].clone()], -1).view(1, dim_0, -1)
                total_x = torch.cat([total_x, x1_])

        return total_x
# train the Net
class singleConcat(nn.Module):
    def __init__(self, n_support, n_query, n_class):
        super(singleConcat, self).__init__()
        self.n_support = n_support
        self.n_query = n_query
        self.n_class = n_class

    def forward(self, x):
        return torch.cat([x[0].clone().unsqueeze(0), x[1].clone().unsqueeze(0)], -1)

class Resize(nn.Module):
    def __init__(self):
        super(Resize, self).__init__()

    def forward(self, x, n_class, n_support, data_dim, need_dim):
        if need_dim*need_dim>data_dim:
            zero = torch.zeros(x.size(0), need_dim*need_dim-data_dim)
            x = torch.cat([x, zero], dim=1).view(n_class, n_support, need_dim, need_dim)
        return x

class midifewFinalNet(nn.Module):
    def __init__(self, encoders):
        super(midifewFinalNet, self).__init__()
        self.encoders = encoders
        self.linear = nn.Linear(26, 16)
        self.flatten = Flatten()
        self.resize_block = Resize()

    def forward(self, x, n_class, n_support, n_query):
        x = self.encoders[0].forward(x)
        xs = self.resize_block.forward(x[:n_class * n_support], n_class, n_support, x.size(-1),
                                       math.ceil(x.size(-1)**0.5))
        xq = x[n_class * n_support:]
        xs = self.encoders[1].forward(xs)
        return torch.cat([xs, xq], 0)

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

        for i in range(n_query):
            z = torch.cat()

        # z = self.forward(x, n_class, n_support, n_query)

        z_dim = z.size(-1)

        z_proto = z[:n_class]
        zq = z[n_class:]

        dists = euclidean_dist(zq, z_proto)

        log_p_y = F.log_softmax(-dists, dim=1)

        log_p_y = log_p_y.view(n_class, n_query, -1)

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

        meters = {
            'loss': loss_val.item(),
            'acc': acc_val.item(),
            'Accuracy': acc,
            'Precision': pre,
            'Recall': rec,
            'F1': F1s
        }

        return loss_val, meters

class midifewNet2d(nn.Module):
    def __init__(self, encoder):
        super(midifewNet2d, self).__init__()
        self.encoder = encoder

    def loss(self, sample, data):
        #计算最终loss并返回
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

        data_dim = data.size(-1)
        need_dim = math.ceil(data_dim**0.5)
        zq = data[n_class * n_support:]

        if need_dim*need_dim>data_dim:
            zero = torch.zeros(data.size(0), need_dim*need_dim-data_dim)
            data = torch.cat([data, zero], dim=1)

        input_data = data[:n_class * n_support].view(n_class, n_support, need_dim, need_dim)
        z_proto = self.encoder.forward(input_data)



        dists = euclidean_dist(zq, z_proto)

        log_p_y = F.log_softmax(-dists, dim=1)

        log_p_y = log_p_y.view(n_class, n_query, -1)

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

        # x = x.permute(0, 2, 1)
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

        log_p_y = F.log_softmax(-dists, dim=1)

        log_p_y = log_p_y.view(n_class, n_query, -1)

        loss_log = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()



        _, y_hat = log_p_y.max(2)
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

        ind_FP = [i for i,val in enumerate(y_hat[0]) if val!=0]
        y_re = target_inds.squeeze()

        ind_FP_tensor = torch.tensor(ind_FP).view(-1,1)

        'gather(dim, tensor_index)用法：用于按dim维度获取tensor值，示例：' \
        'tensor=[[1,2,3],[4,5,6],[7,8,9]]' \
        'tensor_inds = [[0,1,2],[0,2,1],[1,0,2]]' \
        'tensor.gather(0,tensor_inds)->[[1,5,9],[1,8,6],[4,2,9]] 即，dim维度上按给定的index确定，其余维度按当前index的下标顺序确定'
        FP_loss = 0 if len(ind_FP)==0 else -log_p_y[0].gather(0, ind_FP_tensor).mean()

        loss_val = 1*loss_log+0*FP_loss

        y_real = np.array(y_re.cpu()).reshape(-1)
        y_pred = np.array(y_hat.cpu()).reshape(-1)
        acc = accuracy_score(y_real, y_pred)  # TP+TN/(TP+FN+FP+TN)
        pre = precision_score(y_real, y_pred, average='binary')  # TP/TP+FP
        rec = recall_score(y_real, y_pred, average='binary')  # TP/TP+FN
        F1s = f1_score(y_real, y_pred, average='binary')  # 2*(pre*recall/(pre+recall))
        # F1s, pre, rec, TP = f_score(y_real, y_pred)

        if kwargs['y_cache'] is not None:
            saveForRoc(dists, n_class, n_query, kwargs['y_cache'])

        meters = {
        'loss': loss_val.item(),
        'acc': acc_val.item(),
        'Accuracy': acc,
        'Precision': pre,
        'Recall': rec,
        'F1': F1s
        }

        # return loss_val, {
        #     'loss': loss_val.item(),
        #     'acc': acc_val.item(),
        #     'Accuracy': acc,
        #     'Precision': pre,
        #     'Recall': rec,
        #     'F1': F1s
        # }
        return loss_val, meters



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

        loss_student = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()

        _, y_hat = log_p_y.max(2)

        ind_FP = [i for i, val in enumerate(y_hat[0]) if val != 0]
        y_re = target_inds.squeeze()

        ind_FP_tensor = torch.tensor(ind_FP).view(-1, 1)

        'gather(dim, tensor_index)用法：用于按dim维度获取tensor值，示例：' \
        'tensor=[[1,2,3],[4,5,6],[7,8,9]]' \
        'tensor_inds = [[0,1,2],[0,2,1],[1,0,2]]' \
        'tensor.gather(0,tensor_inds)->[[1,5,9],[1,8,6],[4,2,9]] 即，dim维度上按给定的index确定，其余维度按当前index的下标顺序确定'
        FP_loss = 0 if len(ind_FP) == 0 else -log_p_y[0].gather(0, ind_FP_tensor).mean()

        # distill setting
        # if teacher_model is exist, setting args
        if teacher_model is not None:
            w_teacher = 0.5
            w_student = 0.5
            w_FP = 0.00
            teacher_model.eval()
            T = 20
            if xq.is_cuda:
                teacher_model.cuda()

            teacher_z = teacher_model.encoder.forward(x)

            teacher_z_dim = teacher_z.size(-1)

            teacher_z_proto = teacher_z[:n_class * n_support].view(n_class, n_support, teacher_z_dim).mean(1)
            teacher_zq = teacher_z[n_class * n_support:]

            teacher_dists = euclidean_dist(teacher_zq, teacher_z_proto)

            soft_label = F.softmax(teacher_dists/T, dim=1)
            student_label = F.log_softmax(dists/T, dim=1)

            loss_ditill = nn.KLDivLoss(reduction="batchmean")(student_label, soft_label) * T * T

            loss_val = loss_ditill * w_teacher + loss_student * w_student + FP_loss * w_FP

        else:
            w_student = 1
            w_FP = 0.0
            loss_val = loss_student * w_student + FP_loss * w_FP


        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

        y_re = target_inds.squeeze()

        if kwargs['y_cache'] is not None:
            saveForRoc(dists, n_class, n_query, kwargs['y_cache'])

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

def saveForRoc(dists, n_class, n_query, y_cache):
    soft_val = F.softmax(-dists, dim=1).view(n_class, n_query, -1)
    soft_target_inds = torch.zeros(n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
    soft_target_inds = Variable(soft_target_inds, requires_grad=False)
    y_soft = soft_val.gather(2, soft_target_inds).squeeze().view(-1)

    target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
    target_inds = Variable(target_inds, requires_grad=False)

    y_real = np.array(target_inds.squeeze().cpu()).reshape(-1)
    y_pred = np.array(y_soft.detach().cpu()).reshape(-1)
    _, y_hat = soft_val.max(2)
    decision_val = np.array(soft_val.detach().cpu()).reshape(-1,n_class)
    y_hat = np.array(y_hat.squeeze().cpu()).reshape(-1)
    carry = np.ones(y_hat.shape[0])
    y_hat = carry+y_hat
    y_real = carry+y_real


    if len(y_cache['predic_label'])==0:
        y_cache['predic_label'].append(y_hat)
        y_cache['decision_val'].append(decision_val)
        y_cache['y_re'].append(y_real)
    else:
        y_cache['predic_label'][0] = np.append(y_cache['predic_label'][0], y_hat)
        y_cache['decision_val'][0] = np.append(y_cache['decision_val'][0], decision_val, axis=0)
        y_cache['y_re'][0] = np.append(y_cache['y_re'][0], y_real)

    # re_file = os.path.join(file_path, 'y_real.csv')
    # pre_file = os.path.join(file_path, 'y_pred.csv')
    #
    # f1 = open(re_file, 'a', newline='')
    # writer1 = csv.writer(f1)
    # writer1.writerow(y_real)
    # f2 = open(pre_file, 'a', newline='')
    # writer2 = csv.writer(f2)
    # writer2.writerow(y_pred)
    # f1.close()
    # f2.close()





@register_model('midifew_teacher_conv1d')
def load_protonet_conv1d(**kwargs):
    x_dim = kwargs['x_dim']
    hid_dim = kwargs['hid_dim']
    z_dim = kwargs['z_dim']

    def conv1d_block_3(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2)
        )
    def conv1d_block_5(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=2),
            nn.BatchNorm1d(out_channels, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

    encoder = nn.Sequential(
        conv1d_block_3(x_dim[0], 32),
        conv1d_block_3(32, 32),
        Flatten()
    )

    return midifewNet1d_teacher(encoder)

@register_model('midifew_teacher_conv2d')
def load_protonet_conv1d(**kwargs):
    x_dim = kwargs['x_dim']
    hid_dim = kwargs['hid_dim']
    z_dim = kwargs['z_dim']
    print(x_dim[0])

    def conv2d_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels, momentum=1, affine=True),
            nn.ReLU()
        )

    def conv2d_block_3(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
    def conv1d_block_5(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=2),
            nn.BatchNorm1d(out_channels, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

    encoder = nn.Sequential(
        conv2d_block(x_dim[0], 32),
        conv2d_block_3(32, 16),
        conv2d_block(16, 16),
        conv2d_block(16, 16),
        Flatten()
    )

    return midifewNet1d_teacher(encoder)

@register_model('midifew_student_conv1d')
def load_protonet_conv1d(**kwargs):
    x_dim = kwargs['x_dim']
    hid_dim = kwargs['hid_dim']
    z_dim = kwargs['z_dim']

    def conv1d_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels, momentum=1, affine=True),
            nn.ReLU(),
        )

    def conv1d_block_3(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
    def linear(in_channels, out_channels):
        return nn.Linear(in_channels,out_channels)
    def conv1d_block_5(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=2),
            nn.BatchNorm1d(out_channels, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

    encoder = nn.Sequential(
        # linear(20, 32),
        # linear(32, 32),
        # linear(32, 32),

        conv1d_block(x_dim[0], 16),
        conv1d_block(16, 16),
        conv1d_block(16, 16),
        conv1d_block_3(16, 8),
        conv1d_block_3(8, 8),
        conv1d_block_3(8, 8),
        # conv1d_block_3(16, 16),
        #conv1d_block_3(32, 32),
        # conv1d_block_3(32, 32),
        # conv1d_block_3(32, 16),


        #conv1d_block_3(8, 8),
        #linear(26, 32),
        Flatten()
    )

    return midifewNet1d_student(encoder)

@register_model('midifew_student_conv2d')
def load_protonet_conv1d(**kwargs):
    x_dim = kwargs['x_dim']
    hid_dim = kwargs['hid_dim']
    z_dim = kwargs['z_dim']
    print(x_dim[0])

    def conv2d_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels, momentum=1, affine=True),
            nn.ReLU()
        )

    def conv2d_block_3(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
    def conv1d_block_5(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=2),
            nn.BatchNorm1d(out_channels, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

    encoder = nn.Sequential(
        conv2d_block(x_dim[0], 16),
        conv2d_block_3(16, 16),
        Flatten()
    )

    return midifewNet1d_student(encoder)

# @register_model('midifew_final')
# def load_midifew_conv2d(**kwargs):
#     x_dim = kwargs['x_dim']
#     hid_dim = kwargs['hid_dim']
#     z_dim = kwargs['z_dim']
#
#     def conv1d_block_3(in_channels, out_channels):
#         return nn.Sequential(
#             nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm1d(out_channels, momentum=1, affine=True),
#             nn.ReLU(),
#             nn.MaxPool1d(2)
#         )
#
#     def conv2d_block(in_channels, out_channels):
#         return nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )
#
#     def linear_block(in_channels, out_channels):
#         return nn.Linear(in_channels, out_channels)
#
#     encoder1 = nn.Sequential(
#         conv1d_block_3(x_dim[0], 32),
#         conv1d_block_3(32, 16),
#         # Dropout(),
#         Flatten()
#     )
#
#     encoder2 = nn.Sequential(
#         Flatten(),
#         linear_block(500, 512),
#         linear_block(512, 256),
#         linear_block(256, 128),
#         linear_block(128, 96)
#     )
#
#
#     encoders = nn.ModuleList([encoder1, encoder2])
#
#     return midifewFinalNet(encoders)

@register_model('midifew_preteacher_conv1d')
def load_protonet_conv1d(**kwargs):
    x_dim = kwargs['x_dim']
    hid_dim = kwargs['hid_dim']
    z_dim = kwargs['z_dim']
    print(x_dim[0])

    def conv1d_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels, momentum=1, affine=True),
            nn.ReLU()
        )

    def conv1d_block_3(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2)
        )
    def conv1d_block_5(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=2),
            nn.BatchNorm1d(out_channels, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

    encoder = nn.Sequential(
        conv1d_block(x_dim[0], 32),
        conv1d_block(32, 32),
        conv1d_block_3(32, 16),
        conv1d_block_3(16, 16),
        Flatten()
    )

    return midifewNet1d_teacher(encoder)

@register_model('midifew_preteacher_conv2d')
def load_protonet_conv1d(**kwargs):
    x_dim = kwargs['x_dim']
    hid_dim = kwargs['hid_dim']
    z_dim = kwargs['z_dim']
    print(x_dim[0])

    def conv2d_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels, momentum=1, affine=True),
            nn.ReLU()
        )

    def conv2d_block_3(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
    def conv1d_block_5(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=2),
            nn.BatchNorm1d(out_channels, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

    encoder = nn.Sequential(
        conv2d_block(x_dim[0], 32),
        conv2d_block_3(32, 16),
        Flatten()
    )

    return midifewNet1d_teacher(encoder)

@register_model('extend_model')
def load_protonet_conv1d(**kwargs):
    x_dim = kwargs['x_dim']

    def conv1d_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels, momentum=1, affine=True),
            nn.ReLU(),
        )

    def conv1d_block_3(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2)
        )

    def conv2d_block_3(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
    def conv2d_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels, momentum=1, affine=True),
            nn.ReLU(),
        )

    encoder = nn.Sequential(
        conv2d_block(16, 16),
        conv2d_block(16, 16),
        Flatten()
    )

    return encoder
