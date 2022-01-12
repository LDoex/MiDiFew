import MiDiFewNets.utils.data as data_utils
import argparse
import torch.nn as nn
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.autograd import Variable
from tqdm import trange
import torchnet as tnt
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
import numpy as np
import os
import math

parser = argparse.ArgumentParser(description='Train FC_Net')

default_dataset = 'pipeline'
parser.add_argument('--data.dataset', type=str, default=default_dataset, metavar='DS',
                    help="data set name (default: {:s})".format(default_dataset))
default_split = 'vinyals'
parser.add_argument('--data.split', type=str, default=default_split, metavar='SP',
                    help="split name (default: {:s})".format(default_split))
parser.add_argument('--data.way', type=int, default=2, metavar='WAY',
                    help="number of classes per episode (default: 60)")
parser.add_argument('--data.shot', type=int, default=1, metavar='SHOT',
                    help="number of support examples per class (default: 5)")
parser.add_argument('--data.query', type=int, default=30, metavar='QUERY',
                    help="number of query examples per class (default: 5)")
parser.add_argument('--data.test_way', type=int, default=2, metavar='TESTWAY',
                    help="number of classes per episode in test. 0 means same as data.way (default: 5)")
parser.add_argument('--data.test_shot', type=int, default=1, metavar='TESTSHOT',
                    help="number of support examples per class in test. 0 means same as data.shot (default: 0)")
parser.add_argument('--data.test_query', type=int, default=30, metavar='TESTQUERY',
                    help="number of query examples per class in test. 0 means same as data.query (default: 15)")
parser.add_argument('--data.train_episodes', type=int, default=20, metavar='NTRAIN',
                    help="number of train episodes per epoch (default: 100)")
parser.add_argument('--data.test_episodes', type=int, default=20, metavar='NTEST',
                    help="number of test episodes per epoch (default: 100)")
parser.add_argument('--data.trainval', action='store_true', default=False, help="run in train+validation mode (default: False)")
#parser.add_argument('--data.sequential', action='store_true', default=False, help="use sequential sampler instead of episodic (default: False)")
parser.add_argument('--data.cuda', action='store_true', default=False, help="run in CUDA mode (default: False)")

# log args
default_fields = 'loss,Accuracy,Precision,Recall,F1'
parser.add_argument('--log.fields', type=str, default=default_fields, metavar='FIELDS',
                    help="fields to monitor during training (default: {:s})".format(default_fields))
default_exp_dir = './FC_results'
parser.add_argument('--log.exp_dir', type=str, default=default_exp_dir, metavar='EXP_DIR',
                    help="directory where experiments should be saved (default: {:s})".format(default_exp_dir))

args = vars(parser.parse_args())


class conv1d_block1_3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv1d_block1_3, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=0),
            nn.BatchNorm1d(out_channels, momentum=1, affine=True),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.seq(x)

class conv1d_block2_3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv1d_block2_3, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=0),
            nn.BatchNorm1d(out_channels, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2)
        )
    def forward(self, x):
        return self.seq(x)

#对应文中Concatenation操作部分
class Concat(nn.Module):
    def __init__(self, n_support, n_query, n_class):
        super(Concat, self).__init__()
        self.n_support = n_support
        self.n_query = n_query
        self.n_class = n_class

    def forward(self, x):
        #分别提取标签为0、1的support样本
        x0 = x[:self.n_support]
        x1 = x[self.n_support: self.n_class * self.n_support]
        #提取query样本
        xq = x[self.n_class * self.n_support:]

        total_x = None
        #对xq中的每个样本x_q，将x_q与x0和x1中的每个样本横向拼接形成新的张量
        for i in range(self.n_class * self.n_query):
            dim_0 = xq[i].shape[0]
            #将x0中的每个样本与x_q拼接并加入total_x
            for j in range(self.n_support):
                x0_ = torch.cat([x0[j], xq[i]], -1).view(1, dim_0, -1)
                if total_x is None:
                    total_x = x0_
                else:
                    total_x = torch.cat([total_x, x0_])
            # 将x1中的每个样本与x_q拼接并加入total_x
            for j in range(self.n_support):
                x1_ = torch.cat([x1[j], xq[i]], -1).view(1, dim_0, -1)
                total_x = torch.cat([total_x, x1_])

        return total_x

# class FC_Net(nn.Module):
#     def __init__(self):
#         super(FC_Net, self).__init__()
#         self.block1 = conv1d_block1_3(1, 8)
#         self.block2 = nn.Dropout(0.4)
#         self.block3 = conv1d_block1_3(8, 8)
#         self.block4 = nn.Dropout(0.4)
#         self.block5 = Concat(10, 20, 2)
#         self.block6 = conv1d_block2_3(8, 8)
#         self.block7 = nn.Dropout(0.2)
#         self.block8 = conv1d_block2_3(8, 8)
#         self.block9 = nn.Dropout(0.2)
#
#         self.linear1 = nn.Linear(80, 64)
#         self.linear2 = nn.Linear(64, 1)
#         # self.sigmoid = nn.Sigmoid()
#         self.dense = nn.Flatten()
#
#     def forward(self, x):
#         x = self.block1(x)
#         x = self.block2(x)
#         x = self.block3(x)
#         x = self.block4(x)
#         x = self.block5(x)
#         x = self.block6(x)
#         x = self.block7(x)
#         x = self.block8(x)
#         x = self.block9(x)
#         #x = x.view(-1, self.num_flat_features(x))
#         x = self.dense(x)
#         x = self.linear1(x)
#         x = self.linear2(x)
#         # x = x.view(40, 2, 10).mean(-1)
#         # x = self.sigmoid(x)
#         return x
#
#     def num_flat_features(self, x):
#         size = x.size()[1:]  # all dimensions except the batch dimension
#         num_features = 1
#         for s in size:
#             num_features *= s
#         return num_features

class F_Net(nn.Module):
    def __init__(self):
        super(F_Net, self).__init__()
        self.block1 = conv1d_block1_3(1, 64)
        self.block2 = nn.Dropout(0)
        self.block3 = conv1d_block1_3(64, 64)
        self.block4 = nn.Dropout(0)
        # self.block5 = Concat(10, 20, 2)
        # self.block6 = conv1d_block2_3(8, 8)
        # self.block7 = nn.Dropout(0.2)
        # self.block8 = conv1d_block2_3(8, 8)
        # self.block9 = nn.Dropout(0.2)
        #
        # self.linear1 = nn.Linear(80, 64)
        # self.linear2 = nn.Linear(64, 1)
        # # self.sigmoid = nn.Sigmoid()
        # self.dense = nn.Flatten()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        # x = x.view(40, 2, 10).mean(-1)
        # x = self.sigmoid(x)
        return x


class C_Net(nn.Module):
    def __init__(self):
        super(C_Net, self).__init__()
        # self.block1 = conv1d_block1_3(1, 8)
        # self.block2 = nn.Dropout(0.4)
        # self.block3 = conv1d_block1_3(8, 8)
        # self.block4 = nn.Dropout(0.4)
        # self.block5 = Concat(10, 20, 2)
        self.block6 = conv1d_block2_3(128, 64)
        self.block7 = nn.Dropout(0)
        self.block8 = conv1d_block2_3(64, 64)
        self.block9 = nn.Dropout(0)

        self.linear1 = nn.Linear(256, 64)
        self.linear2 = nn.Linear(64, 1)
        # self.sigmoid = nn.Sigmoid()
        self.dense = nn.Flatten()

    def forward(self, x):
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        # x = x.view(40, 2, 10).mean(-1)
        # x = self.sigmoid(x)
        return x



def train(F_model, C_model, train_loader, device, F_optimizer, C_optimizer):
    torch.manual_seed(1234)
    model1 = F_model.to(device)
    model1.cpu()
    model1.train()
    model2 = C_model.to(device)
    model2.cpu()
    model2.train()

    total_loss = 0.0
    avg_loss = 0.0
    correct = 0.0
    acc = 0.0

    with tqdm(train_loader) as episodes:
        for idx, data in enumerate(episodes):
            F_optimizer.zero_grad()
            C_optimizer.zero_grad()
            #假如设置n_support=10，n_class=2，n_query=20
            xs = Variable(data['xs'])  # support样本集合，用于小样本的training，按假设中的设置，
            # 共有n_class*n_support=20个样本，且前10个全是标签为0的样本，后10个全是标签为1的样本

            xq = Variable(data['xq']) # query样本集合，用于小样本的validation，按假设中的设置，
            # 共有n_class*n_query=40个样本，且前20个全是标签为0的样本，后20个全是标签为1的样本


            n_class = xs.size(0)
            assert xq.size(0) == n_class
            n_support = xs.size(1)
            n_query = xq.size(1)



            # #把xs和xq拼接成x，一次性传入模型进行处理
            # x = torch.cat([xs.view(n_class * n_support, *xs.size()[2:]),
            #                xq.view(n_class * n_query, *xq.size()[2:])], 0)

            #按给定的n_query设置标签
            target0 = torch.zeros(n_query, dtype=torch.int64)
            target1 = torch.ones(n_query, dtype=torch.int64)
            targets = torch.concat([target0, target1], 0)
            # binary_targets0 = torch.range(n_class-1, 0, -1).view(1, n_class).repeat(n_query, 1)
            # binary_targets1 = torch.range(0, n_class-1).view(1, n_class).repeat(n_query, 1)
            # binary_targets = torch.concat([binary_targets0, binary_targets1], 0)
            # binary_targets = torch.autograd.Variable(binary_targets)
            #
            # loss_fn = nn.MSELoss()
            # loss_val = loss_fn(z, targets.float())

            #target indexs，方便从预测值pred中取对应的张量
            target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
            target_inds = Variable(target_inds, requires_grad=False)


            ###
            xs_features = model1(xs.view(n_class * n_support, *xs.size()[2:])) #()
            xs_features = xs_features.view(n_class, n_support, 64, -1)
            xs_features = torch.sum(xs_features, 1).squeeze(1)
            xq_features = model1(xq.view(n_class * n_query, *xq.size()[2:]))

            xs_features_ext = xs_features.unsqueeze(0)
            xs_features_ext = xs_features_ext.repeat(n_query*n_class, 1, 1, 1)
            xq_features_ext = xq_features.unsqueeze(0).repeat(n_class, 1, 1, 1)
            xq_features_ext = torch.transpose(xq_features_ext, 0, 1)

            relation_pairs = torch.cat((xs_features_ext, xq_features_ext), 2)
            relation_pairs = relation_pairs.view(n_class*n_class*n_query, 128, -1)
            pred = model2(relation_pairs).view(-1, n_class)
            ###


            _, y_hat = pred.max(-1)

            #计算loss
            loss_fn = nn.MSELoss()
            one_hot_targets = Variable(torch.zeros(pred.size(0), n_class).scatter_(1, targets.view(targets.size(0), 1), 1))
            loss = loss_fn(pred, one_hot_targets)

            loss.backward()
            F_optimizer.step()
            C_optimizer.step()

            #评价指标计算
            total_loss += loss
            avg_loss = total_loss / (idx + 1)
            pred_ = pred
            pred_ = pred_.argmax(dim=1)
            correct += pred_.eq(targets.view_as(pred_)).sum()
            acc = correct / len(train_loader.dataset) * 100

def val(F_model, C_model, test_loader, device, best_acc):
    if not os.path.isdir(args['log.exp_dir']):
        os.makedirs(args['log.exp_dir'])

    model1 = F_model.to(device)
    model1.cpu()
    model1.eval()
    model2 = C_model.to(device)
    model2.cpu()
    model2.eval()

    total_loss = 0.0
    avg_loss = 0.0
    correct = 0.0
    acc = 0.0
    meters = {'train': {field: tnt.meter.AverageValueMeter() for field in args['log.fields']}}

    with tqdm(test_loader) as episodes:
        for idx, data in enumerate(episodes):
            xs = Variable(data['xs'])  # support
            xq = Variable(data['xq'])

            n_class = xs.size(0)
            assert xq.size(0) == n_class
            n_support = xs.size(1)
            n_query = xq.size(1)

            x = torch.cat([xs.view(n_class * n_support, *xs.size()[2:]),
                           xq.view(n_class * n_query, *xq.size()[2:])], 0)
            target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
            target_inds = Variable(target_inds, requires_grad=False)

            ###
            xs_features = model1(xs.view(n_class * n_support, *xs.size()[2:]))  # ()
            xs_features = xs_features.view(n_class, n_support, 64, -1)
            xs_features = torch.sum(xs_features, 1).squeeze(1)
            xq_features = model1(xq.view(n_class * n_query, *xq.size()[2:]))

            xs_features_ext = xs_features.unsqueeze(0)
            xs_features_ext = xs_features_ext.repeat(n_query * n_class, 1, 1, 1)
            xq_features_ext = xq_features.unsqueeze(0).repeat(n_class, 1, 1, 1)
            xq_features_ext = torch.transpose(xq_features_ext, 0, 1)

            relation_pairs = torch.cat((xs_features_ext, xq_features_ext), 2)
            relation_pairs = relation_pairs.view(n_class * n_class * n_query, 128, -1)
            pred = model2(relation_pairs).view(-1, n_class)
            ###

            pred_ = pred
            _, y_hat = pred.max(-1)

            target0 = torch.zeros(n_query, dtype=torch.int64)
            target1 = torch.ones(n_query, dtype=torch.int64)
            targets = torch.concat([target0, target1], 0)

            loss_fn = nn.MSELoss()
            one_hot_targets = Variable(torch.zeros(pred.size(0), n_class).scatter_(1, targets.view(targets.size(0), 1), 1))
            loss = loss_fn(pred, one_hot_targets)


            total_loss += loss
            avg_loss = total_loss / (idx + 1)
            pred_ = pred_.argmax(dim=1)
            correct += pred_.eq(targets.view_as(pred_)).sum()
            acc = correct / len(targets) * 100
    print(acc/len(test_loader), avg_loss)

    if acc/len(test_loader)>best_acc:
        model1.cpu()
        model2.cpu()

        torch.save(model1, os.path.join(args['log.exp_dir'], "best_FNet.pt"))
        torch.save(model2, os.path.join(args['log.exp_dir'], "best_CNet.pt"))
        best_acc = acc/len(test_loader)
        print("save model, best_acc={}".format(best_acc))

def test(test_loader):

    model1 = torch.load(os.path.join(args['log.exp_dir'], "best_FNet.pt"))
    model1.cpu()
    model1.eval()
    model2 = torch.load(os.path.join(args['log.exp_dir'], "best_CNet.pt"))
    model2.cpu()
    model2.eval()

    torch.manual_seed(1234)

    total_loss = 0.0
    avg_loss = 0.0
    correct = 0.0
    acc = 0.0
    best_acc = 0.0
    meters = {field: tnt.meter.AverageValueMeter() for field in args['log.fields']}

    with tqdm(test_loader) as episodes:
        for idx, data in enumerate(episodes):
            xs = Variable(data['xs'])  # support
            xq = Variable(data['xq'])

            n_class = xs.size(0)
            assert xq.size(0) == n_class
            n_support = xs.size(1)
            n_query = xq.size(1)

            x = torch.cat([xs.view(n_class * n_support, *xs.size()[2:]),
                           xq.view(n_class * n_query, *xq.size()[2:])], 0)
            target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
            target_inds = Variable(target_inds, requires_grad=False)

            ###
            xs_features = model1(xs.view(n_class * n_support, *xs.size()[2:]))  # ()
            xs_features = xs_features.view(n_class, n_support, 64, -1)
            xs_features = torch.sum(xs_features, 1).squeeze(1)
            xq_features = model1(xq.view(n_class * n_query, *xq.size()[2:]))

            xs_features_ext = xs_features.unsqueeze(0)
            xs_features_ext = xs_features_ext.repeat(n_query * n_class, 1, 1, 1)
            xq_features_ext = xq_features.unsqueeze(0).repeat(n_class, 1, 1, 1)
            xq_features_ext = torch.transpose(xq_features_ext, 0, 1)

            relation_pairs = torch.cat((xs_features_ext, xq_features_ext), 2)
            relation_pairs = relation_pairs.view(n_class * n_class * n_query, 128, -1)
            pred = model2(relation_pairs).view(-1, n_class)
            ###

            pred_ = pred
            _, y_hat = pred.max(-1)

            target0 = torch.zeros(n_query, dtype=torch.int64)
            target1 = torch.ones(n_query, dtype=torch.int64)
            targets = torch.concat([target0, target1], 0)

            loss_fn = nn.MSELoss()
            one_hot_targets = Variable(torch.zeros(pred.size(0), n_class).scatter_(1, targets.view(targets.size(0), 1), 1))
            loss = loss_fn(pred, one_hot_targets)

            y_re = targets
            y_real = np.array(y_re.cpu()).reshape(-1)
            y_pred = np.array(y_hat.cpu()).reshape(-1)
            acc = accuracy_score(y_real, y_pred)  # TP+TN/(TP+FN+FP+TN)
            pre = precision_score(y_real, y_pred, average='binary')  # TP/TP+FP
            rec = recall_score(y_real, y_pred, average='binary')  # TP/TP+FN
            F1s = f1_score(y_real, y_pred, average='binary')  # 2*(pre*recall/(pre+recall))

            output = {
            'loss': loss.item(),
            'Accuracy': acc,
            'Precision': pre,
            'Recall': rec,
            'F1': F1s
            }

            for field, meter in meters.items():
                meter.add(output[field])

    for field, meter in meters.items():
        mean, std = meter.value()
        print("test {:s}: {:0.6f} +/- {:0.6f}".format(field, mean,
                                                      1.96 * std / math.sqrt(args_copy['data.test_episodes'])))





if __name__ == '__main__':
    data = data_utils.load(args, ['train', 'val'])
    train_loader = data['train']
    val_loader = data['val']

    args['log.fields'] = args['log.fields'].split(',')

    args_copy = args.copy()
    args_copy['data.test_episodes'] = 100
    test_data = data_utils.load(args_copy, ['test'])
    test_loader = test_data['test']

    epochs = 5

    # model = FC_Net()

    F_model = F_Net()
    C_model = C_Net()

    F_optimizer = torch.optim.SGD(
        F_model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4
    )
    C_optimizer = torch.optim.SGD(
        C_model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4
    )
    # optimizer = torch.optim.SGD(
    #     model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4
    # )

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    F_scheduler = torch.optim.lr_scheduler.StepLR(F_optimizer, step_size=20, gamma=0.1)
    C_scheduler = torch.optim.lr_scheduler.StepLR(C_optimizer, step_size=20, gamma=0.1)
    loss = 0
    acc = 0
    acc_best = 0.0

    for epoch in range(epochs):
        print("epoch:{}".format(epoch))
        train(F_model, C_model, train_loader, 'cpu', F_optimizer, C_optimizer)
        F_scheduler.step()
        C_scheduler.step()
        val(F_model, C_model, val_loader, 'cpu', acc_best)
        print(acc_best)

    test(test_loader)




