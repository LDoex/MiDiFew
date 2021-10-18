import torch
import torch.nn as nn
import torch.nn.functional as F

def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

class CosineMarginLoss(nn.Module):
    def __init__(self, embed_dim, num_classes, isCUDA, m=0.35, s=64):
        super(CosineMarginLoss, self).__init__()
        self.w = nn.Parameter(torch.randn(embed_dim, num_classes))
        if isCUDA:
            self.w.cuda()
        self.num_classes = num_classes
        self.m = m
        self.s = s

    def loss(self, x, labels):

        x_norm = x / torch.norm(x, dim=1, keepdim=True)
        w_norm = self.w / torch.norm(self.w, dim=0, keepdim=True)
        xw_norm = torch.matmul(x_norm, w_norm)

        label_one_hot = F.one_hot(labels.view(-1), self.num_classes).float() * self.m
        value = self.s * (xw_norm - label_one_hot)
        return F.cross_entropy(input=value, target=labels.view(-1))