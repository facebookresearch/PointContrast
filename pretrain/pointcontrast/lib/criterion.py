import torch
from torch import nn

class NCESoftmaxLoss(nn.Module):
    def __init__(self):
        super(NCESoftmaxLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, label):
        bsz = x.shape[0]
        x = x.squeeze()
        loss = self.criterion(x, label)
        return loss
