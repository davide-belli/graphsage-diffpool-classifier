import torch
import torch.nn as nn
from torch.nn import functional as F


from models.diffpool import DiffPool
from models.graphsage import GraphSAGE

class Classifier(nn.Module):

    def __init__(self, device="cuda:0"):
        super(Classifier, self).__init__()
        self.device = device
        self.sage1 = GraphSAGE(7, 14, device=self.device)
        self.pool1 = DiffPool(14, 14, device=self.device)
        self.sage2 = GraphSAGE(14, 28, device=self.device)
        self.pool2 = DiffPool(28, 1, final_layer=True, device=self.device)
        self.linear1 = nn.Linear(28, 10)
        self.linear2 = nn.Linear(10, 2)

    def forward(self, x, a):
        x = self.sage1(x, a)
        x, a = self.pool1(x, a)
        x = self.sage2(x, a)
        x, a = self.pool2(x, a)
        y = self.linear1(x.squeeze(0))
        y = self.linear2(y)
        return x, a, y
