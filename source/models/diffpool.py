import torch
import torch.nn as nn
from torch.nn import functional as F

from models.graphsage import GraphSAGE

class DiffPool(nn.Module):

    def __init__(self, feature_size, output_dim, device="cuda:0", final_layer=False):
        super(DiffPool, self).__init__()
        self.device = device
        self.feature_size = feature_size
        self.output_dim = output_dim
        self.embed = GraphSAGE(self.feature_size, self.feature_size, device=self.device)
        self.pool = GraphSAGE(self.feature_size, self.output_dim, device=self.device)
        self.final_layer = final_layer

    def forward(self, x, a):
        z = self.embed(x, a)
        if self.final_layer:
            s = torch.ones(x.size(0), self.output_dim, device=self.device)
        else:
            s = F.softmax(self.pool(x, a), dim=1)
        x_new = s.t() @ z
        a_new = s.t() @ a @ s
        return x_new, a_new

