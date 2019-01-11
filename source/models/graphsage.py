import torch
import torch.nn as nn
from torch.nn import functional as F

class GraphSAGE(nn.Module):

    def __init__(self, input_dim, output_dim, device="cuda:0", normalize=True):
        super(GraphSAGE, self).__init__()
        self.device = device
        self.normalize = normalize
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(self.input_dim, self. output_dim)
        self.layer_norm = nn.LayerNorm(self.output_dim)  # elementwise_affine=False
        nn.init.xavier_uniform_(self.linear.weight)

    def aggregate_convolutional(self, x, a):
        a += torch.eye(a.shape[0], dtype=torch.float, device=self.device)
        h_hat = a @ x
        
        return h_hat
    
    def forward(self, x, a):
        h_hat = self.aggregate_convolutional(x, a)
        h = F.relu(self.linear(h_hat))
        # h = self.linear(h_hat)
        if self.normalize:
            # h = F.normalize(h, p=2, dim=1)  # Normalize edge embeddings
            h = self.layer_norm(h)  # Normalize layerwise (mean=0, std=1)
            # print()
        
        return h
