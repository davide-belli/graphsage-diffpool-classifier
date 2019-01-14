import torch
import torch.nn as nn
from torch.nn import functional as F

class GraphSAGE(nn.Module):

    def __init__(self, input_feat, output_feat, device="cuda:0", normalize=True):
        super(GraphSAGE, self).__init__()
        self.device = device
        self.normalize = normalize
        self.input_feat = input_feat
        self.output_feat = output_feat
        self.linear = nn.Linear(self.input_feat, self. output_feat)
        self.layer_norm = nn.LayerNorm(self.output_feat)  # elementwise_affine=False
        nn.init.xavier_uniform_(self.linear.weight)

    def aggregate_convolutional(self, x, a):
        eye = torch.eye(a.shape[0], dtype=torch.float, device=self.device)
        a = a + eye
        h_hat = a @ x
        
        return h_hat
    
    def forward(self, x, a):
        h_hat = self.aggregate_convolutional(x, a)
        h = F.relu(self.linear(h_hat))
        if self.normalize:
            # h = F.normalize(h, p=2, dim=1)  # Normalize edge embeddings
            h = self.layer_norm(h)  # Normalize layerwise (mean=0, std=1)
            
        return h
