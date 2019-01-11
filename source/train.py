import argparse
import os
import os.path
import time

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.nn import functional as F
from torchvision.utils import make_grid
from torchvision.utils import save_image

from models.model import Model
from models.graphsage import GraphSAGE
from utils.utils import DatasetHelper

SEED = 42
NORMALIZE = True

def main():
    print(ARGS)
    start_time = time.time()
    
    device = torch.device(ARGS.device)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if ARGS.device != "cpu":
        torch.cuda.manual_seed(SEED)
    normalize = NORMALIZE
    
    dataset_helper = DatasetHelper()
    dataset_helper.load_dataset(ARGS.dataset, device=device, seed=SEED, normalize=normalize)
    (x_train, a_train, labels_train) = dataset_helper.train
    (x_valid, a_valid, labels_valid) = dataset_helper.valid
    feature_size = dataset_helper.feature_size
    print("Imported dataset, generated train and validation splits, took: {:.3f}s".format(time.time() - start_time))
    
    gcn = GraphSAGE(feature_size, feature_size*2, device=device, normalize=normalize)
    gcn = gcn.to(device=device)
    
    gcn.train()
    
    x1 = x_train[0]
    a1 = a_train[0]
    t1 = labels_train[0]
    weights = gcn.linear.weight
    z1 = gcn(x1, a1)
    print()
    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=10, type=int,
                        help='max number of epochs')
    parser.add_argument('--batch_size', default=16, type=int,
                        help='size of each batch')
    parser.add_argument('--lr_rate', default=1e-3, type=float,
                        help='learning rate')
    parser.add_argument('--device', default="cuda:0", type=str,
                        help='training device')
    parser.add_argument('--dataset', default="../data/mutag.graph", type=str,
                        help='dataset path')
    
    ARGS = parser.parse_args()
    
    main()

