import argparse
import os
import os.path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.nn import functional as F
from torchvision.utils import make_grid
from torchvision.utils import save_image

import os

from models.model import Model
from utils.utils import load_dataset

def main():
    SEED = 42
    print(ARGS)

    device = torch.device(ARGS.device)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if ARGS.device != "cpu":
        torch.cuda.manual_seed(SEED)
    
    train, valid = load_dataset(ARGS.dataset, device=device, seed=SEED)
    print("Imported dataset, generated train and validation splits")
    
    # Add timing


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')
    parser.add_argument('--zdim', default=2, type=int,
                        help='dimensionality of latent space')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='size of each batch')
    parser.add_argument('--lr_rate', default=1e-3, type=float,
                        help='learning rate')
    parser.add_argument('--device', default="cuda:0", type=str,
                        help='training device')
    parser.add_argument('--dataset', default="../data/mutag.graph", type=str,
                        help='dataset path')
    
    ARGS = parser.parse_args()
    
    
    main()