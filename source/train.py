import argparse
import os
import os.path
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.nn import functional as F
from torchvision.utils import make_grid
from torchvision.utils import save_image

from models.classifier import Classifier
from models.graphsage import GraphSAGE
from models.diffpool import DiffPool
from utils.utils import *

SEED = 42
NORMALIZE = True

def compute_accuracy(pred, target, device="cuda:0"):
    pred_labels = torch.stack(pred, dim=0).to(device=device)
    acc = (pred_labels.long() == target.long()).float().mean()
    
    return acc

def run_epoch(classifier, optimizer, criterion, x_data, a_data, t_data, eval=False, device="cuda:0"):
    data_len = x_data.size(0)
    pred = []
    losses = []
    scores = []
    if eval:
        classifier.eval()
    else:
        classifier.train()
        
    for i in range(data_len):
        optimizer.zero_grad()
        x = x_data[i]
        a = a_data[i]
        t = t_data[i]
        
        _, _, y = classifier(x, a)
        loss = criterion(y.unsqueeze(0), t.long().unsqueeze(0))
        if not eval:
            loss.backward()
            optimizer.step()
        pred.append(y.argmax())
        losses.append(loss)
        scores.append(y)
    acc_epoch = compute_accuracy(pred, t_data, device=device).item()
    loss_epoch = torch.FloatTensor(losses).mean().item()
    
    return acc_epoch, loss_epoch
    

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
    # feature_size = dataset_helper.feature_size
    print("Imported dataset, generated train and validation splits, took: {:.3f}s".format(time.time() - start_time))

    # x1 = x_train[0]
    # a1 = a_train[0]
    # t1 = labels_train[0]
    # # t1_onehot = to_onehot_labels(t1)
    # classifier = Classifier(device=device).to(device=device)
    # _ = classifier(x1, a1)
    
    # # Try GraphSAGE
    # gcn = GraphSAGE(feature_size, feature_size*2, device=device, normalize=normalize)to(device=device)
    # gcn.train()
    # weights = gcn.linear.weight
    # z1 = gcn(x1, a1)
    # print()
    #
    # # Try DiffPool
    # diffpool = DiffPool(feature_size, x1.size(0)//2, device=device)to(device=device)
    # diffpool.train()
    # x1_new, a1_new = diffpool(x1, a1)
    # print()
    # input("remove test")
    
    # Try Classifier
    
    classifier = Classifier(device=device).to(device=device)
    optimizer = optim.Adam(classifier.parameters())
    criterion = nn.CrossEntropyLoss()
    
    assert x_train.size(0) == dataset_helper.train_size == labels_train.size(0)
    assert x_valid.size(0) == dataset_helper.valid_size == labels_valid.size(0)
    assert (labels_train.sum() + labels_valid.sum()) < 1000000
    
    print("\nStarted training")
    acc_valid, loss_valid = run_epoch(classifier, optimizer, criterion, x_valid, a_valid, labels_valid, eval=True,
                                      device=device)
    print("Epoch {:.0f} | Acc (Valid): {:.3f} | Loss (Valid): {:.3f} |".format(0, acc_valid, loss_valid))
    
    measures = {
        "acc" : {
            "train" : [],
            "valid" : []
        },
        "loss" : {
            "train" : [],
            "valid" : []
        }
    }
    
    for e in range(ARGS.epochs):
        
        acc_train, loss_train = run_epoch(classifier, optimizer, criterion, x_train, a_train, labels_train, eval=False, device=device)
        acc_valid, loss_valid = run_epoch(classifier, optimizer, criterion, x_valid, a_valid, labels_valid, eval=True, device=device)
        print("Epoch {:.0f} | Acc (Train/Valid): {:.3f}/{:.3f} | Loss (Train/Valid): {:.3f}/{:.3f} |"
              .format(e + 1, acc_train, acc_valid, loss_train, loss_valid))
        
        measures["acc"]["train"].append(acc_train)
        measures["acc"]["valid"].append(acc_valid)
        measures["loss"]["train"].append(loss_train)
        measures["loss"]["valid"].append(loss_valid)
        
        if e % 10 == 0:
            for k in measures.keys():
                generate_plot(measures[k]["train"], measures[k]["valid"], title=k)
        
    print()
    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=100, type=int,
                        help='max number of epochs')
    parser.add_argument('--batch_size', default=16, type=int,
                        help='size of each batch')
    parser.add_argument('--lr_rate', default=1e-4, type=float,
                        help='learning rate')
    parser.add_argument('--device', default="cuda:0", type=str,
                        help='training device')
    parser.add_argument('--dataset', default="../data/mutag.graph", type=str,
                        help='dataset path')
    
    ARGS = parser.parse_args()
    
    main()

