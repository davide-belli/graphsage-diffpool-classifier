import os
import os.path
import errno
import pickle

import matplotlib.pyplot as plt

import numpy as np
import torch
from torch.nn import functional as F

KNOWN_DATASETS = {"../data/mutag.graph"}


class DatasetHelper(object):
    
    def __init__(self):
        self.train = None
        self.valid = None
        self.feature_size = -1
        self.n_graphs = -1
        self.n_nodes = -1
        self.train_size = -1
        self.valid_size = -1
        self.onehot = None
        self.labels_set = None
        self.TRAIN_RATIO = 0.8


    def read_file(self, ds_name):
        
        if not os.path.isfile(ds_name):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), ds_name)
        if ds_name not in KNOWN_DATASETS:
            raise NotImplementedError("Dataset unknown to 'load_dataset()' in 'utils/utils.py'")
        
        f = open(ds_name, "rb")
        print("Found dataset:", ds_name)
        data = pickle.load(f, encoding="latin1")
        graphs = data["graph"]
        labels = np.array(data["labels"], dtype=np.float)
        
        return graphs, labels
    
    def normalize_label(self, l):
        # from -1,1 to 0,1
        n = (l + 1) // 2
        if n != 0 and n != 1:
            raise ValueError
        return n
    

    def load_dataset(self, ds_name, device="cuda:0", seed=42, normalize=True, onehot=True):
        
        graphs, labels = self.read_file(ds_name)
        self.labels_set = set(labels)
        
        # Compute shape dimensions n_graphs, n_nodes, features_size
        self.n_graphs = len(graphs)
        self.n_nodes = -1  # Max number of nodes among all of the graphs
        self.onehot = onehot
        
        # Find the feature size (scalar or onehot)
        if self.onehot:
            # Find the size of the onehot vector for the features (i.e.: the maximum value present in the dataset)
            self.feature_size = 0  # Feature array size for each node is onehot vector
            # min_value = 1000
            for i in range(self.n_graphs):
                for j in range(len(graphs[i])):
                    for _, d in enumerate(graphs[i][j]['label']):
                        self.feature_size = max(self.feature_size, d)
                        # min_value = min(min_value, d)
        else:
            self.feature_size = len(graphs[0][0]['label'])  # Feature array size for each node
            
        # Find number of nodes
        for gidxs, graph in graphs.items():
            self.n_nodes = max(self.n_nodes, len(graph))
        assert self.n_nodes > 0, "Apparently,there are no nodes in these graphs"
        
        # Generate train and valid splits
        torch.manual_seed(seed)
        shuffled_idx = torch.randperm(self.n_graphs)
        self.train_size = int(self.n_graphs * self.TRAIN_RATIO)
        self.valid_size = self.n_graphs - self.train_size
        train_idx = shuffled_idx[:self.train_size]
        valid_idx = shuffled_idx[self.train_size:]
    
        # Generate PyTorch tensors for train
        a_train = torch.zeros((self.train_size, self.n_nodes, self.n_nodes), dtype=torch.float, device=device)
        x_train = torch.zeros((self.train_size, self.n_nodes, self.feature_size), dtype=torch.float, device=device)
        labels_train = torch.LongTensor(self.train_size).to(device=device)
        for i in range(self.train_size):
            idx = train_idx[i].item()
            labels_train[i] = self.normalize_label(labels[idx])
            for j in range(len(graphs[idx])):
                for k in graphs[idx][j]['neighbors']:
                    a_train[i, j, k] = 1
                for k, d in enumerate(graphs[idx][j]['label']):
                    if self.onehot:
                        x_train[i, j, :] = to_onehot(d, self.feature_size, device)
                    else:
                        x_train[i, j, k] = float(d)
    
        # Generate PyTorch tensors for valid
        a_valid = torch.zeros((self.valid_size, self.n_nodes, self.n_nodes), dtype=torch.float, device=device)
        x_valid = torch.zeros((self.valid_size, self.n_nodes, self.feature_size), dtype=torch.float, device=device)
        labels_valid = torch.LongTensor(self.valid_size).to(device=device)
        for i in range(self.valid_size):
            idx = valid_idx[i].item()
            labels_valid[i] = self.normalize_label(labels[idx])
            for j in range(len(graphs[idx])):
                for k in graphs[idx][j]['neighbors']:
                    a_valid[i, j, k] = 1
                for k, d in enumerate(graphs[idx][j]['label']):
                    if self.onehot:
                        x_valid[i, j, :] = to_onehot(d, self.feature_size, device)
                    else:
                        x_valid[i, j, k] = float(d)
                    
        if normalize:
            x_train = F.normalize(x_train, p=2, dim=1)
            x_valid = F.normalize(x_valid, p=2, dim=1)
        
        self.train = (x_train, a_train, labels_train)
        self.valid = (x_valid, a_valid, labels_valid)


def to_onehot(x, size, device="cuda:0"):
    t = torch.zeros(size, device=device)
    t[x - 1] = 1 # Since the minimum value is 1, we index -1 the features because we don't need element 0
    return t

def to_onehot_labels(x, device="cuda:0"):
    t = torch.zeros(2, device=device)
    if x.item() == 1:
        t[1] = 1
    else:
        t[0] = 1
    return t

def generate_plot(train, valid, title="Insert Title here"):
    plt.figure()
    plt.plot(valid, label="Validation")
    plt.plot(train, label="Training")
    plt.xlabel("Epoch")
    # plt.ylabel()
    plt.legend()
    plt.title(title)
    # plt.show()
    plt.savefig("./"+title+".svg", format='svg', dpi=1000)
    return
