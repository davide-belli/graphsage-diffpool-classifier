import os
import os.path
import errno
import pickle

import numpy as np
import torch


def read_file(ds_name):
    KNOWN_DATASETS = {"../data/mutag.graph"}
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


def load_dataset(ds_name, device="cuda:0", seed=42):
    train, valid = None, None
    
    graphs, labels = read_file(ds_name)
    
    # Compute shape dimensions N_graphs, N_nodes, D (features size)
    N_graphs = len(graphs)
    N_nodes = -1  # Max number of nodes among all of the graphs
    D = len(graphs[0][0]['label'])  # Feature array size for each node
    for gidxs, graph in graphs.items():
        N_nodes = max(N_nodes, len(graph))
            
    assert N_nodes > 0, "Apparently,there are no nodes in these graphs"
    
    # Generate train anf valid splits
    torch.manual_seed(seed)
    shuffled_idx = torch.randperm(N_graphs)
    N_train = int(N_graphs * 0.8)
    N_valid = N_graphs - N_train
    train_idx = shuffled_idx[:N_train]
    valid_idx = shuffled_idx[N_train:]

    # Generate PyTorch tensors
    A_train = torch.zeros((N_train, N_nodes, N_nodes), dtype=torch.int32, device=device)
    X_train = torch.zeros((N_train, N_nodes, D), dtype=torch.float64, device=device)
    labels_train = torch.FloatTensor((N_train), device=device)
    for i in range(N_valid):
        idx = train_idx[i].item()
        labels_train[i] = labels[idx]
        for j in range(len(graphs[idx])):
            for k in graphs[idx][j]['neighbors']:
                A_train[i, j, k] = 1
            for k, d in enumerate(graphs[idx][j]['label']):
                X_train[i, j, k] = float(d)

    A_valid = torch.zeros((N_valid, N_nodes, N_nodes), dtype=torch.int32, device=device)
    X_valid = torch.zeros((N_valid, N_nodes, D), dtype=torch.float64, device=device)
    labels_valid = torch.FloatTensor((N_valid), device=device)
    for i in range(N_valid):
        idx = valid_idx[i].item()
        labels_valid[i] = labels[idx]
        for j in range(len(graphs[idx])):
            for k in graphs[idx][j]['neighbors']:
                A_valid[i, j, k] = 1
            for k, d in enumerate(graphs[idx][j]['label']):
                X_valid[i, j, k] = float(d)

    train = (X_train, A_train, labels_train)
    valid = (X_valid, A_valid, labels_valid)
    
    return train, valid