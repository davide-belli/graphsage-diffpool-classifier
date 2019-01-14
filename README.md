# Graph Autoencoder PyTorch Implementation
### Generating a graph autoencoder in PyTorch, combining Graph Convolutions and Graph Pooling

[Davide Belli](https://github.com/davide-belli)

 - [GCNs](https://arxiv.org/pdf/1609.02907.pdf)
 - [GraphSAGE](https://arxiv.org/pdf/1706.02216.pdf)
 - [DiffPool](https://arxiv.org/pdf/1806.08804.pdf)
 - [Graph U-net](https://openreview.net/pdf?id=HJePRoAct7)

## Papers Reading
### 20/11/2018

- SEMI-SUPERVISED CLASSIFICATION WITH GRAPH CONVOLUTIONAL NETWORKS (GCNs)
- GraphRNN: Generating Realistic Graphs with Deep Auto-regressive Models (GraphRNN)
- Hierarchical Graph Representation Learning with Differentiable Pooling (DiffPool)
- Deep Generative Models for Graphs: VAEs, GANs, and reinforcement learning for de novo drug discovery (Nicola's Thesis)
- MolGAN: An implicit generative model for small molecular graphs
- Neural Guided Constraint Logic Programming for Program Synthesis
- Combined Reinforcement Learning via Abstract Representations
---

## Initial Experiments on Datasets
### 30/11/2018

- created repository
- studied COLLAB dataset and computed statistics about graphs
---

## Papers Reading
### 02/12/2018

- DEEP GEOMETRICAL GRAPH CLASSIFICATION
- GRAPH U-NET
- Inductive Representation Learning on Large Graphs
---

## Further experiments on Datasets
### 06/12/2018

- extended analysis on datasets: 
   - COLLAB: Without node features,
   - MUTAG: With node features
- utils.py: Added function to import graphs dataset and return tensor representation (A, X, labels)
---

## GraphSAGE and improved dataset loader
### 07/12/2018

- changed dataset loader to Class framework
- implemented GraphSAGE layer in PyTorch
- questions:
- - In GraphSAGE, how do we work in case of one-dimensional features? 
 	e.g. the output will have feature dimensions zero, or shall we increase it? If not, with normalization all the values will be 1, since they are scalars (shall we normalize also the feature in input in this case?). Also, if the weights are initialized as negative all the activations after a relu will be zero after the first forward
- - What about zero-dimensional ones?

## Papers Reading
### 01/01/2019

- Hierarchical Bipartite GCNs
- Adaptive Skip Intervals: Temporal Abstraction for Recurrent Dynamical Models
- Variational Graph Auto-Encoders
- (PARTIALLY) A Comprehensive Survey on Graph Neural Networks
- (PARTIALLY) Graph Neural Networks: A Review of Methods and Applications 
- Compositional Imitation Learning: Explaining and executing one task at a time

## DiffPool
### 11/01/2019

- added LayerNorm layer for GraphSAGE
- added one-hot encoding for scalar features
- implemented DiffPool
- designed and implemented a simple graph classifier with convolution and pooling layers

## Training Code
### 14/01/2019

- fixed bugs in dataset generation and GraphSAGE (inplace operation)
- added code for training
- added code for plotting
- executed training and generated accuracy/loss plots