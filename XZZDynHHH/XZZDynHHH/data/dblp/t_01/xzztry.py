import pickle as pkl
import torch
import networkx as nx

# with open('node_features.pkl', "rb") as f:
#     graphs = pkl.load(f)
# print(graphs.shape)
# print(type(graphs))
a = torch.load('node_features.pth')
print(type(a))
print(a.shape)
print(a[1])