import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import TransformerConv
import h5py
import scipy.io as sio
import pickle
class SurrogateGNN(nn.Module):
    def __init__(self, in_dim, edge_dim, hidden_dim=64, n_layers=3, dropout=0.1, out_node_dim=3):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.res_projs = nn.ModuleList()
        for i in range(n_layers):
            ind = in_dim if i == 0 else hidden_dim
            conv = TransformerConv(ind, hidden_dim, edge_dim=edge_dim, heads=1, dropout=dropout)
            self.layers.append(conv)
            self.norms.append(nn.LayerNorm(hidden_dim))
            self.res_projs.append(nn.Linear(ind, hidden_dim) if ind != hidden_dim else nn.Identity())
        self.node_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_node_dim))

    def forward(self, x, edge_index, edge_attr):
        h = x
        for conv, norm, proj in zip(self.layers, self.norms, self.res_projs):
            h_in = h
            h = conv(h, edge_index, edge_attr)
            h = F.relu(h)
            h = norm(h)
            h = h + proj(h_in)
        return self.node_head(h)