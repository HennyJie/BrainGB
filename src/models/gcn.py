import torch
from torch_geometric.nn import global_add_pool, global_mean_pool, MessagePassing
from torch.nn import Parameter
import numpy as np
from torch.nn import functional as F
from torch_geometric.nn.inits import glorot, zeros
from typing import Tuple
from torch import Tensor
from torch_geometric.nn import GCNConv
from torch import nn
import torch_geometric
import math


class MPGCNConv(GCNConv):
    def __init__(self, in_channels, out_channels, edge_emb_dim: int, gcn_mp_type: str, bucket_sz: float, 
                 normalize: bool = True, bias: bool = True):
        super(MPGCNConv, self).__init__(in_channels=in_channels, out_channels=out_channels, aggr='add')

        self.edge_emb_dim = edge_emb_dim
        self.gcn_mp_type = gcn_mp_type
        self.bucket_sz = bucket_sz
        self.bucket_num = math.ceil(2.0/self.bucket_sz)
        if gcn_mp_type == "bin_concate":
            self.edge2vec = nn.Embedding(self.bucket_num, edge_emb_dim)

        self.normalize = normalize
        self._cached_edge_index = None
        self._cached_adj_t = None
        
        input_dim = out_channels
        if gcn_mp_type == "bin_concate" or gcn_mp_type == "edge_weight_concate":
            input_dim = out_channels + edge_emb_dim
        elif gcn_mp_type == "edge_node_concate":
            input_dim = out_channels*2 + 1
        elif gcn_mp_type == "node_concate":
            input_dim = out_channels*2
        self.edge_lin = torch.nn.Linear(input_dim, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def message(self, x_i, x_j, edge_weight):
        # x_j: [E, in_channels]
        if self.gcn_mp_type == "weighted_sum": 
            # use edge_weight as multiplier
            msg = edge_weight.view(-1, 1) * x_j
        elif self.gcn_mp_type == "bin_concate":
            # concat xj and learned bin embedding
            bucket = torch.div(edge_weight + 1, self.bucket_sz, rounding_mode='trunc').int()
            msg = torch.cat([x_j, self.edge2vec(bucket)], dim=1)
            msg = self.edge_lin(msg) 
        elif self.gcn_mp_type == "edge_weight_concate":
            # concat xj and tiled edge attr
            msg = torch.cat([x_j, edge_weight.view(-1, 1).repeat(1, self.edge_emb_dim)], dim=1)
            msg = self.edge_lin(msg) 
        elif self.gcn_mp_type == "edge_node_concate": 
            # concat xi, xj and edge_weight
            msg = torch.cat([x_i, x_j, edge_weight.view(-1, 1)], dim=1)
            msg = self.edge_lin(msg)
        elif self.gcn_mp_type == "node_concate":
            # concat xi and xj
            msg = torch.cat([x_i, x_j], dim=1)
            msg = self.edge_lin(msg)
        else:
            raise ValueError(f'Invalid message passing variant {self.gcn_mp_type}')
        return msg

        
class GCN(torch.nn.Module):
    def __init__(self, input_dim, args, num_nodes, num_classes):
        super(GCN, self).__init__()
        self.activation = torch.nn.ReLU()
        self.convs = torch.nn.ModuleList()
        self.pooling = args.pooling
        self.num_nodes = num_nodes

        hidden_dim = args.hidden_dim
        num_layers = args.n_GNN_layers
        edge_emb_dim = args.edge_emb_dim
        gcn_mp_type = args.gcn_mp_type
        bucket_sz = args.bucket_sz
        gcn_input_dim = input_dim

        for i in range(num_layers-1):
            conv = torch_geometric.nn.Sequential('x, edge_index, edge_attr', [
                (MPGCNConv(gcn_input_dim, hidden_dim, edge_emb_dim, gcn_mp_type, bucket_sz, normalize=True, bias=True),'x, edge_index, edge_attr -> x'),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(negative_slope=0.2),
                nn.BatchNorm1d(hidden_dim)
            ])
            gcn_input_dim = hidden_dim
            self.convs.append(conv)

        input_dim = 0

        if self.pooling == "concat":
            node_dim = 8
            conv = torch_geometric.nn.Sequential('x, edge_index, edge_attr', [
                (MPGCNConv(hidden_dim, hidden_dim, edge_emb_dim, gcn_mp_type, bucket_sz, normalize=True, bias=True),'x, edge_index, edge_attr -> x'),
                nn.Linear(hidden_dim, 64),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(64, node_dim),
                nn.LeakyReLU(negative_slope=0.2),
                nn.BatchNorm1d(node_dim)
            ])
            input_dim = node_dim*num_nodes

        elif self.pooling == 'sum' or self.pooling == 'mean':
            node_dim = 256
            input_dim = node_dim
            conv = torch_geometric.nn.Sequential('x, edge_index, edge_attr', [
                (MPGCNConv(hidden_dim, hidden_dim, edge_emb_dim, gcn_mp_type, bucket_sz, normalize=True, bias=True),'x, edge_index, edge_attr -> x'),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(negative_slope=0.2),
                nn.BatchNorm1d(node_dim)
            ])

        self.convs.append(conv)

        self.fcn = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(256, 32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(32, 2)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        z = x
        edge_attr = torch.abs(edge_attr)
   
        for i, conv in enumerate(self.convs):
            # bz*nodes, hidden
            z = conv(z, edge_index, edge_attr)

        if self.pooling == "concat":
            z = z.reshape((z.shape[0]//self.num_nodes, -1))
        elif self.pooling == 'sum':
            z = global_add_pool(z,  batch)  # [N, F]
        elif self.pooling == 'mean':
            z = global_mean_pool(z, batch)  # [N, F]

        out = self.fcn(z)
        return out

        

