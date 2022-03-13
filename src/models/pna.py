import os.path as osp
import numpy
import torch
import torch.nn.functional as F
from torch.nn import ModuleList, Embedding
from torch.nn import Sequential, ReLU, Linear
from torch_geometric.data import DataLoader
from torch_geometric.nn import PNAConv, BatchNorm, global_add_pool


class PNA(torch.nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, deg, nodes_num, edge_emb_dim):
        super(PNA, self).__init__()
        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        if isinstance(deg, numpy.ndarray):
            deg = torch.LongTensor(deg)
        for i in range(1):
            if i == 0:
                conv = PNAConv(in_channels=node_feature_dim, out_channels=hidden_dim,
                               aggregators=aggregators, scalers=scalers, deg=deg,
                               edge_dim=edge_emb_dim, towers=hidden_dim // 4, pre_layers=1, post_layers=1,
                               divide_input=False)  # out_channels % towers == 0
            else:
                conv = PNAConv(in_channels=hidden_dim, out_channels=hidden_dim,
                               aggregators=aggregators, scalers=scalers, deg=deg,
                               edge_dim=edge_emb_dim, towers=hidden_dim // 4, pre_layers=1, post_layers=1,
                               divide_input=False)  # out_channels % towers == 0
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(hidden_dim))

        self.mlp = Sequential(Linear(hidden_dim, hidden_dim // 2), ReLU(), Linear(hidden_dim // 2, hidden_dim // 4),
                              ReLU(),
                              Linear(hidden_dim // 4, 2))

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        edge_attr = edge_attr.unsqueeze(1)
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index, edge_attr)
            x = batch_norm(x)
            x = F.relu(x)

        x = global_add_pool(x, batch)
        x = self.mlp(x)
        return F.log_softmax(x, dim=-1)
