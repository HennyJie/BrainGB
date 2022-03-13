import torch
from node2vec import Node2Vec as Node2Vec_
from .brain_data import BrainData
from torch_geometric.data import Data
from networkx.convert_matrix import from_numpy_matrix
from .utils import binning, LDP
import networkx as nx
from .base_transform import BaseTransform
from numpy import linalg as LA
import numpy as np


class FromSVTransform(BaseTransform):
    def __init__(self, sv_transform):
        super(FromSVTransform, self).__init__()
        self.sv_transform = sv_transform

    def __call__(self, data):
        keys = list(filter(lambda x: x.startswith('edge_index'), data.keys))
        for key in keys:
            if key.startswith('edge_index'):
                postfix = key[10:]
                edge_index = data[f'edge_index{postfix}']
                edge_attr = data[f'edge_attr{postfix}']
                svdata = Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=data.num_nodes)
                svdata_transformed = self.sv_transform(svdata)
                data[f'x{postfix}'] = svdata_transformed.x
                data[f'edge_index{postfix}'] = svdata_transformed.edge_index
                data[f'edge_attr{postfix}'] = svdata_transformed.edge_attr
        return data

    def __str__(self):
        return self.sv_transform.__class__.__name__


class Identity(BaseTransform):
    def __call__(self, data: BrainData):
        """
        Returns a diagonal matrix with ones on the diagonal.
        :param data: BrainData
        :return: torch.Tensor
        """
        data.x = torch.diag(torch.ones(data.num_nodes))
        return data


class Degree(BaseTransform):
    def __call__(self, data: BrainData):
        """
        Returns a diagonal matrix with the degree of each node on the diagonal.
        :param data: BrainData
        :return: torch.Tensor
        """
        adj = torch.sparse_coo_tensor(data.edge_index, data.edge_attr, [data.num_nodes, data.num_nodes])
        adj = adj.to_dense()
        data.x = torch.Tensor(adj.sum(dim=1, keepdim=True)).float()
        return data

    def __str__(self):
        return 'Degree'


class LDPTransform(BaseTransform):
    def __call__(self, data: BrainData):
        """
        Returns node feature with LDP transform.
        :param data: BrainData
        :return: torch.Tensor
        """
        adj = torch.sparse_coo_tensor(data.edge_index, data.edge_attr, [data.num_nodes, data.num_nodes])
        adj = adj.to_dense()
        data.x = torch.Tensor(
                    LDP(nx.from_numpy_array(adj.numpy()))
                ).float()
        return data

    def __str__(self):
        return 'LDP'


class DegreeBin(BaseTransform):
    def __call__(self, data: BrainData):
        """
        Returns node feature with degree bin transform.
        :param data: BrainData
        :return: torch.Tensor
        """
        adj = torch.sparse_coo_tensor(data.edge_index, data.edge_attr, [data.num_nodes, data.num_nodes])
        adj = adj.to_dense()
        return torch.Tensor(binning(adj.sum(dim=1))).float()

    def __str__(self):
        return 'Degree_Bin'


class Adj(BaseTransform):
    def __call__(self, data: BrainData):
        """
        Returns adjacency matrix.
        :param data: BrainData
        :return: torch.Tensor
        """
        adj = torch.sparse_coo_tensor(data.edge_index, data.edge_attr, [data.num_nodes, data.num_nodes])
        adj = adj.to_dense()
        data.x = adj
        return data

    def __str__(self):
        return 'Adj'


class Eigenvector(BaseTransform):
    def __call__(self, data: BrainData):
        """
        Returns node feature with eigenvector.
        :param data: BrainData
        :return: torch.Tensor
        """
        adj = torch.sparse_coo_tensor(data.edge_index, data.edge_attr, [data.num_nodes, data.num_nodes])
        adj = adj.to_dense()
        w, v = LA.eig(adj.numpy())
        # indices = np.argsort(w)[::-1]
        v = v.transpose()
        data.x = torch.Tensor(v).float()
        return data


class EigenNorm(BaseTransform):
    def __call__(self, data: BrainData):
        """
        Returns node feature with eigen norm.
        :param data: BrainData
        :return: torch.Tensor
        """
        adj = torch.sparse_coo_tensor(data.edge_index, data.edge_attr, [data.num_nodes, data.num_nodes])
        adj = adj.to_dense()
        sum_of_rows = adj.sum(dim=1)
        adj /= sum_of_rows
        adj = torch.nan_to_num(adj)
        w, v = LA.eig(adj.numpy())
        # indices = np.argsort(w)[::-1]
        v = v.transpose()
        data.x = torch.Tensor(v).float()
        return data


class Node2Vec(BaseTransform):
    def __init__(self, feature_dim=32, walk_length=5, num_walks=200, num_workers=4,
                 window=10, min_count=1, batch_words=4):
        super(Node2Vec, self).__init__()
        self.feature_dim = feature_dim
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.num_workers = num_workers
        self.window = window
        self.min_count = min_count
        self.batch_words = batch_words

    def __call__(self, data):
        """
        Returns node feature with node2vec transform.
        :param data: BrainData
        :return: torch.Tensor
        """
        adj = torch.sparse_coo_tensor(data.edge_index, data.edge_attr, [data.num_nodes, data.num_nodes])
        adj = adj.to_dense()
        if (adj < 0).int().sum() > 0:
            # split the adjacency matrix into two (negative and positive) parts
            pos_adj = adj.clone()
            pos_adj[adj < 0] = 0
            neg_adj = adj.clone()
            neg_adj[adj > 0] = 0
            neg_adj = -neg_adj
            adjs = [pos_adj, neg_adj]
        else:
            adjs = [adj]

        xs = []
        for adj in adjs:
            x = torch.zeros((data.num_nodes, self.feature_dim))
            graph = from_numpy_matrix(adj.numpy())
            node2vec = Node2Vec_(graph, dimensions=self.feature_dim, walk_length=self.walk_length,
                                 num_walks=self.num_walks, workers=self.num_workers)
            model = node2vec.fit(window=self.window, min_count=self.min_count,
                                 batch_words=self.batch_words)
            for i in range(data.num_nodes):
                x[i] = torch.Tensor(model.wv[f'{i}'].copy())
            xs.append(x)
        data.x = torch.cat(xs, dim=-1)
        return data

    def __str__(self):
        return 'Node2Vec'
