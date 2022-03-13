import torch
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import networkx as nx
from .maskable_list import MaskableList
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data


def build_dataset(a1, args, y):
    x1 = compute_x(a1, args)
    data_list = MaskableList([])
    for i in range(a1.shape[0]):
        edge_index, edge_attr = dense_to_sparse(a1[i])
        data = Data(x=x1[i], edge_index=edge_index, edge_attr=edge_attr, y=y[i], adj=a1[i])
        data_list.append(data)
    return data_list


def compute_x(a1: np.ndarray, node_feature: str):
    # construct node features X
    if node_feature == 'identity':
        x = torch.cat([torch.diag(torch.ones(a1.shape[1]))] * a1.shape[0]).reshape([a1.shape[0], a1.shape[1], -1])
        x1 = x.clone()

    # elif args.node_features == 'node2vec':
    #     X = np.load(f'{path}/{args.dataset_name}_{args.modality}.emb', allow_pickle=True).astype(np.float32)
    #     x1 = torch.from_numpy(X)

    elif node_feature == 'degree':
        a1b = (a1 != 0).float()
        x1 = a1b.sum(dim=2, keepdim=True)

    elif node_feature == 'degree_bin':
        a1b = (a1 != 0).float()
        x1 = binning(a1b.sum(dim=2))

    elif node_feature == 'adj':
        x1 = a1.float()

    elif node_feature == 'LDP':
        a1b = (a1 != 0).float()
        x1 = []
        n_graphs: int = a1.shape[0]
        for i in range(n_graphs):
            x1.append(LDP(nx.from_numpy_array(a1b[i].numpy())))
    else:
        raise ValueError(f'Unknown node feature {node_feature}')
    x1 = torch.Tensor(x1).float()
    return x1


# for LDP node features
def LDP(g, key='deg'):
    x = np.zeros([len(g.nodes()), 5])

    deg_dict = dict(nx.degree(g))
    for n in g.nodes():
        g.nodes[n][key] = deg_dict[n]

    for i in g.nodes():
        nodes = g[i].keys()

        nbrs_deg = [g.nodes[j][key] for j in nodes]

        if len(nbrs_deg) != 0:
            x[i] = [
                np.mean(nbrs_deg),
                np.min(nbrs_deg),
                np.max(nbrs_deg),
                np.std(nbrs_deg),
                np.sum(nbrs_deg)
            ]

    return x


# for degree_bin node features
def binning(a, n_bins=10):
    n_graphs = a.shape[0]
    n_nodes = a.shape[1]
    _, bins = np.histogram(a, n_bins)
    binned = np.digitize(a, bins)
    binned = binned.reshape(-1, 1)
    enc = OneHotEncoder()
    return enc.fit_transform(binned).toarray().reshape(n_graphs, n_nodes, -1).astype(np.float32)