from torch import Tensor
import numpy as np
from torch_geometric.data import Data


def calculate_bin_edges(dataset: [Data], num_bins: int = 10) -> Tensor:
    """
    Calculate the bin edges for a given edge attribute tensor.
    :param dataset: The dataset to calculate the bin edges for.
    :param num_bins: The number of bins.
    :return: The bin edges.
    """
    all_edges = np.concatenate([data.edge_attr.numpy() for data in dataset])
    hist, bin_edges = np.histogram(all_edges, bins=num_bins)
    bin_edges = np.sort(bin_edges)
    return Tensor(bin_edges)
