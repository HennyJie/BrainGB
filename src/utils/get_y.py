from torch_geometric.data import Data


def get_y(dataset: [Data]):
    """
    Get the y values from a list of Data objects.
    """
    y = []
    for d in dataset:
        y.append(d.y.item())
    return y
