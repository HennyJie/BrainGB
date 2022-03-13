from torch_geometric.data import Data


class BrainData(Data):
    def __init__(self, num_views=None, num_nodes=None, y=None, *args, **kwargs):
        super(BrainData, self).__init__()
        self.num_views = num_views
        self.num_nodes = num_nodes
        self.y = y
        for k, v in kwargs.items():
            if k.startswith('x') or k.startswith('edge_index') or k.startswith('edge_attr'):
                self.__dict__[k] = v

    def __inc__(self, key, value):
        if key.startswith('edge_index'):
            return self.num_nodes
        else:
            return super().__inc__(key, value)
