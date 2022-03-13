from src.dataset.transforms import *
from src.dataset.base_transform import BaseTransform


def get_transform(transform_type: str) -> BaseTransform:
    """
    Maps transform_type to transform class
    :param transform_type: str
    :return: BaseTransform
    """
    if transform_type == 'identity':
        return Identity()
    elif transform_type == 'degree':
        return Degree()
    elif transform_type == 'degree_bin':
        return DegreeBin()
    elif transform_type == 'LDP':
        return LDPTransform()
    elif transform_type == 'adj':
        return Adj()
    elif transform_type == 'node2vec':
        return Node2Vec()
    elif transform_type == 'eigenvector':
        return Eigenvector()
    elif transform_type == 'eigen_norm':
        return EigenNorm()
    else:
        raise ValueError('Unknown transform type: {}'.format(transform_type))
