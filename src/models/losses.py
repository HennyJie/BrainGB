import torch

import numpy as np
import torch.nn.functional as F


def _similarity(z1: torch.Tensor, z2: torch.Tensor):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return z1 @ z2.t()


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor,
                 batch_size: int, temperature: float):
    # Space complexity: O(BN) (semi_loss: O(N^2))
    device = z1.device
    num_nodes = z1.size(0)
    num_batches = (num_nodes - 1) // batch_size + 1
    f = lambda x: torch.exp(x / temperature)
    indices = torch.arange(0, num_nodes).to(device)
    losses = []

    for i in range(num_batches):
        batch_mask = indices[i * batch_size: (i + 1) * batch_size]
        intra_similarity = f(_similarity(z1[batch_mask], z1))  # [B, N]
        inter_similarity = f(_similarity(z1[batch_mask], z2))  # [B, N]

        positives = inter_similarity[:, batch_mask].diag()
        negatives = intra_similarity.sum(dim=1) + inter_similarity.sum(dim=1)\
                    - intra_similarity[:, batch_mask].diag()

        losses.append(-torch.log(positives / negatives))

    return torch.cat(losses)


def debiased_nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor,
                          tau: float, tau_plus: float):
    f = lambda x: torch.exp(x / tau)
    intra_similarity = f(_similarity(z1, z1))
    inter_similarity = f(_similarity(z1, z2))

    pos = inter_similarity.diag()
    neg = intra_similarity.sum(dim=1) - intra_similarity.diag() \
          + inter_similarity.sum(dim=1) - inter_similarity.diag()

    num_neg = z1.size()[0] * 2 - 2
    ng = (-num_neg * tau_plus * pos + neg) / (1 - tau_plus)
    ng = torch.clamp(ng, min=num_neg * np.e ** (-1. / tau))

    return -torch.log(pos / (pos + ng))


def hardness_nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor,
                          tau: float, tau_plus: float, beta: float):
    f = lambda x: torch.exp(x / tau)
    intra_similarity = f(_similarity(z1, z1))
    inter_similarity = f(_similarity(z1, z2))

    pos = inter_similarity.diag()
    neg = intra_similarity.sum(dim=1) - intra_similarity.diag() \
          + inter_similarity.sum(dim=1) - inter_similarity.diag()

    num_neg = z1.size()[0] * 2 - 2
    imp = (beta * neg.log()).exp()
    reweight_neg = (imp * neg) / neg.mean()
    ng = (-num_neg * tau_plus * pos + reweight_neg) / (1 - tau_plus)
    ng = torch.clamp(ng, min=num_neg * np.e ** (-1. / tau))

    return -torch.log(pos / (pos + ng))


def jsd_loss(z1, z2, discriminator, pos_mask, neg_mask=None):
    if neg_mask is None:
        neg_mask = 1 - pos_mask
    num_neg = neg_mask.int().sum()
    num_pos = pos_mask.int().sum()
    similarity = discriminator(z1, z2)

    E_pos = (np.log(2) - F.softplus(- similarity * pos_mask)).sum()
    E_pos /= num_pos
    neg_similarity = similarity * neg_mask
    E_neg = (F.softplus(- neg_similarity) + neg_similarity - np.log(2)).sum()
    E_neg /= num_neg

    return E_neg - E_pos