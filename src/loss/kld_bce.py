import torch
import torch.nn as nn


def KLD_BCE(x, y, mu, logvar):

    bce = nn.BCELoss()

    # eps = torch.from_numpy(np.array(np.spacing(1))).clone()
    kld = -(0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))
    reconstruction = bce(y, x)
    # reconstruction
    # torch.sum(x * torch.log(y + eps) + (1 - x) * torch.log(1 - y + eps))
    total = reconstruction + kld

    return kld, reconstruction, total
