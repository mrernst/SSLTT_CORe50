#!/usr/bin/python
# _____________________________________________________________________________

# ----------------
# import libraries
# ----------------

# standard libraries
# -----
import math

# configuration module
# -----
import config

# utilities
# -----
import torch.nn.functional as F
import torch.nn as nn
import torch


# custom classes
# -----

class SimCLR_TT_Loss(nn.Module):
    def __init__(self, sim_func, batch_size, temperature):
        """Initialize the SimCLR_TT_Loss class"""
        super(SimCLR_TT_Loss, self).__init__()

        self.batch_size = batch_size
        self.temperature = temperature

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.sim_func = sim_func

    def mask_correlated_samples(self, batch_size):
        """
        mask_correlated_samples takes the int batch_size
        and returns an np.array of size [2*batchsize, 2*batchsize]
        which masks the entries that are the same image or
        the corresponding positive contrast
        """
        mask = torch.ones(2 * batch_size, 2 * batch_size, dtype=torch.bool)
        mask = mask.fill_diagonal_(0)

        # fill off-diagonals corresponding to positive samples
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, x, x_pair, labels=None):
        """
        Given a positive pair, we treat the other 2(N − 1)
        augmented examples within a minibatch as negative examples.
        to control for negative samples we just cut off losses
        """
        N = 2 * self.batch_size

        z = torch.cat((x, x_pair), dim=0)

        sim = self.sim_func(z, z) / self.temperature

        # get the entries corresponding to the positive pairs
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)

        # we take all of the negative samples
        negative_samples = sim[self.mask].reshape(N, -1)

        if config.N_negative:
            # if we specify N negative samples: do random permutation of negative sample losses,
            # such that we do consider different positions in the batch
            negative_samples = torch.take_along_dim(
                negative_samples, torch.rand(*negative_samples.shape, device=config.DEVICE).argsort(dim=1), dim=1)
            # cut off array to only consider N_negative samples per positive pair
            negative_samples = negative_samples[:, :config.N_negative]
            # so what we are doing here is basically using the batch to sample N negative
            # samples.

        # the following is more or less a trick to reuse the cross-entropy function for the loss
        # Think of the loss as a multi-class problem and the label is 0
        # such that only the positive similarities are picked for the numerator
        # and everything else is picked for the denominator in cross-entropy
        labels = torch.zeros(N).to(config.DEVICE).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss

# _____________________________________________________________________________

# Stick to 80 characters per line
# Use PEP8 Style
# Comment your code

# -----------------
# top-level comment
# -----------------

# medium level comment
# -----

# low level comment
