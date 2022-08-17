#!/usr/bin/python
# _____________________________________________________________________________

# ----------------
# import libraries
# ----------------

# standard libraries
# -----
import sys
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


# custom functions
# -----
def get_transformations(contrast_type, rgb_mean, rgb_std, crop_size):
    """
    Get train and val_transform given the contrast_type and parameters specified.
    Returns two instances of type torchvision.transform
    contrast_type: str 'classic', 'time', 'nocontrast'
    rgb_mean: tuple of float (r, g, b)
    rgb_std: tuple of float (r, g, b)
    crop_size: int, pixels
    """

    # setup for case contrast_type == 'combined'
    normalize = transforms.Normalize(mean=rgb_mean, std=rgb_std)

    s = 1.0
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=crop_size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    if contrast_type == 'classic':
        # if classic use the TwoContrastTransform to create contrasts
        train_transform = TwoContrastTransform(train_transform)
    elif contrast_type == 'time':
        # if time, replace the created train_transform with val_transform
        train_transform = val_transform
        # or just return ToTensor() (what we do so far, but normalization might be good)
        # train_transform, val_transform = transforms.ToTensor(), transforms.ToTensor()
    elif contrast_type == 'nocontrast':
        train_transform = TwoContrastTransform(val_transform)

    # if augmentation_type == 'combined', just return
    return train_transform, val_transform

# custom classes
# -----


class TwoContrastTransform:
    """
    Create two contrasts of the same image using the given
    transform: torchvision.transform
    """

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


# ----------------
# main program
# ----------------
if __name__ == "__main__":
    pass


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