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
    contrast_type: str 'classic', 'cltt', else
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
    elif contrast_type == 'supervised':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=crop_size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        train_transform = TwoContrastTransform(train_transform)
    elif contrast_type == 'combined_jitterpluscrop':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=crop_size, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
            ], p=0.8),
            transforms.ToTensor(),
            normalize,
        ])
    elif contrast_type == 'combined_jitter':
        train_transform = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
            ], p=0.8),
            transforms.ToTensor(),
            normalize,
        ])
    elif contrast_type == 'combined_grayscale':
        train_transform = transforms.Compose([
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
            ])
    elif contrast_type == 'combined_jitterplusgrayscale':
        train_transform = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
            ])
    
    
    # if augmentation_type == combined, just return
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



class GaussianBlur(object):
    """blur a single image on CPU"""
    
    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias
    
        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )
    
        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()
    
    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)
    
        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)
    
        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))
    
        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()
    
        img = self.tensor_to_pil(img)
    
        return img


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
