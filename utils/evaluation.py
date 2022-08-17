#!/usr/bin/python
# _____________________________________________________________________________

# ----------------
# import libraries
# ----------------

# standard libraries
# -----
import torch
from torch.linalg import lstsq
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.cm import get_cmap
from matplotlib.colors import ListedColormap
import numpy as np


import config


# custom functions
# -----
@torch.no_grad()
def get_representations(model, data_loader):
    """
    Get all representations of the dataset given the network and the data loader
    params:
        model: the network to be used (torch.nn.Module)
        data_loader: data loader of the dataset (DataLoader)
    return:
        representations: representations output by the network (Tensor)
        labels: labels of the original data (LongTensor)
    """
    model.eval()
    features = []
    labels = []
    for data_samples, data_labels in data_loader:
        features.append(model(data_samples.to(config.DEVICE))[0])
        labels.append(data_labels.to(config.DEVICE))

    features = torch.cat(features, 0)
    labels = torch.cat(labels, 0)
    return features, labels


@torch.no_grad()
def lls_fit(train_features, train_labels, n_classes):
    """
        Fit a linear least square model
        params:
            train_features: the representations to be trained on (Tensor)
            train_labels: labels of the original data (LongTensor)
            n_classes: int, number of classes
        return:
            ls: the trained lstsq model (torch.linalg) 
    """
    ls = lstsq(train_features, F.one_hot(train_labels, n_classes).type(torch.float32))
    
    return ls

@torch.no_grad()
def lls_eval(trained_lstsq_model, eval_features, eval_labels):
    """
    Evaluate a trained linear least square model
    params:
        trained_lstsq_model: the trained lstsq model (torch.linalg)
        eval_features: the representations to be evaluated on (Tensor)
        eval_labels: labels of the data (LongTensor)
    return:
        acc: the LLS accuracy (float)
    """
    prediction = (eval_features @ trained_lstsq_model.solution)
    acc = (prediction.argmax(dim=-1) == eval_labels).sum() / len(eval_features)
    return prediction, acc



class AverageMeter(object):
    """
    Computes and stores the average and current float value
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        """
        Reset all stored values to zero
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        """
        Update stored values
        params:
            val: the tracked value (float)
            n: multiplier (int)
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the
    specified values of k
    params:
        output: network output, classification nodes
        target: labels in same format    
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

@torch.no_grad()
def supervised_eval(model, dataloader, criterion, no_classes):
    """
    Supervised evaluation run
    params:
        model: network model (torch.nn.Module)
        dataloader: torchvision dataloader (torch.util.data.DataLoader)
        criterion: loss function
        no_classes: number of total classes for one-hot function (int)
    """
    model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()
    
    for idx, (images, labels) in enumerate(dataloader):
        images = images.float()
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        bsz = labels.shape[0]
        
        # forward
        _, output = model(images)
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        
        top1.update(acc1[0], bsz)
        
        # update confusion matrix
        # confusion matrix not available in this code release
        
    print('Test: [{0}/{1}]\t'
          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
          'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
              idx, len(dataloader),
              loss=losses, top1=top1))
    
    # would normally also return confusion matrix
    return top1.avg, losses.avg, None
    

@torch.no_grad()
def wcss_bcss(representations, labels, n_classes):
    """
        Calculate the within-class and between-class average distance ratio
        params:
            representations: the representations to be evaluated (Tensor)
            labels: labels of the original data (LongTensor)
        return:
            wb: the within-class and between-class average distance ratio (float)
    """
    representations = [representations[labels == i] for i in range(n_classes)]
    centroids = torch.stack([r.mean(0, keepdim=True) for r in representations])
    wcss = [(r - centroids[i]).norm(dim=-1) for i,r in enumerate(representations)]
    wcss = torch.cat(wcss).mean()
    bcss = F.pdist(centroids.squeeze()).mean()
    wb = wcss / bcss
    return wb




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
