#!/usr/bin/python
# _____________________________________________________________________________

# ----------------
# import libraries
# ----------------

# standard libraries
# -----
import os
import math
import torch
import errno
# configuration module
# -----
import config
import json

# custom functions
# -----

@torch.no_grad()
def update_target_network_parameters(target, source, tau):
    """
    function used to update the target net parameters to follow the running exponential average of online network.
        target: target network
        source: online network
        tau: hyper-parameter that controls the update
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data = target_param.data * tau + param.data * (1.0 - tau)


def initialize_target_network(target, source):
    """
    function used to initialize the target network in BYOL.
        target: target network
        source: online network
    """
    # init momentum network as encoder net
    for param_q, param_k in zip(source.parameters(), target.parameters()):
        param_k.data.copy_(param_q.data)  # initialize
        param_k.requires_grad = False  # not update by gradient


def save_model(net, writer, epoch):
    """
        function used to save model parameters to the log directory
            net: network to be saved
            writer: summary writer to get the log directory
            epoch: epoch indicator
    """
    log_dir = writer.get_logdir()
    path = os.path.join(log_dir, 'models')
    if not os.path.exists(path):
        os.mkdir(path)
    torch.save(net.state_dict(), os.path.join(path, f'epoch_{epoch}.pt'))


def load_model(net, path, device):
    """
        function used to load model parameters to the log directory
            net: network to load the parameters
            path: path to the saved model state dict
    """
    
    net.load_state_dict(torch.load(path, map_location=device))
    pass


def mkdir_p(path):
    """
    mkdir_p takes a string path and creates a directory at this path if it
    does not already exist.
    params:
        path: path to wherever (str)
    """
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
    
def save_args(results_dir, args):
    """
    save_args takes a string results_dir and a dict args and saves the dict
    to a json file 'args.json' at the directory results_dir
    params:
        results_dir: path to wherever (str)
        args: arguments to be saved (dict)
    """
    filename = 'args.json'
    # Save args
    l = []
    for a in args:
      try:
        l.append(args[a].items())
      except:
        l.append((a,args[a]))
    with open(os.path.join(results_dir, filename), 'w') as f:
        json.dump(l, f, indent=2)
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