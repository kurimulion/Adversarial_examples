import torch
import torch.nn as nn
from torch.nn import init
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# code is adapted from assignments of CS231n

def show_images(images):
    images = np.reshape(images, [images.shape[0], -1])  # images reshape to (batch_size, D)
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape([sqrtimg,sqrtimg]))
    return

# weight initialization
# The selection doesn't really matter, when using batch norm in the network.
def initialize_weights(m, method='xavier', act='relu'):
    if method == 'He':
        if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight.data, nonlinearity=act)
    else:
        if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)

def sample_noise(batch_size, dim):
    """
    Generate a PyTorch Tensor of uniform random noise.

    Input:
    - batch_size: Integer giving the batch size of noise to generate.
    - dim: Integer giving the dimension of noise to generate.

    Output:
    - A PyTorch Tensor of shape (batch_size, dim) containing uniform
      random noise in the range (-1, 1).
    """
    return torch.rand(batch_size, dim) * 2 - 1
