import math
import torch
import torch.nn as nn
import torch.nn.functional as F
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

#

def adversarial_loss(model, x, target_y):
    """
    We implement the f6 from "Towards Evaluating the Robustness of Neural Networks", Nicholas Carlini & David Wagner.
    The loss is essential the difference between the maximum of logits of classes other than the target class and logits of the target
    class. Minimizing the loss in turns maximizes the difference which usually means maximize the logits of the target class
    and minimize logits of other classes.

    Inputs:
    model: target model
    x: perturbed input
    target_y: target class
    """
    # logits: N x 10
    logits = model(x)
    onehot_labels = F.one_hot(target_y, 10).type(dtype)
    target = torch.sum(logits * onehot_labels, dim=1)
    other, _ = torch.max((1 - onehot_labels) * logits, dim=1)
    zeros = torch.zeros_like(other)
    loss = torch.max(other - target, zeros).mean()
    return loss

def perturbation_loss(p, eps):
    """
    Perturbation loss, L2 norm of the perturbations, is used to bound the magnitude of perturbations.

    Inputs:
    p: generated perturbations
    eps: hyperparameter
    """
    N, C, _, _ = p.size()
    l2_norm = p.view(N, C, -1).norm(dim=2)
    loss = F.relu(l2_norm - eps).mean()
    return loss

# Hinge loss
def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss described above.

    Inputs:
    - logits_real: PyTorch Variable of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Variable of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Variable containing (scalar) the loss for the discriminator.
    """
    loss = None
    d_loss_real = torch.mean(F.relu(1 - logits_real))
    s_loss_fake = torch.mean(F.relu(1 + logits_fake))
    loss = d_loss_real + s_loss_fake
    return loss

def generator_loss(logits_fake):
    """
    Computes the generator loss described above.

    Inputs:
    - logits_fake: PyTorch Variable of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Variable containing the (scalar) loss for the generator.
    """
    loss = -torch.mean(logits_fake)
    return loss

# least square loss
def discriminator_loss_LS(logits_real, logits_fake):
    """

    Inputs:
    - logits_real: PyTorch Variable of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Variable of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Variable containing (scalar) the loss for the discriminator.
    """
    loss = None
    probs_real = torch.sigmoid(logits_real)
    probs_fake = torch.sigmoid(logits_fake)
    d_loss_real = F.mse_loss(probs_real, torch.ones_like(probs_real))
    s_loss_fake = F.mse_loss(probs_fake, torch.zeros_like(probs_fake))
    loss = 1/2 * (d_loss_real + s_loss_fake)
    return loss

def generator_loss_LS(logits_fake):
    """

    Inputs:
    - logits_fake: PyTorch Variable of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Variable containing the (scalar) loss for the generator.
    """
    probs_fake = torch.sigmoid(logits_fake)
    loss = 1/2 * F.mse_loss(probs_fake, torch.ones_like(probs_fake))
    return loss

# BCE loss
def discriminator_loss_BCE(logits_real, logits_fake):
    """
    Computes the discriminator loss described above.

    Inputs:
    - logits_real: PyTorch Variable of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Variable of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Variable containing (scalar) the loss for the discriminator.
    """
    loss = None
    probs_real = torch.sigmoid(logits_real)
    probs_fake = torch.sigmoid(logits_fake)
    d_loss_real = F.binary_cross_entropy(probs_real, torch.ones_like(probs_real))
    s_loss_fake = F.binary_cross_entropy(probs_fake, torch.zeros_like(probs_fake))
    loss = d_loss_real + s_loss_fake
    return loss

def generator_loss_BCE(logits_fake):
    """

    Inputs:
    - logits_fake: PyTorch Variable of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Variable containing (scalar) the loss for the generator.
    """
    loss = None
    probs_fake = torch.sigmoid(logits_fake)
    loss = F.binary_cross_entropy(probs_fake, torch.ones_like(probs_fake))
    return loss

# the module is taken from https://github.com/pytorch/pytorch/issues/8985#issuecomment-405080775
class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_classes):
        super(ConditionalBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0

    def forward(self, x, y):
        out = self.bn(x)
        y_ = y.type(torch.cuda.LongTensor)
        gamma, beta = self.embed(y_).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        return out
    
class ConditionalInstanceNorm2d(nn.Module):
    def __init__(self, num_features, num_classes):
        super(ConditionalInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.bn = nn.InstanceNorm2d(num_features, affine=False, track_running_stats=True)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0

    def forward(self, x, y):
        out = self.bn(x)
        y_ = y.type(torch.cuda.LongTensor)
        gamma, beta = self.embed(y_).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        return out
