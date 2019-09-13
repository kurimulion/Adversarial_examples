import torch
import torch.nn.functional as F

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
