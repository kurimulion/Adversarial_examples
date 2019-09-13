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
