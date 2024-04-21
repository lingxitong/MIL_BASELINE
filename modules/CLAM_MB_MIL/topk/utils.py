import math
import torch

import torch.autograd as ag


def delta(y, labels, alpha=None):
    """
    Compute zero-one loss matrix for a vector of ground truth y
    """


    labels = labels.to(y.device)
    if isinstance(y, ag.Variable):
        labels = ag.Variable(labels, requires_grad=False)
    
    delta = torch.ne(y[:, None], labels[None, :]).float()

    if alpha is not None:
        delta = alpha * delta
    return delta


def split(x, y, labels):
    labels = ag.Variable(labels, requires_grad=False)
    mask = torch.ne(labels[None, :], y[:, None])

    # gather result:
    # x_1: all scores that do contain the ground truth
    x_1 = x[mask].view(x.size(0), -1)
    # x_2: scores of the ground truth
    x_2 = x.gather(1, y[:, None]).view(-1)
    return x_1, x_2


def detect_large(x, k, tau, thresh):
    top, _ = x.topk(k + 1, 1)
    # switch to hard top-k if (k+1)-largest element is much smaller
    # than k-largest element
    hard = torch.ge(top[:, k - 1] - top[:, k], k * tau * math.log(thresh)).detach()
    smooth = hard.eq(0)
    return smooth, hard
