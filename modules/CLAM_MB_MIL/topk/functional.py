import torch
import torch.autograd as ag

from .polynomial.sp import log_sum_exp, LogSumExp
from .logarithm import LogTensor
from .utils import delta, split


def Top1_Hard_SVM(labels, alpha=1.):
    def fun(x, y):
        # max oracle
        max_, _ = (x + delta(y, labels, alpha)).max(1)
        # subtract ground truth
        loss = max_ - x.gather(1, y[:, None]).squeeze()
        return loss
    return fun


def Topk_Hard_SVM(labels, k, alpha=1.):
    def fun(x, y):
        x_1, x_2 = split(x, y, labels)

        max_1, _ = (x_1 + alpha).topk(k, dim=1)
        max_1 = max_1.mean(1)

        max_2, _ = x_1.topk(k - 1, dim=1)
        max_2 = (max_2.sum(1) + x_2) / k

        loss = torch.clamp(max_1 - max_2, min=0)

        return loss
    return fun


def Top1_Smooth_SVM(labels, tau, alpha=1.):
    def fun(x, y):
        # add loss term and subtract ground truth score
        x = x + delta(y, labels, alpha) - x.gather(1, y[:, None])
        # compute loss
        loss = tau * log_sum_exp(x / tau)

        return loss
    return fun


def Topk_Smooth_SVM(labels, k, tau, alpha=1.):

    lsp = LogSumExp(k)

    def fun(x, y):
        x_1, x_2 = split(x, y, labels)
        # all scores are divided by (k * tau)
        x_1.div_(k * tau)
        x_2.div_(k * tau)

        # term 1: all terms that will *not* include the ground truth score
        # term 2: all terms that will include the ground truth score
        res = lsp(x_1)
        term_1, term_2 = res[1], res[0]
        term_1, term_2 = LogTensor(term_1), LogTensor(term_2)

        X_2 = LogTensor(x_2)
        cst = x_2.data.new(1).fill_(float(alpha) / tau)
        One_by_tau = LogTensor(ag.Variable(cst, requires_grad=False))
        Loss_ = term_2 * X_2

        loss_pos = (term_1 * One_by_tau + Loss_).torch()
        loss_neg = Loss_.torch()
        loss = tau * (loss_pos - loss_neg)

        return loss
    return fun
