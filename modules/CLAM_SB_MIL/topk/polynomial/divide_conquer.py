import torch


def divide_and_conquer(x, k, mul):
    """
    Divide and conquer method for polynomial expansion
    x is a 2d tensor of size (n_classes, n_roots)
    The objective is to obtain the k first coefficients of the expanded
    polynomial
    """

    to_merge = []

    while x[0].dim() > 1 and x[0].size(0) > 1:
        size = x[0].size(0)
        half = size // 2
        if 2 * half < size:
            to_merge.append([t[-1] for t in x])
        x = mul([t[:half] for t in x],
                [t[half: 2 * half] for t in x])

    for row in to_merge:
        x = mul(x, row)
    x = torch.cat(x)
    return x
