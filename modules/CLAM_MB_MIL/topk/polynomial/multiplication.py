import operator
import itertools

from future.builtins import range
from functools import reduce
from ..logarithm import LogTensor


def Multiplication(k):
    """
    Generate a function that performs a polynomial multiplication and return coefficients up to degree k
    """
    assert isinstance(k, int) and k > 0

    def isum(factors):
        init = next(factors)
        return reduce(operator.iadd, factors, init)

    def mul_function(x1, x2):

        # prepare indices for convolution
        l1, l2 = len(x1), len(x2)
        M = min(k + 1, l1 + l2 - 1)
        indices = [[] for _ in range(M)]
        for (i, j) in itertools.product(range(l1), range(l2)):
            if i + j >= M:
                continue
            indices[i + j].append((i, j))

        # wrap with log-tensors for stability
        X1 = [LogTensor(x1[i]) for i in range(l1)]
        X2 = [LogTensor(x2[i]) for i in range(l2)]

        # perform convolution
        coeff = []
        for c in range(M):
            coeff.append(isum(X1[i] * X2[j] for (i, j) in indices[c]).torch())
        return coeff

    return mul_function
