import numpy as np
import basisutil

def _unnormalizedGaussian(x, mu, sigma):
    assert(x.shape==mu.shape)
    return np.exp( -((x-mu)**2) / (2*(sigma**2)) )

def _logisticSigmoid(x):
    return 1 / (1 + np.exp(-x))

def _sigmoid(x, mu, sigma):
    assert(x.shape==mu.shape)
    return _logisticSigmoid( (x-mu) / sigma )

class Basis():
    """
    Args:
        x (np.array(shape=(M, D)): input vector (D: data dimension)
            all fields should be normalized to [-1, 1]
    Return:
        basis function value vector (np.array(shape=(M, D)))
    """
    def apply(self, x):
        raise 'Not Implemented'

"""
phi_j(x) = x
"""
class SimpleBasis(Basis):
    def apply(self, x):
        return x

"""
phi_j(x) = x^j
"""
class PowerBasis(Basis):
    def apply(self, x):
        result = x.copy()
        for (m, j), value in numpy.ndenumerate(result):
            result[m, j] = value**(j+1)
        return result

"""
phi_j(x) with unnormalizedGaussian
    mu_j s are determined s.t. they evenly devide [-1, 1]
"""
class GaussianBasis(Basis):
    def __init__(self, sigma=1):
        self.sigma = sigma

    def apply(self, x):
        mus = np.arange(start=-1, stop=1, step=(2/x.shape[1]))
        return _unnormalizedGaussian(x, mus, self.sigma)

"""
phi_j(x) with sigmoid
    mu_j s are determined s.t. they evenly devide [-1, 1]
"""
class SigmoidBasis(Basis):
    def __init__(self, sigma=1):
        self.sigma = sigma

    def apply(self, x):
        mus = np.arange(start=-1, stop=1, step=(2/x.shape[1]))
        return _sigmoid(x, mus, self.sigma)

