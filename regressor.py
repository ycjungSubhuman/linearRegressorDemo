import basis
import numpy as np

def _normalize(X):
    raise 'Not Implemented'

class LinearRegressor():
    def __init__(basis=basis.SimpleBasis):
        self.basis = basis

    """
    Args:
        X (np.array(shape=(M, D))) : Training Data (M : number of training samples, D : data dimension)
        T (np.array(shape=(1, M))) : Target Function Values
    Return:
        A function from (np.array(shape=(1,D))) to number
    """
    def fitTargetFunction(X, T):
        normalizedX = _normalize(X)
        Y = self.basis.apply(X)
        Yaug = np.append( X, np.ones(X.shape[1]) ) # Bias augmented
        raise 'Not Implemented'

