import basis
import numpy as np

def _normalize(X):
    result = X.copy()
    mins = np.amix(X, axis=0)
    maxs = np.amax(X, axis=0)
    for (m, j), value in np.ndenumerate(result):
        mid = (maxs[j] + minx[j]) / 2
        result[m, j] = (value - m) / (maxs[j] - m)

    return result

class LinearRegressor():
    def __init__(basis=basis.SimpleBasis):
        self.basis = basis

    """
    Args:
        X (np.array(shape=(M, D))) : Training Data (M : number of training samples, D : data dimension)
        T (np.array(shape=(M,))) : Target Function Values
    Return:
        A function from (np.array(shape=(D,))) to number
    """
    def fitTargetFunction(X, T):
        designMatrixNonAugmented = self.basis.apply(_normalize(X))
        designMatrix = np.insert( designMatrixNonAugmented, 0, np.ones(designMatrixNonAugmented.shape[0]) )
        mlWeights = np.matmul(np.linalg.pinv(designMatrix), T)

        """
        Arg:
            x (np.array(shape=(D,))
        """
        def targetFunction(x):
            return np.dot(mlWeights, x)

        return targetFunction

