import basis
import numpy as np

class LinearRegressor():
    def __init__(self, basis=basis.SimpleBasis):
        self.basis = basis

    """
    Args:
        X (np.array(shape=(M, D))) : Training Data (M : number of training samples, D : data dimension)
        T (np.array(shape=(M,))) : Target Function Values
    Return:
        A function from (np.array(shape=(D,))) to number
    """
    def fitTargetFunction(self, X, T):
        designMatrixNonAugmented = self.basis.apply(X)
        designMatrix = np.insert( designMatrixNonAugmented, 0, np.ones(designMatrixNonAugmented.shape[0]), axis=1)
        mlWeights = np.matmul(np.linalg.pinv(designMatrix), T)

        """
        Arg:
            x (np.array(shape=(D,))
        """
        def targetFunction(x):
            return np.dot( mlWeights, np.insert(x, 0, 1) )

        return targetFunction

