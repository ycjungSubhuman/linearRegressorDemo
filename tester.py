import itertools
import numpy as np
from scipy.stats import f_oneway

def _getRMSE(targetFunction, validateX, validateT):
    N = validateT.shape[0]
    return np.sqrt(sum([(validateT[i] - targetFunction(validateX[i]))**2 for i in range(N)]) / N)

def _normalize(X):
    result = X.copy()
    mins = np.amin(X, axis=0)
    maxs = np.amax(X, axis=0)
    for (m, j), value in np.ndenumerate(result):
        mid = (maxs[j] + mins[j]) / 2
        result[m, j] = (value - mid) / (maxs[j] - mid)

    return result

class CCPPTester:
    _FIELD_LABEL = ['T', 'P', 'H', 'V']
    def __init__(self, regressor):
        self.data = []
        print ('Loading CCPP data...')
        for i in range(0, 5):
            self.data.append( _normalize(np.loadtxt('CCPP/data{}.csv'.format(i+1), delimiter=',')) )
        print ('Loading Done')

        self.regressor = regressor
        self.msresOfFeatureSelection = []

    def _runFeatureSelection(self, featureIndices):
        print ( '  Parameters : {}'.format( '&'.join([self._FIELD_LABEL[i] for i in featureIndices]) ) )
        msres = np.zeros((5, 2))
        for batch in range(0, 5):
            Xs, Ts = np.split(self.data[batch][:, featureIndices], 2), np.split(self.data[batch][:, 4], 2)
            for i in range(0, 2):
                trainX, validateX = Xs[i], Xs[(i+1) % 2]
                trainT, validateT = Ts[i], Ts[(i+1) % 2]
                targetFunction = self.regressor.fitTargetFunction(trainX, trainT)
                msres[batch, i] = _getRMSE(targetFunction, validateX, validateT)
        self.msresOfFeatureSelection.append( msres.flatten() )
        print ( '  MSRE mean : {}, stddev : {}'.format( np.mean(msres.flatten()), np.std(msres.flatten()) ) )

    def _runParamDimension(self, paramDimension):
        indices = [0, 1, 2, 3]
        print ( '---Running {}C{} tests with {} parameters---'.format(4, paramDimension, paramDimension) )
        for featureIndices in itertools.combinations(indices, paramDimension):
            self._runFeatureSelection(featureIndices)

    def _anova(self):
        (fValue, pValue) = f_oneway(*self.msresOfFeatureSelection)
        print ( '---ANOVA Result---' )
        print ( '  F Value : {}'.format(fValue) )
        print ( '  P Value : {}'.format(pValue) )

    def run(self):
        for paramDimension in range(1, 5):
            self._runParamDimension(paramDimension)
        self._anova()

