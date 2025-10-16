import numpy as np
from .causal_residual_mse import CausalResidualMSE
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


    
class CausalResidualTree(DecisionTreeRegressor):
    """
    A decision tree that learns residuals, defined as E[resid|leaf] = E[Y^1-Y^0|leaf] - E[score|leaf].
    Note that "scores" here refer to calibrated base scores, not raw scores.
    """
    
    def fit(self, X, t, y, scores, sample_weight=None, check_input=True):

        X = np.array(X)
        t = np.array(t).astype('int32')
        y = np.array(y)
        scores = np.array(scores).astype('float64')

        # Set criterion
        self.criterion = CausalResidualMSE(1, X.shape[0])
        self.criterion.set_sample_parameters(t, scores)
        super().fit(X, y, sample_weight=sample_weight, check_input=check_input)
        return self

    def predict(self, X, check_input=True):
        """
        Return residuals, so the cate prediction = score + resid.
        """
        return super().predict(X, check_input=check_input)
    
    
    

class CausalResidualForest(RandomForestRegressor):
    """
    An ensemble of causal residual trees.
    """
    
    def fit(self, X, t, y, scores, sample_weight=None):

        X = np.array(X)
        t = np.array(t).astype('int32')
        y = np.array(y)
        scores = np.array(scores).astype('float64')

        # Set criterion
        self.criterion = CausalResidualMSE(1, X.shape[0])
        self.criterion.set_sample_parameters(t, scores)
        super().fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return super().predict(X)
    

    

