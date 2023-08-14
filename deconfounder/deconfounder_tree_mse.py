from sklearn.tree import DecisionTreeRegressor
from deconfound_criterion_mse import DeconfoundCriterion
import numpy as np
import pandas as pd

class DeconfounderTree(DecisionTreeRegressor):

    def fit(self, X, y, sample_weight=None, check_input=True):

        n_samples = X.shape[0]
        treatment, y_true, scores, cost = y[:, 0], y[:, 1], y[:, 2], y[:, 3]
        treatment = treatment.astype('int32')

        self.criterion = DeconfoundCriterion(1, n_samples)
        self.criterion.set_sample_parameters(treatment, scores)
        DecisionTreeRegressor.fit(self, X, y_true, sample_weight=sample_weight, check_input=check_input)

        return self

    def predict(self, X, check_input=True):
        return DecisionTreeRegressor.predict(self, X, check_input=check_input)
    
    def score(self, X, y, sample_weight=None):
        # Greater is better
        X, y = np.array(X), np.array(y)
        treatment, y_true, scores, cost = y[:, 0], y[:, 1], y[:, 2], y[:, 3]
        
        node_ids = self.apply(X)
        corrected_scores = scores - np.atleast_1d(self.tree_.value.squeeze())[node_ids]
        avg_yt = pd.Series(y_true[treatment==1]).groupby(node_ids[treatment==1]).mean()
        avg_yu = pd.Series(y_true[treatment==0]).groupby(node_ids[treatment==0]).mean()
        eff_test = (avg_yt - avg_yu).fillna(0)[node_ids].values

        mse = -2 * (corrected_scores * eff_test).mean() + np.square(corrected_scores).mean()
        return -mse
