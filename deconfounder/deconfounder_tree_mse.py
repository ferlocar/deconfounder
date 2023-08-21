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

        p_t = np.mean(treatment)
        y_star = treatment * y_true / p_t - (1 - treatment) * y_true / (1 - p_t)
        corrected_scores = scores - self.predict(X)

        # mse = np.mean((corrected_scores - y_star)**2)
        mse = -2 * (corrected_scores * y_star).mean() + np.square(corrected_scores).mean()
        return -mse
