from sklearn.tree import DecisionTreeRegressor
from deconfound_classification import DeconfoundClassification
import numpy as np
import pandas as pd

class DeconfoundClassifier(DecisionTreeRegressor):

    def fit(self, X, y, sample_weight=None, check_input=True):

        n_samples = X.shape[0]
        treatment, y_true, scores = y[:, 0], y[:, 1], y[:, 2]
        treatment = treatment.astype('int32')

        # Sort sample by prediction values in descending order.
        # After sorting, the sample index indicates the ranking by prediction.
        order = np.argsort(scores, kind="mergesort")[::-1]
        X, y_true, treatment, scores = X[order], y_true[order], treatment[order], scores[order]
        cost = np.zeros(n_samples)

        self.criterion = DeconfoundClassification(1, n_samples)
        self.criterion.set_sample_parameters(treatment, scores, cost)
        DecisionTreeRegressor.fit(self, X, y_true, sample_weight=sample_weight, check_input=check_input)

        return self

    def predict(self, X, check_input=True):
        return DecisionTreeRegressor.predict(self, X, check_input=check_input)
    
    def score(self, X, y, sample_weight=None):
        # Greater is better
        X, y = np.array(X), np.array(y)
        treatment, y_true, scores, cost = y[:, 0], y[:, 1], y[:, 2], y[:, 3]
        
        p_t = np.mean(treatment)
        corrected_scores = scores - self.predict(X)
        decision = (corrected_scores > 0)
        p = treatment * p_t + (1-treatment) * (1 - p_t)
        score_ = np.sum((decision == treatment) * y_true / p) / np.sum((decision == treatment) / p)
        return score_
