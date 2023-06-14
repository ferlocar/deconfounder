
from sklearn.tree import DecisionTreeRegressor
from deconfound_criterion import DeconfoundCriterion
# from score.metrics import uplift_auc_score
import pandas as pd
import numpy as np

class DeconfounderTree(DecisionTreeRegressor):

    def fit(self, X, comb, sample_weight=None, check_input=True):
        """
        Replaces the string stored in criterion by an instance of a class.
        """
        n_samples = X.shape[0]
        X, comb = np.array(X), np.array(comb)
        treatment, y, prediction, cost = comb[:, 0], comb[:, 1], comb[:, 2], comb[:, 3]
        treatment = treatment.astype('int32')
        prediction = prediction.astype('float64')

        if isinstance(cost, (int, float)):
            cost = np.ones(n_samples, dtype=np.float64) * cost
        elif isinstance(cost, (list, np.ndarray)):
            cost = np.array(cost, dtype=np.float64)
        else:
            raise ValueError(f'cost should be a number or a list or an array')
        
        # Sort sample by prediction values in descending order.
        # After sorting, the sample index indicates the ranking by prediction.
        order = np.argsort(prediction, kind="mergesort")[::-1]
        X, y, treatment, prediction, cost = X[order], y[order], treatment[order], prediction[order], cost[order]

        p_t = np.sum(treatment)/n_samples
        
        self.criterion = DeconfoundCriterion(1, n_samples)
        self.criterion.set_sample_parameters(treatment, prediction, cost, p_t)
        DecisionTreeRegressor.fit(self, X, y, sample_weight=sample_weight, check_input=check_input)
        return self

    def predict(self, X, check_input=True):
        return DecisionTreeRegressor.predict(self, X, check_input=check_input)
    
    def score(self, X, comb, sample_weight=None):
        # Greater is better

        n_samples = X.shape[0]
        X, comb = np.array(X), np.array(comb)
        treatment, y, priority_score, cost = comb[:, 0], comb[:, 1], comb[:, 2], comb[:, 3]
        
        p_t = np.sum(treatment)/n_samples
        p_u = 1 - p_t
        corrected_score = priority_score - self.predict(X)
        decision = (corrected_score > 0)
        reward = (decision == treatment) * (y - decision*cost)
        score_ = (reward / ((decision==1)*p_t + (decision==0)*p_u)).mean()
        return score_