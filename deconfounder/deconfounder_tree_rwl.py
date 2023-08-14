
from sklearn.tree import DecisionTreeRegressor
from deconfound_criterion_rwl import DeconfoundCriterion
from score.metrics import evaluate
import pandas as pd
import numpy as np


class DeconfounderTree(DecisionTreeRegressor):

    def fit(self, X, y, sample_weight=None, check_input=True):

        n_samples = X.shape[0]
        treatment, y_true, scores, cost = y[:, 0], y[:, 1], y[:, 2], y[:, 3]
        treatment = treatment.astype('int32')

        # Sort sample by prediction values in descending order.
        # After sorting, the sample index indicates the ranking by prediction.
        order = np.argsort(scores, kind="mergesort")[::-1]
        X, y_true, treatment, scores, cost = X[order], y_true[order], treatment[order], scores[order], cost[order]

        self.criterion = DeconfoundCriterion(1, n_samples)
        self.criterion.set_sample_parameters(treatment, scores, cost)
        DecisionTreeRegressor.fit(self, X, y_true, sample_weight=sample_weight, check_input=check_input)

        # node_ids = self.apply(X)
        # avg_yt = pd.Series(y_true[treatment==1]).groupby(node_ids[treatment==1]).mean()
        # avg_yu = pd.Series(y_true[treatment==0]).groupby(node_ids[treatment==0]).mean()
        # self.node_shift = (avg_yt + avg_yu) / 2

        return self

    def predict(self, X, check_input=True):
        return DecisionTreeRegressor.predict(self, X, check_input=check_input)
    
    def score(self, X, y, sample_weight=None):
        # Greater is better
        X, y = np.array(X), np.array(y)
        treatment, y_true, scores, cost = y[:, 0], y[:, 1], y[:, 2], y[:, 3]

        node_ids = self.apply(X)
        p_t = pd.Series(treatment).groupby(node_ids).mean()[node_ids].values
        avg_yt = pd.Series(y_true[treatment==1]).groupby(node_ids[treatment==1]).mean()
        avg_yu = pd.Series(y_true[treatment==0]).groupby(node_ids[treatment==0]).mean()
        shift = ((avg_yt + avg_yu) / 2).fillna(0)[node_ids].values
        # shift = self.node_shift[node_ids].values

        corrected_scores = scores - np.atleast_1d(self.tree_.value.squeeze())[node_ids]
        decision = (corrected_scores > 0)
        reward = (y_true - shift) / (treatment * p_t + (1 - treatment) * (1 - p_t))
        score_ = np.mean((decision == treatment) * reward)
        return score_