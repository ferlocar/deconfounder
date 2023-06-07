from sklearn.tree import DecisionTreeRegressor
from deconfound_criterion import DeconfoundCriterion
import pandas as pd
import numpy as np

class DeconfounderTree(DecisionTreeRegressor):

    def fit(self, X, y, treatment, prediction, cost=0, sample_weight=None, check_input=True):
        """
        Replaces the string stored in criterion by an instance of a class.
        """
        nsamples = X.shape[0]
        X, y, treatment = np.array(X), np.array(y), np.array(treatment, dtype='int32')
        prediction = np.array(prediction, dtype=np.float64)

        if isinstance(cost, (int, float)):
            cost = np.ones(nsamples, dtype=np.float64) * cost
        elif isinstance(cost, (list, np.ndarray)):
            cost = np.array(cost, dtype=np.float64)
        else:
            raise ValueError(f'cost should be a number or a list or an array')
        
        # Sort sample by prediction values in descending order.
        # After sorting, the sample index indicates the ranking by prediction.
        order = np.argsort(prediction, kind="mergesort")[::-1]
        X, y, treatment, prediction, cost = X[order], y[order], treatment[order], prediction[order], cost[order]

        p_t = np.sum(treatment)/nsamples
        
        self.criterion = DeconfoundCriterion(1, nsamples)
        self.criterion.set_sample_parameters(treatment, prediction, cost, p_t)
        DecisionTreeRegressor.fit(self, X, y, sample_weight=sample_weight, check_input=check_input)
        return self

    def predict(self, X, check_input=True):
        return DecisionTreeRegressor.predict(self, X, check_input=check_input)

    def score(self, X, y, predictions, sample_weight=None):
        # This method does not support sample_weight
        X_base = X.loc[:, X.columns != "treated"]
        bias_pred = self.predict(X_base)
        df = pd.DataFrame()
        df['bias'] = bias_pred
        df['treated'] = X.treated.values > 0
        df['y'] = np.array(y)
        df['pred'] = np.array(predictions)
        p_t = sum(df.treated)/len(df)
        df['p'] = df.apply(lambda row: p_t if row.treated==1 else 1-p_t, axis=1)
        df['decision'] = df.apply(lambda row: 1 if (row.pred-row.bias > 0) else 0, axis=1)
        df['reward'] = df.apply(lambda row: row.y/row.p if (row.decision==row.treated) else 0,axis=1)
        score_ = - df.reward.mean()
        return score_