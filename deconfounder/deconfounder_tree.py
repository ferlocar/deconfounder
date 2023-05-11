from sklearn.tree import DecisionTreeRegressor
from deconfound_criterion import DeconfoundCriterion
import pandas as pd
import numpy as np

class DeconfounderTree(DecisionTreeRegressor):

    def fit(self, X, y, predictions, sample_weight=None, check_input=True):
        """
        Replaces the string stored in criterion by an instance of a class.
        """
        self.criterion = DeconfoundCriterion(1, X.shape[0])
        # Sort sample by prediction values in descending order.
        # After sorting, the sample index indicates the ranking by predicted values.
        order = list(range(X.shape[0]))
        order.sort(key=lambda i: -predictions[i])
        predictions = np.array(predictions)[order]
        X = X.iloc[order]   
        y = y.iloc[order]
        treated = X.treated.values.astype('int32')
        predictions = predictions.astype('float64')
        p_t = sum(treated)/len(treated)
        self.criterion.set_additional_parameters(treated, predictions, p_t)
        X_base = X.loc[:, X.columns != 'treated']
        DecisionTreeRegressor.fit(self, X_base, y, sample_weight=sample_weight, check_input=check_input)
        return self

    def predict(self, X, check_input=True):
        X_base = X.loc[:, X.columns != 'treated']
        return DecisionTreeRegressor.predict(self, X_base, check_input=check_input)

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
        df['decision'] = df.apply(lambda row: 1 if (row.pred-row.bias >= 0) else 0, axis=1)
        df['reward'] = df.apply(lambda row: row.y/row.p if (row.decision==row.treated) else 0,axis=1)
        score_ = - df.reward.mean()
        return score_