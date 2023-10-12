from sklearn.tree import DecisionTreeRegressor
from mse_causal import CausalCriterion
import pandas as pd
import numpy as np

class CausalTree(DecisionTreeRegressor):

    def fit(self, X, y, sample_weight=None, check_input=True):
        """
        Replaces the string stored in criterion by an instance of a class.
        """
        treatment, y_true = np.array(y)[:, 0], np.array(y)[:, 1]
        self.criterion = CausalCriterion(1, X.shape[0])
        treatment = treatment.astype('int32')
        self.criterion.set_treated(treatment)
        DecisionTreeRegressor.fit(self, X, y_true, sample_weight=sample_weight, check_input=check_input)
        return self

    def predict(self, X, check_input=True):
        return DecisionTreeRegressor.predict(self, X, check_input=check_input)

    def score(self, X, y, sample_weight=None):
        # This method does not support sample_weight
        treatment, y_true = np.array(y)[:, 0], np.array(y)[:, 1]
        node_pred = self.apply(X)
        df = pd.DataFrame()
        df['node'] = node_pred
        df['treated'] = treatment>0
        df['y'] = np.array(y_true)
        treated_groups = df[df.treated].groupby('node').y
        untreated_groups = df[~df.treated].groupby('node').y
        # For more details about the formula, take a look at page 7357 (page 5) of the Athey & Imbens PNAS paper (Causal Tree)
        test_effs_hat = (treated_groups.mean() - untreated_groups.mean())[df.node]
        model_effs_hat = self.predict(X)
        mse = -2 * (test_effs_hat * model_effs_hat).mean() + np.square(model_effs_hat).mean()
        return -mse