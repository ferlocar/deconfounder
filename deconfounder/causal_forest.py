from sklearn.ensemble import RandomForestRegressor
from mse_causal import CausalCriterion
import pandas as pd
import numpy as np


class CausalForest(RandomForestRegressor):

    def fit(self, X, y, sample_weight=None):
        """
        Replaces the string stored in criterion by an instance of a class.
        """
        self.criterion = CausalCriterion(1, X.shape[0])
        treated = X.treated.values.astype(int)
        self.criterion.set_treated(treated)
        X_base = X.loc[:, X.columns != 'treated']
        RandomForestRegressor.fit(self, X_base, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        X_base = X.loc[:, X.columns != 'treated']
        return RandomForestRegressor.predict(self, X_base)

    def score(self, X, y,  sample_weight=None):
        # This method does not support sample_weight
        X_base = X.loc[:, X.columns != "treated"]
        node_pred = self.apply(X_base)
        df = pd.DataFrame(node_pred, columns=["node_pred_by_tree" + str(i) for i in range(self.n_estimators)])
        df['treated'] = X.treated.values > 0
        df['y'] = np.array(y)
        test_effs_hat = np.zeros(df.shape[0])
        for i in range(self.n_estimators):
            pred_col = 'node_pred_by_tree' + str(i)
            treated_groups = df[df.treated].groupby(pred_col).y
            untreated_groups = df[~df.treated].groupby(pred_col).y
            # For more details about the formula, take a look at page 7357 (page 5) of the Athey & Imbens PNAS paper (Causal Tree)
            test_effs_hat += (treated_groups.mean() - untreated_groups.mean())[df[pred_col]].values
        test_effs_hat /= self.n_estimators
        model_effs_hat = self.predict(X_base)
        mse = -2 * (test_effs_hat * model_effs_hat).mean() + np.square(model_effs_hat).mean()
        return -mse