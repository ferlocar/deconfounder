from sklearn.tree import DecisionTreeRegressor
from mse_causal import CausalCriterion
import pandas as pd
import numpy as np


class CausalTree(DecisionTreeRegressor):

    def fit(self, X, y, sample_weight=None, check_input=True, X_idx_sorted=None):
        """
        Replaces the string stored in criterion by an instance of a class.
        """
        self.criterion = CausalCriterion(1, X.shape[0])
        treated = X.treated.values.astype(np.int32)
        self.criterion.set_treated(treated)
        X_base = X.loc[:, X.columns != 'treated']
        # Careful! scikit-learn 0.24.2 creates the CausalCriterion again when I call the DecisionTreeRegressor.fit method
        # As a result, set_treated is not called, and the init method in CausalCriterion suffers from a segmentation fault error
	# To avoid this error, stick to version 0.24.1.
	# Check out the description of the update in the tree package for clues on what to do:
	# https://scikit-learn.org/stable/whats_new/v0.24.html
        DecisionTreeRegressor.fit(self, X_base, y, sample_weight=sample_weight, check_input=check_input)
        return self

    def predict(self, X, check_input=True):
        X_base = X.loc[:, X.columns != 'treated']
        return DecisionTreeRegressor.predict(self, X_base, check_input=check_input)

    def score(self, X, y,  sample_weight=None):
        # This method does not support sample_weight
        X_base = X.loc[:, X.columns != "treated"]
        node_pred = self.apply(X_base)
        df = pd.DataFrame()
        df['node'] = node_pred
        df['treated'] = X.treated.values > 0
        df['y'] = np.array(y)
        treated_groups = df[df.treated].groupby('node').y
        untreated_groups = df[~df.treated].groupby('node').y
        # For more details about the formula, take a look at page 7357 (page 5) of the Athey & Imbens PNAS paper (Causal Tree)
        test_effs_hat = (treated_groups.mean() - untreated_groups.mean())[df.node]
        model_effs_hat = self.predict(X_base)
        mse = -2 * (test_effs_hat * model_effs_hat).mean() + np.square(model_effs_hat).mean()
        return -mse
