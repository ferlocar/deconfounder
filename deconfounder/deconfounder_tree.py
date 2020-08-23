from sklearn.tree import DecisionTreeRegressor
from .mse_deconfound import DeconfoundCriterion
import pandas as pd
import numpy as np

class DeconfounderTree(DecisionTreeRegressor):

    def fit(self, X, y, sample_weight=None, check_input=True, X_idx_sorted=None):
        """
        Replaces the string stored in criterion by an instance of a class.r
        """
        self.criterion = DeconfoundCriterion(1, X.shape[0])
        treated = X.treated.values.astype(int)
        effects = X.effect.values
        self.criterion.set_treated_and_effects(treated, effects)
        X_base = X.loc[:, X.columns != 'treated']
        X_base = X_base.loc[:, X_base.columns != 'effect']
        DecisionTreeRegressor.fit(self, X_base, y, sample_weight=sample_weight, check_input=check_input,
                                  X_idx_sorted=X_idx_sorted)
        return self

    def predict(self, X, check_input=True):
        X_base = X.loc[:, X.columns != 'treated']
        X_base = X_base.loc[:, X_base.columns != 'effect']
        return DecisionTreeRegressor.predict(self, X_base, check_input=check_input)

    def score(self, X, y,  sample_weight=None):
        # This method does not support sample_weight
        X_base = X.loc[:, X.columns != 'treated']
        X_base = X_base.loc[:, X_base.columns != 'effect']
        node_pred = self.apply(X_base)
        df = pd.DataFrame()
        df['node'] = node_pred
        df['treated'] = X.treated.values > 0
        df['effect'] = X.effect.values
        df['y'] = np.array(y)
        obs_effect_groups = df.groupby('node').effect
        exp_treated_groups = df[df.treated].groupby('node').y
        exp_untreated_groups = df[~df.treated].groupby('node').y


        test_obs_hat = obs_effect_groups.mean()
        test_exp_hat = exp_treated_groups.mean() - exp_untreated_groups.mean()
        test_bias_hat = (test_obs_hat - test_exp_hat)[df.node]
        model_bias_hat = self.predict(X_base)
        mse = -2 * (test_bias_hat * model_bias_hat).mean() + np.square(model_bias_hat).mean()
        return -mse