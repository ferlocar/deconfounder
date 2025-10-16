# %%
import os
import numbers
import random
import numpy as np
import pandas as pd
import copy
import datetime
from sklearn.base import clone
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
from sklearn.utils.parallel import Parallel, delayed
from time import time
from metrics import subgroup_mse, get_rate
from metrics import expected_policy_outcome, expected_random_policy_outcome
from metrics import get_num_bins
from causal_tree.calibration import LinearEffectCalibration
from lightgbm import LGBMRegressor



class CriteoPipeline:
    """
    Input: 
        - df: DataFrame.
        - features: Feature names.
        - treated_prop: Proportion of treated individuals. Used for evaluating decision policy.
        - target: Column name of base scores
    """
    def __init__(self, df, features, treated_prop=0.1, target='visit'):
        self.df = df
        self.n_features = len(features)
        self.features = features
        self.treated_prop = treated_prop
        self.num_to_treat = None
        self.target = target


    def set_res_dir(self, res_dir):
        if not os.path.exists(res_dir):
            os.makedirs(res_dir, exist_ok=True)
        self.res_dir = res_dir


    def prepare_data(self, test_size=0.2, obs_size=1000000, random_state=None):

        random_state = check_random_state(random_state)

        treated_prop = self.treated_prop

        # 20% of data as test
        train_all, test = train_test_split(np.arange(self.df.shape[0]), 
                                           test_size=test_size, 
                                           random_state=random_state)
        self.num_to_treat = int(np.ceil(treated_prop * test.shape[0]))
        
        # Sample data from the control training set, to fit a scoring model.
        mask = (self.df.loc[train_all].treatment.values == 0)
        train_all_ctrl = train_all[mask]
        train_all_trmnt = train_all[~mask]

        obs, train_ctrl = train_test_split(train_all_ctrl, train_size=obs_size, random_state=random_state)

        # Sample data from the treated training set, with a size equal to "train_ctrl".
        train_trmnt = random_state.choice(train_all_trmnt, size=len(train_ctrl), replace=False)
        # Their combinations constitutes the balanced training set.
        train = list(train_ctrl) + list(train_trmnt)
        random_state.shuffle(train)

        # Store indices.
        self.obs = np.array(obs)
        self.train = np.array(train)
        self.test = np.array(test)


    def make_base_scores(self, random_state=None):
        """
        The target can be either 'visit' or 'f9'. 
        - For 'visit', fit a model to predict it. 
        - For 'f9', directly extract them as base scores.
        """
        features = self.features
        train = self.train
        test = self.test
        obs = self.obs
        target = self.target

        exp_indices = list(train) + list(test)
        if target == 'f9':
            base_scores = self.df.loc[exp_indices, target].values
            base_scores = pd.Series(data=base_scores, index=exp_indices)
        elif target == 'visit':
            X = self.df.loc[obs, features].values
            y = self.df.loc[obs, target].values
            model = LGBMRegressor(n_estimators=100, min_child_samples=5000, random_state=random_state)
            model.fit(X, y)
            base_scores = model.predict(self.df.loc[exp_indices, features].values)
            base_scores = pd.Series(data=base_scores, index=exp_indices)

        bs_te = base_scores.loc[test].values
        self.base_score_order = np.argsort(bs_te)   # The ordering will be used to calculate bin-based MSE.
        self.base_scores = base_scores

        return self.evaluate(bs_te, 'BS', len(self.obs))
    
    def evaluate_overall_ate(self):
        """
        Predict each test individual to be the ATE of the entire data.
        """
        t = self.df['treatment'].values
        y = self.df['conversion'].values
        ate = y[t==1].mean() - y[t==0].mean()
        self.ate = ate
        return self.evaluate(ate, 'ATE', self.df.shape[0])
    

    def evaluate(self, eff_pred, model_name, train_size):
        """
        Evaluate model performance on the test set for effect estimation,
        ranking and classification. 
        """
        test = self.test
        treated_prop = self.treated_prop
        num_to_treat = self.num_to_treat

        t_te = self.df.loc[test, 'treatment'].values
        y_te = self.df.loc[test, 'conversion'].values

        mse = subgroup_mse(t_te, y_te, eff_pred, order=self.base_score_order)
        autoc, qini = get_rate(t_te, y_te, eff_pred)

        if model_name.startswith('ATE'):
            epo = expected_random_policy_outcome(t_te, y_te, treated_prop)
        else:
            order = np.argsort(eff_pred)[::-1]
            decisions = np.zeros(eff_pred.shape[0], dtype=int)
            decisions[order[:num_to_treat]] = 1
            epo = expected_policy_outcome(t_te, y_te, decisions)
            

        res = {'model': model_name, 'train_size': train_size, 'mse': round(mse,9), 
               'autoc': round(autoc,9), 'qini': round(qini,9), 'epo': round(epo,9)}
        
        return res

    def linear_calibration(self, train_ix):
        """
        Monotonic post-processing.
        """
        bs_te = self.base_scores[self.test].values
        bs_tr = self.base_scores[train_ix].values
        t_tr = self.df.loc[train_ix, 'treatment'].values
        y_tr = self.df.loc[train_ix, 'conversion'].values

        n_bins = get_num_bins(len(train_ix))

        model = LinearEffectCalibration(n_bins=n_bins)
        model.fit(bs_tr, t_tr, y_tr)
        eff_pred = model.predict(bs_te)
        model_name = f"MC"
        res = self.evaluate(eff_pred, model_name, train_ix.shape[0])

        return res, model


    def causal_residual_learning(self, est_pair, train_ix, calibrator, random_state=None):
        """
        Causal residual forest.
        """
        feats = self.features
        test = self.test

        # Calibrated base scores
        bs_tr = self.base_scores[train_ix].values
        bs_te = self.base_scores[test].values
        cbs_tr = calibrator.predict(bs_tr)
        cbs_te = calibrator.predict(bs_te)

        t_tr = self.df.loc[train_ix, 'treatment'].values
        y_tr = self.df.loc[train_ix, 'conversion'].values
        X_tr = self.df.loc[train_ix, feats].values
        X_te = self.df.loc[test, feats].values

        model_name, estimator = est_pair
        model = clone(estimator)
        model.set_params(random_state=random_state)
        model.fit(X_tr, t_tr, y_tr, cbs_tr)
        eff_pred = cbs_te + model.predict(X_te)
        res = self.evaluate(eff_pred, model_name, train_ix.shape[0])

        return res
        
    
    def causal_effect_learning(self, est_pair, train_ix, random_state=None):
        """
        Causal forest.
        """
        feats = self.features
        test = self.test

        t_tr = self.df.loc[train_ix, 'treatment'].values
        y_tr = self.df.loc[train_ix, 'conversion'].values

        bs_tr = self.base_scores[train_ix].values
        bs_te = self.base_scores[test].values

        X_tr = self.df.loc[train_ix, feats].values
        X_te = self.df.loc[test, feats].values

        model_name, estimator = est_pair
        if model_name.startswith('CF-BS'):
            X_tr = np.c_[X_tr, bs_tr]
            X_te = np.c_[X_te, bs_te]
            X_tr = np.c_[X_tr, bs_tr]
            X_te = np.c_[X_te, bs_te]
        
        model = clone(estimator)
        model.set_params(random_state=random_state)
        model.fit(X_tr, t_tr, y_tr)
        eff_pred = np.squeeze(model.predict(X_te))
        res = self.evaluate(eff_pred, model_name, train_ix.shape[0])
        return res


    def single_experiment(self, est_pairs, train_size, seed=None):
        """
        Given the train size, run all models.
        """
        
        random_state = np.random.RandomState(seed)

        results = []
        train_ix = self.train[:train_size]                

        res, calibrator = self.linear_calibration(train_ix)
        results.append(res)
        
        for est_pair in est_pairs:
            model_name = est_pair[0]
            if model_name.startswith('CRF'):
                results.append(self.causal_residual_learning(est_pair, train_ix, calibrator, random_state=random_state))
            elif model_name.startswith('CF'):
                results.append(self.causal_effect_learning(est_pair, train_ix, random_state=random_state))
        return results
    
    def run_experiments(self, est_pairs, *, test_size, train_size_grid,
                        obs_size=1000000, seed=None, n_jobs=1):
        """
        Fix the test set and run all models once for each train size. 
        """
        
        start = time()

        # Set random generator 
        rng = np.random.default_rng(seed)
        # Generate random seeds: one for the master, N for the subprocesses.
        # For each subprocess, transfer a random state among functions. 
        all_seeds = rng.integers(low=0, high=2**32 - 1, size=1 + len(train_size_grid))
        main_seed = all_seeds[0]
        sub_seeds = all_seeds[1:]

        # Set random state for the main process.
        main_rs = np.random.RandomState(main_seed)

        self.prepare_data(test_size, obs_size, random_state=main_rs)
        bs_res = self.make_base_scores(obs_size)
        ate_res = self.evaluate_overall_ate()

        all_res = Parallel(n_jobs=n_jobs)(
            delayed(self.single_experiment)
            (
                est_pairs=est_pairs,
                train_size=train_size,
                seed=sub_seeds[i]
            ) for i, train_size in enumerate(train_size_grid)
        )

        all_res = [res for res_lst in all_res for res in res_lst]
        all_res = [ate_res, bs_res] + all_res
        res_df = pd.DataFrame(all_res)
        res_df.to_csv(os.path.join(self.res_dir, f'{seed}.csv'), index=False)

        print(f"Runtime = {(time()-start)/60:.3f}min")

        return res_df
    


