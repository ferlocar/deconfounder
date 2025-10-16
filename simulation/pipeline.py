"""
Run all models once for each training size.
"""
# %%
import os
import numpy as np
import pandas as pd
import copy
from time import time
from metrics import get_mse, expected_policy_effect, rate_with_effect, get_num_bins
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.utils.parallel import Parallel, delayed
from causal_tree.calibration import LinearEffectCalibration


def is_leaf_size_bad(train_size, model_name):
    """
    filter out too large leaves for small data and too small leaves for large data.
    """
    msl_hi = train_size / 2     # if the leaf is too large, then tree may fail to split.
    msl_lo = train_size * 1e-4

    msl = int(model_name.split('_')[1])
    if (msl > msl_hi) or (msl < msl_lo):
        return True
    return False


class SimulationPipeline:
    """
    Input: 
        - dg: DataGenerator
        - use_thresh: whether to use a fixed threshold to target.
        - treated_thresh: if use_thresh=True, the individuals with scores greated than treated_thresh will be targeted.
        - treated_prop: if use_thresh=False, it determines the proportion of targeted individuals.
        - has_pretuned: whether to use pre-tuned parameters, used on large dataset.
    """

    def __init__(
            self, 
            dg, 
            use_thresh=False, 
            treated_thresh=0,  
            treated_prop=0.1,
            has_pretuned=False          
        ):
        self.dg = copy.deepcopy(dg)
        self.n_features = dg.n_features
        self.use_thresh = use_thresh 
        self.treated_thresh = treated_thresh
        self.treated_prop = treated_prop     
        self.num_to_treat = None                                                 
        self.features = [f"x{k}" for k in range(dg.n_features)]
        self.has_pretuned = has_pretuned


    def set_res_dir(self, res_dir):
        if not os.path.exists(res_dir):
            os.makedirs(res_dir, exist_ok=True)
        self.res_dir = res_dir

    def load_pretuned_data(self, filepath):
        self.df_pre = pd.read_csv(filepath)

    def setup_experiment(self, exp_size, test_size, seed, num_subs):
 
        print(f"Experimental data size: {exp_size}, of which test size: {test_size}")

        # Set random generator 
        rng = np.random.default_rng(seed)
        # Generate random seeds: one for the master, N for the subprocesses.
        # For each subprocess, transfer a random state among functions. 
        all_seeds = rng.integers(low=0, high=2**32 - 1, size=1+num_subs)
        main_seed = all_seeds[0]
        sub_seeds = all_seeds[1:]

        # Set random state for the main process.
        main_rs = np.random.RandomState(main_seed)
        
        self.experimental_data(exp_size, test_size, random_state=main_rs)

        return sub_seeds


    def experimental_data(self, exp_size, test_size, random_state=None):
        """
        Generate experimental data and split it into train and test set.
        """
        treated_prop = self.treated_prop

        self.dg.reset_coef(random_state=random_state)
        exp_data = self.dg.generate_data(exp_size, random_state=random_state)

        # Split data into train and test set.
        train, test = train_test_split(np.arange(exp_size), test_size=test_size, random_state=random_state)

        num_to_treat = int(np.ceil(treated_prop  * test_size))

        self.exp_data = exp_data
        self.train = train
        self.test = test
        self.num_to_treat = num_to_treat
        

    def make_base_scores(self, test_ix, train_size=0):
        """
        - Evaluate base scores on the test set.
        """
        bs_te = self.exp_data.loc[test_ix, 'bs'].values
        res = self.evaluate(test_ix, bs_te, 'BS', train_size)
        return res


    def evaluate(self, test_ix, eff_pred, model_name, train_size=0):
        """
        Evaluate model performance on the test set for effect estimation,
        ranking and classification. 
        """
        num_to_treat = self.num_to_treat

        eff_true = self.exp_data.loc[test_ix, 'cate'].values

        mse = get_mse(eff_true, eff_pred)
        autoc, qini = rate_with_effect(eff_true, eff_pred)

        if self.use_thresh:
            decisions = 1 * (eff_pred > self.treated_thresh)
        else:
            order = np.argsort(eff_pred)[::-1]
            decisions = np.zeros(eff_pred.shape[0], dtype=int)
            decisions[order[:num_to_treat]] = 1

        epg = expected_policy_effect(eff_true, decisions)

        res = {'model': model_name, 'train_size': train_size, 'mse': round(mse, 9), 'autoc': 
               round(autoc, 9), 'qini': round(qini, 9), 'epg': round(epg, 9)}

        return res
    
    
    def linear_calibration(self, train_ix, test_ix):
        """
        Monotonic calibration.
        """

        bs_te = self.exp_data.loc[test_ix, 'bs'].values
        bs_tr = self.exp_data.loc[train_ix, 'bs'].values
        t_tr = self.exp_data.loc[train_ix, 't'].values
        y_tr = self.exp_data.loc[train_ix, 'y'].values

        n_bins = get_num_bins(len(train_ix))

        model = LinearEffectCalibration(n_bins=n_bins)
        model.fit(bs_tr, t_tr, y_tr)
        eff_pred = model.predict(bs_te)
        model_name = "MC"
        res = self.evaluate(test_ix, eff_pred, model_name, train_ix.shape[0])

        return res, model


    def causal_residual_learning(self, est_pair, train_ix, test_ix, calibrator=None, random_state=None):
        """
        Causal residual forest.
        """
        feats = self.features

        model_name, estimator = est_pair

        # Calibrated base scores
        bs_te = self.exp_data.loc[test_ix, 'bs'].values
        bs_tr = self.exp_data.loc[train_ix, 'bs'].values
        cbs_tr = calibrator.predict(bs_tr)
        cbs_te = calibrator.predict(bs_te)

        t_tr = self.exp_data.loc[train_ix, 't'].values
        y_tr = self.exp_data.loc[train_ix, 'y'].values
        X_tr = self.exp_data.loc[train_ix, feats].values
        X_te = self.exp_data.loc[test_ix, feats].values
       
        model = clone(estimator)
        model.set_params(random_state=random_state)
        model.fit(X_tr, t_tr, y_tr, cbs_tr)
        eff_pred = cbs_te + model.predict(X_te)
        res = self.evaluate(test_ix, eff_pred, model_name, train_ix.shape[0])

        return res
        
    
    def causal_effect_learning(self, est_pair, train_ix, test_ix, random_state=None):
        """
        Causal forest.
        """
        feats = self.features

        t_tr = self.exp_data.loc[train_ix, 't'].values
        y_tr = self.exp_data.loc[train_ix, 'y'].values

        bs_te = self.exp_data.loc[test_ix, 'bs'].values
        bs_tr = self.exp_data.loc[train_ix, 'bs'].values

        X_tr = self.exp_data.loc[train_ix, feats].values
        X_te = self.exp_data.loc[test_ix, feats].values

        model_name, estimator = est_pair
        if model_name.startswith('CF-BS'):
            X_tr = np.c_[X_tr, bs_tr]
            X_te = np.c_[X_te, bs_te]

        model = clone(estimator)
        model.set_params(random_state=random_state)
        model.fit(X_tr, t_tr, y_tr)
        eff_pred = np.squeeze(model.predict(X_te))
        res = self.evaluate(test_ix, eff_pred, model_name, train_ix.shape[0])
        return res


    def single_train_size(self, est_pairs, train_size, seed=None):
        """
        Given a training size, run all models.
        """
        random_state = np.random.RandomState(seed)

        results = []
        test_ix = self.test  
        train_ix = self.train[:train_size]      

        res, calibrator = self.linear_calibration(train_ix, test_ix)
        results.append(res)
        
        for est_pair in est_pairs:
            model_name = est_pair[0]

            if  is_leaf_size_bad(train_size, model_name):
                continue

            if model_name.startswith('CRF'):
                results.append(
                    self.causal_residual_learning(est_pair, train_ix, test_ix, calibrator, random_state=random_state))
            elif model_name.startswith('CF'):
                results.append(
                    self.causal_effect_learning(est_pair, train_ix, test_ix, random_state=random_state))
        return results
    
    
    def single_simulation(self, est_pairs, *, exp_size, test_size, train_size_grid, seed=None, n_jobs=1):
        """
        Fix the test set and run all models once for each train size. 
        """
        
        start = time()

        # Setup experiment (data and random seeds).
        sub_seeds = self.setup_experiment(exp_size, test_size, seed, num_subs=len(train_size_grid))

        # BS performance does not change with train size.
        bs_res = self.make_base_scores(self.test) 

        all_res = Parallel(n_jobs=n_jobs)(
            delayed(self.single_train_size)
            (
                est_pairs=est_pairs,
                train_size=train_size,
                seed=sub_seeds[i],
            ) for i, train_size in enumerate(train_size_grid)
        )

        all_res = [res for res_lst in all_res for res in res_lst]
        all_res = [bs_res] + all_res
        res_df = pd.DataFrame(all_res)
        res_df.to_csv(os.path.join(self.res_dir, f'{seed}.csv'), index=False)

        print(f"Runtime = {(time()-start)/60:.3f}min")

        return
    
    

