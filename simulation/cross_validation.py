"""
- For each simulation, conduct repeated cross-validation for each model at each training size. Record the mean performance across folds.
- The model selection pipeline inherits from the simulation pipeline to maintain consistency in the data for each simulation.
- During evaluation, true CATE cannot be used. Instead, treatment and outcome are used for estimation purposes.
"""
# %%
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

import numpy as np
import pandas as pd
from time import time
import itertools
import argparse
from metrics import transformed_mse
from metrics import expected_policy_outcome
from metrics import get_rate
from metrics import get_num_bins
from sklearn.base import clone
from sklearn.model_selection import RepeatedKFold
from sklearn.utils.parallel import Parallel, delayed
from causal_tree.causal_residual_tree import CausalResidualForest
from econml.grf import CausalForest
from simulation.pipeline import SimulationPipeline, is_leaf_size_bad
from simulation.data_generation import DataGenerator


def get_n_repeats(train_size):
    if train_size<=8000:
        return 10
    return 1

def get_n_splits(train_size):
    return 5

def get_n_folds(train_size):
    """
    For large data size, just run the first fold but still use repeatedKFold.
    """
    if train_size >= 128000:
        return 1
    n_folds = get_n_repeats(train_size) * get_n_splits(train_size)
    return int(n_folds)


def use_pretuned_config(train_size):
    """
    use the pretuned hyperparameters for medium to large data.
    """
    return (train_size >= 16000)


class ModelSelectionForSimulation(SimulationPipeline):

    def skip_model(self, train_size, model_name):
        """
        Skip the model for some reasons:
            - improper min_samples_leaf values.
            - for medium data, the model is not the pretuned config.
        """
        if is_leaf_size_bad(train_size, model_name):
            return True
        
        if self.has_pretuned and use_pretuned_config(train_size):
            mask = self.df_pre.train_size==train_size
            is_pretuned = any(self.df_pre.loc[mask, 'model']==model_name)
            return not is_pretuned
        return False
        

    def evaluate(self, val_ix, eff_pred, model_name, train_size=0):
        """
        Evaluate model performance on the validation set for effect estimation,
        ranking and classification. 
        """
        treated_prop = self.treated_prop
        n_bins = get_num_bins(val_ix.shape[0])

        t_val = self.exp_data.loc[val_ix, 't'].values
        y_val = self.exp_data.loc[val_ix, 'y'].values

        mse = transformed_mse(t_val, y_val, eff_pred)
        autoc, qini = get_rate(t_val, y_val, eff_pred, n_bins=n_bins)       # n_bins is adaptive to val size.

        if self.use_thresh:
            decisions = 1 * (eff_pred > self.treated_thresh)
        else:
            num_to_treat = int(np.ceil(t_val.shape[0] * treated_prop))
            order = np.argsort(eff_pred)[::-1]
            decisions = np.zeros(eff_pred.shape[0], dtype=int)
            decisions[order[:num_to_treat]] = 1

        epo = expected_policy_outcome(t_val, y_val, decisions)

        train_size += val_ix.shape[0]   # Save the whole training set size.

        res = {'model': model_name, 'train_size': train_size, 'mse': mse, 
               'autoc': autoc, 'qini': qini, 'epo': epo}

        return res
    
    def single_fold(self, est_pairs, train_size, n_repeats, n_splits, fold_index, seed=None):
        
        random_state = np.random.RandomState(seed)

        # Reintialize the cv splitter using the same random state.
        cv = RepeatedKFold(n_repeats=n_repeats, n_splits=n_splits,random_state=random_state)

        # Use itertools.islice to get the j-th fold.
        # It consumes the generator up to the j-th split and yields it.
        fold_j = itertools.islice(cv.split(range(train_size)), fold_index, fold_index + 1)
        tr_pos, val_pos = next(fold_j)

        # Get train index and val index.
        train_ix = self.train[:train_size]
        tr_ix = train_ix[tr_pos]
        val_ix = train_ix[val_pos]

        # Start CV blocks.
        results = []

        # Base scores
        results.append(self.make_base_scores(val_ix, tr_ix.shape[0]))

        # Monotonic calibration
        res, calibrator = self.linear_calibration(tr_ix, val_ix)
        results.append(res)

        # CF, CF-BS and CRF
        for est_pair in est_pairs:
            model_name = est_pair[0]

            if self.skip_model(train_size, model_name):
                continue
            if model_name.startswith('CRF'):
                res = self.causal_residual_learning(est_pair, tr_ix, val_ix, calibrator, random_state=random_state)
                results.append(res)
            elif model_name.startswith('CF'):
                res = self.causal_effect_learning(est_pair, tr_ix, val_ix, random_state=random_state)
                results.append(res)

        return results
    

    def single_simulation(self, est_pairs, *, exp_size, test_size, train_size_grid, seed=None, n_jobs=1):
        """
        Given a dataset, run cv for all models for each train size. 
        """

        start = time()

        # Setup main logger
        self.master_seed = seed
        # logger = self.set_main_logger()

        # Setup experiment (data and random seeds)
        sub_seeds = self.setup_experiment(exp_size, test_size, seed, num_subs=len(train_size_grid))

        nested_list = Parallel(n_jobs=n_jobs)(
            delayed(self.single_fold)(
                est_pairs,
                train_size,
                n_repeats=get_n_repeats(train_size),
                n_splits=get_n_splits(train_size),
                fold_index=j,
                seed=sub_seeds[i]
            )
            for i, train_size in enumerate(train_size_grid)
            for j in range(get_n_folds(train_size))
        )

        all_res = [res for res_lst in nested_list for res in res_lst]
        df_res = pd.DataFrame(all_res)

        df_res = df_res.groupby(by=['model', 'train_size']).mean().reset_index()
        columns_to_round = ['mse', 'autoc', 'qini', 'epo']
        df_res[columns_to_round] = df_res[columns_to_round].round(9)
        
        df_res.to_csv(os.path.join(self.res_dir, f'{seed}.csv'), index=False)

        print(f"Runtime = {(time()-start)/60:.3f}min")

        return 


    
# %%
if __name__=="__main__":

    start = time()

    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretune_file', help='The path with pretuned paramters.')
    parser.add_argument('--rho', help='Alignment between base scores and CATE', type=float, default=0.7)
    parser.add_argument('--var_noise', help='Variance of outcome noise', type=float, default=1.6)
    parser.add_argument('--seed', help='Random seed', type=int, default=1)
    args = parser.parse_args()

    pretune_file = args.pretune_file
    rho = args.rho
    var_noise = args.var_noise
    seed = args.seed

    print(f"rho = {rho}")
    print(f"var_noise = {var_noise}")
    print(f"seed = {seed}")
    
    # DGP setup
    n_features = 20
    dg = DataGenerator(
        n_features=n_features, 
        y0_mean=0,
        ate=0.1,
        var_cay0=1,
        var_ratio=0.1,
        rho=rho,
        var_noise=var_noise,
        feature_map='identity',
        nonlinearity=None,
    )

    # Model setups
    est_pairs = []
    msl_grid = [25*(2**i) for i in range(8)]
    estimators = {
        'CF': CausalForest(n_estimators=100, criterion='het', min_samples_split=2, honest=False),
        'CF-BS': CausalForest(n_estimators=100, criterion='het', min_samples_split=2, honest=False),
        'CRF': CausalResidualForest(n_estimators=100, min_samples_split=2)
    }
    for method in ['CF', 'CF-BS', 'CRF']:
        for msl in msl_grid:
            model_name =f'{method}_{msl}'
            estimator = clone(estimators[method])
            estimator.set_params(min_samples_leaf=msl)
            est_pairs.append((model_name, estimator))

    # Pipeline setup
    pipe = ModelSelectionForSimulation(dg, use_thresh=True, treated_thresh=0, has_pretuned=True)
    res_dir = f"simulation/results/cv/{rho}-{var_noise}"
    pipe.set_res_dir(res_dir)

    train_size_grid = [250 * (2**i) for i in range(13)]
    pipe.load_pretuned_data(pretune_file)

    # Run CV for all models at each training size.
    pipe.single_simulation(
        est_pairs, 
        exp_size=2000000,
        test_size=50000, 
        train_size_grid=train_size_grid,
        seed=seed,
        n_jobs=4
    )
