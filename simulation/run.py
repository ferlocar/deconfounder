# %%
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

from time import time
import argparse
import numpy as np
from sklearn.base import clone

# Model import
from causal_tree.causal_residual_tree import CausalResidualForest
from econml.grf import CausalForest

# DGP
from simulation.data_generation import DataGenerator
from simulation.pipeline import SimulationPipeline

# %%
if __name__ == "__main__":

    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--rho', help='alignment between base scores and CATE', type=float, default=0.7)
    parser.add_argument('--var_noise', help='variance of outcome noise', type=float, default=1.6)
    parser.add_argument('--seed', help='random seed', type=int, default=1)
    args = parser.parse_args()

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

    # Model setup
    est_pairs = []
    msl_grid = [25*(2**i) for i in range(8)]
    estimators = {
        'CF': CausalForest(n_estimators=100, criterion='het', min_samples_split=2, honest=False),
        'CF-BS': CausalForest(n_estimators=100, criterion='het', min_samples_split=2, honest=False),
        'CRF': CausalResidualForest(n_estimators=100, min_samples_split=2)
    }
    for method in estimators.keys():
        for msl in msl_grid:
            model_name =f'{method}_{msl}'
            estimator = clone(estimators[method])
            estimator.set_params(min_samples_leaf=msl)
            est_pairs.append((model_name, estimator))


    # Pipeline setup
    pipe = SimulationPipeline(dg, use_thresh=True, treated_thresh=0)
    res_dir = f"simulation/results/oracle/{rho}-{var_noise}"
    pipe.set_res_dir(res_dir)

    # Run all models once for each training size.
    train_size_grid = [250 * (2**i) for i in range(9)]
    pipe.single_simulation(
        est_pairs, 
        exp_size=2000000,
        test_size=50000, 
        train_size_grid=train_size_grid,
        seed=seed,
        n_jobs=2
    )

    
