# %%
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

import argparse
import numpy as np
import pandas as pd
from sklearn.base import clone

# Model import
from causal_tree.causal_residual_tree import CausalResidualForest
from econml.grf import CausalForest

from criteo.pipeline import CriteoPipeline

# %%
if __name__ == "__main__":
    """
    Each experiment (all models and all train size) needs to run 100 times.
    """
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', help='The Criteo data file.')
    parser.add_argument('--target', help='Base score (visit or f9)', default='f9')
    parser.add_argument('--seed', help='Random seed', type=int, default=1)
    args = parser.parse_args()

    data_file = args.data_file
    target = args.target
    seed = args.seed

    print(f"{target} will be used as base scores.")
    print(f"seed={seed}")

    # Model setups
    est_pairs = []
    msl_grid = [0.025, 0.05, 0.1, 0.2, 0.4]
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

    # Load data
    df = pd.read_csv(data_file)

    features = [f'f{i}' for i in range(12)]
    if target == 'f9':
        features.remove('f9')

    # Pipeline setup
    train_size_grid = list(range(10000, 250001, 30000))
    pipe = CriteoPipeline(df, features=features, treated_prop=0.1, target=target)
    res_dir = f"criteo/results/{target}"
    pipe.set_res_dir(res_dir)

    _ = pipe.run_experiments(
        est_pairs, 
        test_size=0.2, 
        train_size_grid=train_size_grid,
        obs_size=1000000, 
        seed=seed, 
        n_jobs=2
    )


    
