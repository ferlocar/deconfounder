# %%
from dataclasses import dataclass
from typing import Optional
import numpy as np
from copy import copy
import random
from multiprocessing import Pool
from deconfound_auuc import DeconfoundAUUC
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.tree import DecisionTreeRegressor


class DeconfoundStump(DecisionTreeRegressor):
    
    def __init__(self, n_levels, max_buckets, **kwargs):

        self.n_levels = n_levels
        self.max_buckets = max_buckets

        super().__init__(max_depth=1, **kwargs)

    def fit(self, X, treated, y, scores):

        X, treated, y, scores = np.array(X), np.array(treated), np.array(y), np.array(scores)
        n_samples = X.shape[0]
        order = np.argsort(scores, kind='mergesort')[::-1]
        X, treated, y, scores = X[order], treated[order], y[order], scores[order]

        n_levels = self.n_levels
        max_buckets = self.max_buckets
        if isinstance(max_buckets, int):
            n_buckets = (min(max_buckets, n_samples) // n_levels) * n_levels
        else:
            n_buckets = (int(max_buckets * n_samples) // n_levels) * n_levels

        # Divide into levels and buckets
        
        sample_ids = np.arange(n_samples)
        bkt_ids = binning(sample_ids, n_buckets)
        level_ids = binning(bkt_ids, n_levels)

        self.n_buckets = n_buckets

        # Set criterion
        self.criterion = DeconfoundAUUC(n_samples, n_levels, n_buckets)

        treated = treated.astype(np.uint8)
        level_ids = level_ids.astype(np.int32)
        bkt_ids = bkt_ids.astype(np.int32)
        self.criterion.set_sample_parameters(treated, scores, level_ids, bkt_ids)

        DecisionTreeRegressor.fit(self, X, y)

        return self
        
       
    def predict(self, X):
        return -DecisionTreeRegressor.predict(self, X)


class DeconfoundRanker:
    '''
    An ensemble of deconfounder trees with at most 1 split.
    '''

    def __init__(
        self, 
        n_trees=1, 
        n_levels=10, 
        max_buckets=1.0,
        subsample=1.0,
        **kwargs
    ):

        self.n_trees = n_trees 
        self.n_levels = n_levels
        self.max_buckets = max_buckets
        self.subsample = subsample
        self.kwargs = kwargs
    
    def fit(self, X, y):

        n_levels = self.n_levels
        max_buckets = self.max_buckets

        X, y = np.array(X), np.array(y)
        treated, y_true, pred_scores = y[:, 0], y[:, 1], y[:, 2]

        n_samples = X.shape[0]

        self.trees = []

        for k in range(self.n_trees):

            sample_mask = create_mask(n_samples, p=self.subsample)

            # Build a deconfounder tree based on the updated scores 
            stump = DeconfoundStump(n_levels, max_buckets, **self.kwargs)
            stump.fit(X[sample_mask], treated[sample_mask], y_true[sample_mask], pred_scores[sample_mask])

            if stump.get_depth() >= 1:
                # Update scores
                pred_scores -= stump.predict(X)
                self.trees.append(stump)
            else:
                break
         
    def predict(self, X):

        predictions = np.zeros(X.shape[0])

        for tree in self.trees:
            predictions += tree.predict(X)

        return predictions

            
def create_mask(n, p):
    mask = np.random.choice([False, True], size=n, p=[1-p, p])
    return mask

def binning(arr, n_bins):

    quantiles = np.linspace(0, 100, n_bins+1)
    bin_edges = np.asarray(np.percentile(arr, quantiles))
    binned_arr = np.searchsorted(bin_edges[1:-1], arr, side='right')
    return binned_arr
