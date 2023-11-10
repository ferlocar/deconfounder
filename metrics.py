import numpy as np
from sklearn.utils.extmath import stable_cumsum

def mean_squared_error(cate, cate_pred):

    return np.mean((cate - cate_pred) ** 2)

def causal_impact(cate, decisions):

    if np.all(decisions==0):
        return 0
    return cate[decisions].mean() * decisions.mean()

def uplift_curve(cate, scores, n_bins=200):

    n_samples = len(cate)

    cate, scores = np.array(cate), np.array(scores)

    order = np.argsort(scores, kind="mergesort")[::-1]
    cate, scores = cate[order], scores[order]

    threshold_indices = (np.arange(1, n_bins) / n_bins * n_samples).astype('int')
    threshold_indices = np.r_[threshold_indices, n_samples-1]

    num_all = threshold_indices + 1

    curve_values = stable_cumsum(cate)[threshold_indices]

    if num_all.size == 0 or curve_values[0] != 0 or num_all[0] != 0:
        # Add an extra threshold position if necessary
        # to make sure that the curve starts at (0, 0)
        num_all = np.r_[0, num_all]
        curve_values = np.r_[0, curve_values]

    return num_all, curve_values

def auuc_score(cate, scores, n_bins=200):

    n_samples = cate.shape[0]

    num_all, curve_values = uplift_curve(cate, scores, n_bins)
    auuc = np.sum(curve_values / n_samples)
    return auuc
