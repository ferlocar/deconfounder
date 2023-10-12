import numpy as np
from sklearn.utils.extmath import stable_cumsum

def mean_squared_error(eff_true, eff_pred):

    return np.mean((eff_true - eff_pred) ** 2)

def causal_impact(eff_true, decisions):

    if np.all(decisions==0):
        return 0
    return eff_true[decisions].mean() * decisions.mean()

def uplift_curve(y_true, priority_score, treatment, n_bins=200):

    n_samples = len(y_true)

    y_true, priority_score, treatment = np.array(y_true), np.array(priority_score), np.array(treatment)

    order = np.argsort(priority_score, kind="mergesort")[::-1]
    y_true, priority_score, treatment = y_true[order], priority_score[order], treatment[order]

    y_true_ctrl, y_true_trmnt = y_true.copy(), y_true.copy()

    y_true_ctrl[treatment == 1] = 0
    y_true_trmnt[treatment == 0] = 0

    threshold_indices = (np.arange(1, n_bins) / n_bins * n_samples).astype('int')
    threshold_indices = np.r_[threshold_indices, n_samples-1]

    num_trmnt = stable_cumsum(treatment)[threshold_indices]
    y_trmnt = stable_cumsum(y_true_trmnt)[threshold_indices]

    num_all = threshold_indices + 1

    num_ctrl = num_all - num_trmnt
    y_ctrl = stable_cumsum(y_true_ctrl)[threshold_indices]

    curve_values = (np.divide(y_trmnt, num_trmnt, out=np.zeros_like(y_trmnt), where=num_trmnt != 0) -
                    np.divide(y_ctrl, num_ctrl, out=np.zeros_like(y_ctrl), where=num_ctrl != 0)) * num_all

    if num_all.size == 0 or curve_values[0] != 0 or num_all[0] != 0:
        # Add an extra threshold position if necessary
        # to make sure that the curve starts at (0, 0)
        num_all = np.r_[0, num_all]
        curve_values = np.r_[0, curve_values]

    return num_all, curve_values

def auuc_score(y_true, priority_score, treatment, n_bins=200):

    n_samples = y_true.shape[0]

    num_all, curve_values = uplift_curve(y_true, priority_score, treatment, n_bins)
    auuc = np.sum(curve_values / n_samples)
    return auuc
