"""
This file provides metrics for evaluating effect magnitude, ordering, and classification.
"""
# %%
import numpy as np
from sklearn.utils.extmath import stable_cumsum
import matplotlib.pyplot as plt
import numbers

EPSILON = 1e-9

# Number of bins for effect calibration (calculated via the Rice rule)
get_num_bins = lambda n: np.ceil((2 * n) ** (1/3)).astype(int)

## --------------- Assume true CATE and ITE are known. Apply to simulation study.-------------------------

def get_mse(true_values, predictions):
    return np.mean((true_values - predictions) ** 2)
    
def expected_policy_effect(effect, decisions):
    '''
    Expected effect for a given decision policy. 
    The individual reward equals its effect if decision equals 1, and 0 otherwise.
    '''
    return np.mean(effect * (1 * decisions))
    

def targted_ate_with_effect(eff, scores, n_bins=100):
    """
    Partition the data into n bins. Create targeted groups by sequentially selecting the first j bins, where j = 1, 2, ..., n. 
    Compute ATE for each targeted group.
    """
    eff = np.array(eff)
    scores = np.array(scores)

    n_bins = min(eff.shape[0], n_bins)

    # Sort data by descending scores.
    order = np.argsort(scores, kind='mergesort')[::-1]

    # Sum of effects at top-k
    eff_bin = np.array_split(eff[order], n_bins)
    eff_sum_bin = [np.sum(c) for c in eff_bin]
    eff_sum_topk = stable_cumsum(eff_sum_bin)

    # Number of units at top-k
    bin_size = np.array([len(c) for c in eff_bin])
    num_topk = stable_cumsum(bin_size)
    targeted_prop = num_topk / eff.shape[0]

    # ATE at top-k
    targeted_ate = eff_sum_topk / num_topk

    return targeted_prop, targeted_ate

def rate_with_effect(eff, scores, n_bins=100, metrics=['autoc', 'qini']):
    """
    Use true CATE to compute RATE metrics.
    """

    eff = np.array(eff)
    n = eff.shape[0]

    # score is a number 
    if isinstance(scores, (numbers.Integral, numbers.Real)):
        return 0, 0

    ate = np.mean(eff)
    targeted_prop, targeted_ate = targted_ate_with_effect(eff, scores, n_bins)
    toc_values = targeted_ate - ate
    for metric in metrics:
        if metric == 'autoc':
            autoc = np.sum(toc_values) / n_bins
        elif metric == 'qini':
            qini = np.sum(targeted_prop * toc_values) / n_bins
    
    return autoc, qini


### ------------ True CATE or ITE are unknown. Apply to empirical case.------------------------

def transformed_mse(t, y, scores):
    '''
    Mse squared difference between transformed outcomes and 
    predicted scores.
    '''
    # check_is_binary(treatment)
    p_t = np.mean(t)
    y_star = y * (t - p_t) / max(EPSILON, p_t * (1 - p_t))
    mse = np.mean((scores - y_star)**2)
    return mse

def subgroup_mse(t, y, scores, order, n_bins=None):
    """
    - Mean squared difference beteen average predicted effect per bin
    and actual ATE per bin.
    - We use the input order rather than the score-based sorting order.
    """
    t = np.array(t)
    y = np.array(y)

    if n_bins is None:
        n_bins = get_num_bins(t.shape[0])

    # Handle the case where scores is a number.
    if isinstance(scores, (numbers.Integral, numbers.Real)):
        scores = np.full(shape=t.shape[0], fill_value=scores)

    # Binning data
    t, y, scores = t[order], y[order], scores[order]

    y_bin = np.array_split(y, n_bins)
    t_bin = np.array_split(t, n_bins)
    scores_bin = np.array_split(scores, n_bins)

    # Subgroup ATE
    y_sum_bin = np.array([[np.sum(yb[tb==0]), np.sum(yb[tb==1])] 
                          for yb, tb in zip(y_bin, t_bin)])             # (n_bins, 2)
    t_num_bin = np.array([[np.sum(1-tb), np.sum(tb)] for tb in t_bin])
    y_mean_bin = y_sum_bin / np.maximum(EPSILON, t_num_bin)
    ate_bin = y_mean_bin[:, 1] - y_mean_bin[:, 0]

    # Subgroup average scores
    avg_score_bin = np.array([np.mean(sb) for sb in scores_bin])

    return np.mean((ate_bin-avg_score_bin) ** 2)


def expected_random_policy_outcome(t, y, treated_prop):
    """
    Expected potential outcome for a random policy.
    """
    n = t.shape[0]
    p_t = np.mean(t)
    avg_y1 = np.sum(y * t / p_t) / n 
    avg_y0 = np.sum(y * (1-t) / (1-p_t)) / n
    epo = treated_prop * avg_y1 + (1-treated_prop) * avg_y0
    return epo 

    
def expected_policy_outcome(t, y, decisions):
    """
    Compute expected outcome for a decision policy. 
    - If t = 1, the reward is y/p_t; if t = 0, the reward is y/(1-p_t).
    - compute the average reward for the individuals with the same treatment decision and assignment.
    """
    p_t = np.mean(t)
    reward = y / (t*p_t + (1-t)*(1-p_t))
    mean_reward = np.mean(reward * 1 * (t==decisions))
    return mean_reward


def get_targeted_ate(t, y, scores, n_bins=100):
    """
    ATE values for varied targeted proportions.
    """
    n_bins = min(t.shape[0], n_bins)
    
    t = np.array(t)
    y = np.array(y)
    scores = np.array(scores)

    # Sort data by descending scores.
    order = np.argsort(scores, kind='mergesort')[::-1]

    # Sum of Yt and Yc for each bin (n_bins, 2).
    # Number of treated and control units for each bin (n_bins, 2).
    y_bin = np.array_split(y[order], n_bins)
    t_bin = np.array_split(t[order], n_bins)
    y_sum_bin = np.array([[np.sum(yb[tb==0]), np.sum(yb[tb==1])] 
                          for yb, tb in zip(y_bin, t_bin)])                 
    t_num_bin = np.array([[np.sum(1-tb), np.sum(tb)] for tb in t_bin])      

    # Sum of Yt and Yc at varied top-k.
    # Number of treated and control units at varied top-k.
    y_sum_topk = stable_cumsum(y_sum_bin, axis=0)       
    t_num_topk = stable_cumsum(t_num_bin, axis=0)
    
    # Number of units at varied top-k
    num_topk = np.sum(t_num_topk, axis=1)           # (n_bins,)
    targeted_prop = num_topk / t.shape[0]

    # Mean of Yt and Yc at varied top-k (For safe divide).
    y_mean_topk = y_sum_topk / np.maximum(EPSILON, t_num_topk)
    targeted_ate = y_mean_topk[:, 1] - y_mean_topk[:, 0]

    return targeted_prop, targeted_ate

def get_rate(t, y, scores, n_bins=100, metrics=['autoc', 'qini']):
    """
    Compute RATE metrics using observed outcomes.
    """

    t = np.array(t)
    y = np.array(y)

    n = y.shape[0]

    # If score is a number, ATE at top-k would be the same as ATE. 
    if isinstance(scores, (numbers.Integral, numbers.Real)):
        return 0, 0
    
    targeted_prop, targeted_ate = get_targeted_ate(t, y, scores, n_bins)
    ate = y[t==1].mean() - y[t==0].mean()
    toc_values = targeted_ate - ate
    for metric in metrics:
        if metric == 'autoc':
            autoc = np.sum(toc_values) / n_bins
        elif metric == 'qini':
            qini = np.sum(targeted_prop * toc_values) / n_bins
    
    return autoc, qini


