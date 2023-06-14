# %%
import warnings
import numpy as np

from sklearn.metrics._ranking import auc
from sklearn.utils import check_consistent_length
from sklift.utils import check_is_binary

from sklift.metrics import qini_auc_score

def evaluate(metric, y_true, priority_score, treatment, k=0.1, percentile=0.9):
    if metric=="TOC_10%":
        return TOC_at_k(y_true, priority_score, treatment, k)
    elif metric=="AUTOC":
        return RATE(y_true, priority_score, treatment, method="AUTOC")
    elif metric=="QINI":
        return RATE(y_true, priority_score, treatment, method="QINI")
    elif metric=="AUPEC":
        return AUPEC(y_true, priority_score, treatment, percentile)
    else:
        raise ValueError(f"metric={metric} should be 'TOC_10%', 'AUTOC', 'QINI' or 'AUPEC'")

def ATE_at_k(y_true, priority_score, treatment, k=0.3):

    y_true, priority_score, treatment = np.array(y_true), np.array(priority_score), np.array(treatment)

    n_samples = len(y_true)
    order = np.argsort(priority_score, kind='mergesort')[::-1]

    k_type = np.asarray(k).dtype.kind

    if (k_type == 'i' and (k >= n_samples or k <= 0)
            or k_type == 'f' and (k <= 0 or k >= 1)):
        raise ValueError(f'k={k} should be either positive and smaller'
                         f' than the number of samples {n_samples} or a float in the '
                         f'(0, 1) range')

    if k_type not in ('i', 'f'):
        raise ValueError(f'Invalid value for k: {k_type}')

    if k_type == 'f':
        n_size = int(n_samples * k)
    else:
        n_size = k

    # ToDo: _checker_ there are observations among two groups among first k
    y_u_k = y_true[order][:n_size][treatment[order][:n_size] == 0].mean()
    y_t_k = y_true[order][:n_size][treatment[order][:n_size] == 1].mean()

    return y_t_k - y_u_k

def TOC_at_k(y_true, priority_score, treatment, k):
    y_true, priority_score, treatment = np.array(y_true), np.array(priority_score), np.array(treatment)

    ATE_k = ATE_at_k(y_true, priority_score, treatment, k)
    ATE = y_true[treatment==1].mean() - y_true[treatment==0].mean()

    return ATE_k-ATE

def TOC_curve(y_true, priority_score, treatment, subtract_ate_all=True):

    n_samples = len(y_true)

    y_true, priority_score, treatment = np.array(y_true), np.array(priority_score), np.array(treatment)

    ATE = y_true[treatment==1].mean() - y_true[treatment==0].mean()

    order = np.argsort(priority_score, kind="mergesort")[::-1]
    y_true, priority_score, treatment = y_true[order], priority_score[order], treatment[order]

    y_true_u, y_true_t = y_true.copy(), y_true.copy()

    y_true_u[treatment==1] = 0
    y_true_t[treatment==0] = 0

    n_t = stable_cumsum(treatment)
    y_t = stable_cumsum(y_true_t)

    n_all = np.arange(1, n_samples+1, dtype=np.float64)

    n_u = n_all - n_t
    y_u = stable_cumsum(y_true_u)

    curve_values = (np.divide(y_t, n_t, out=np.zeros_like(y_t), where=n_t != 0) -
                    np.divide(y_u, n_u, out=np.zeros_like(y_u), where=n_u != 0))
    
    if not subtract_ate_all:
        return n_all, curve_values
    
    curve_values -= ATE
    # if n_all.size == 0 or curve_values[0] != 0 or n_all[0] != 0:
    #     # Add an extra threshold position if necessary
    #     # to make sure that the curve starts at (0, 0)
    #     n_all = np.r_[0, n_all]
    #     curve_values = np.r_[0, curve_values]

    return n_all, curve_values


def uplift_curve(y_true, uplift, treatment):
    """Compute Uplift curve.

    For computing the area under the Uplift Curve, see :func:`.uplift_auc_score`.

    Args:
        y_true (1d array-like): Correct (true) binary target values.
        uplift (1d array-like): Predicted uplift, as returned by a model.
        treatment (1d array-like): Treatment labels.

    Returns:
        array (shape = [>2]), array (shape = [>2]): Points on a curve.

    References:
        Devriendt, F., Guns, T., & Verbeke, W. (2020). Learning to rank for uplift modeling. ArXiv, abs/2002.05897.
    """

    check_consistent_length(y_true, uplift, treatment)
    check_is_binary(treatment)
    # check_is_binary(y_true)

    y_true, uplift, treatment = np.array(y_true), np.array(uplift), np.array(treatment)

    desc_score_indices = np.argsort(uplift, kind="mergesort")[::-1]
    y_true, uplift, treatment = y_true[desc_score_indices], uplift[desc_score_indices], treatment[desc_score_indices]

    y_true_ctrl, y_true_trmnt = y_true.copy(), y_true.copy()

    y_true_ctrl[treatment == 1] = 0
    y_true_trmnt[treatment == 0] = 0

    distinct_value_indices = np.where(np.diff(uplift))[0]
    threshold_indices = np.r_[distinct_value_indices, uplift.size - 1]

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

def perfect_uplift_curve(y_true, treatment):
    """Compute the perfect (optimum) Uplift curve.

    This is a function, given points on a curve.  For computing the
    area under the Uplift Curve, see :func:`.uplift_auc_score`.

    Args:
        y_true (1d array-like): Correct (true) binary target values.
        treatment (1d array-like): Treatment labels.

    Returns:
        array (shape = [>2]), array (shape = [>2]): Points on a curve.

    See also:
        :func:`.uplift_curve`: Compute the area under the Qini curve.

        :func:`.uplift_auc_score`: Compute normalized Area Under the Uplift curve from prediction scores.

        :func:`.plot_uplift_curve`: Plot Uplift curves from predictions.
    """

    check_consistent_length(y_true, treatment)
    check_is_binary(treatment)
    # check_is_binary(y_true)
    y_true, treatment = np.array(y_true), np.array(treatment)

    cr_num = np.sum((y_true == 1) & (treatment == 0))  # Control Responders
    tn_num = np.sum((y_true == 0) & (treatment == 1))  # Treated Non-Responders

    # express an ideal uplift curve through y_true and treatment
    summand = y_true if cr_num > tn_num else treatment
    perfect_uplift = 2 * (y_true == treatment) + summand

    return uplift_curve(y_true, perfect_uplift, treatment)


def uplift_auc_score(y_true, uplift, treatment):
    """Compute normalized Area Under the Uplift Curve from prediction scores.

    By computing the area under the Uplift curve, the curve information is summarized in one number.
    For binary outcomes the ratio of the actual uplift gains curve above the diagonal to that of
    the optimum Uplift Curve.

    Args:
        y_true (1d array-like): Correct (true) binary target values.
        uplift (1d array-like): Predicted uplift, as returned by a model.
        treatment (1d array-like): Treatment labels.

    Returns:
        float: Area Under the Uplift Curve.

    See also:
        :func:`.uplift_curve`: Compute Uplift curve.

        :func:`.perfect_uplift_curve`: Compute the perfect (optimum) Uplift curve.

        :func:`.plot_uplift_curve`: Plot Uplift curves from predictions.

        :func:`.qini_auc_score`: Compute normalized Area Under the Qini Curve from prediction scores.
    """

    check_consistent_length(y_true, uplift, treatment)
    check_is_binary(treatment)
    # check_is_binary(y_true)
    y_true, uplift, treatment = np.array(y_true), np.array(uplift), np.array(treatment)

    x_actual, y_actual = uplift_curve(y_true, uplift, treatment)
    x_perfect, y_perfect = perfect_uplift_curve(y_true, treatment)
    x_baseline, y_baseline = np.array([0, x_perfect[-1]]), np.array([0, y_perfect[-1]])

    auc_score_baseline = auc(x_baseline, y_baseline)
    auc_score_perfect = auc(x_perfect, y_perfect) - auc_score_baseline
    auc_score_actual = auc(x_actual, y_actual) - auc_score_baseline

    return auc_score_actual / auc_score_perfect


def RATE(y_true, priority_score, treatment, method='AUTOC'):

    if method=="AUTOC":
        wtd_mean = lambda x, w: np.sum(x * w) / np.sum(w)
    elif method=="QINI":
        wtd_mean = lambda x, w: np.sum(np.cumsum(w) / np.sum(w) * w * x) / np.sum(w)
    else:
        raise ValueError(f'Method={method} should be either "AUTOC" or "QINI"')

    sample_weights = np.ones(len(y_true))
    toc_values = TOC_curve(y_true, priority_score, treatment)[1]
    
    return wtd_mean(toc_values, sample_weights)

def AUPEC(y_true, priority_score, treatment, percentile):

    n_samples = len(y_true)

    y_true, priority_score, treatment = np.array(y_true), np.array(priority_score), np.array(treatment)

    ATE = y_true[treatment==1].mean() - y_true[treatment==0].mean()

    ate_values = TOC_curve(y_true, priority_score, treatment, subtract_ate_all=False)[1]

    sample_weights = np.ones(n_samples)

    # only samples at this percentile will be assigned treatment
    mask = np.zeros(n_samples)
    T_size = n_samples - int(n_samples * percentile)
    mask[:T_size] = 1

    # ATE on samples determined for treatment
    ate_tsize = ate_values[T_size-1] if T_size>=1 else ate_values[-1]
    ate_mask = ate_values * mask + ate_tsize * (1 - mask)

    aupec_func = lambda x, w, m: ((np.cumsum(m*w) / np.sum(w) * w * x) - np.cumsum(w) / np.sum(w) * ATE).mean()

    return aupec_func(ate_mask, sample_weights, mask)


def stable_cumsum(arr, axis=None, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum.

    Warns if the final cumulative sum does not match the sum (up to the chosen
    tolerance).

    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat.
    axis : int, default=None
        Axis along which the cumulative sum is computed.
        The default (None) is to compute the cumsum over the flattened array.
    rtol : float, default=1e-05
        Relative tolerance, see ``np.allclose``.
    atol : float, default=1e-08
        Absolute tolerance, see ``np.allclose``.

    Returns
    -------
    out : ndarray
        Array with the cumulative sums along the chosen axis.
    """
    out = np.cumsum(arr, axis=axis, dtype=np.float64)
    expected = np.sum(arr, axis=axis, dtype=np.float64)
    if not np.all(
        np.isclose(
            out.take(-1, axis=axis), expected, rtol=rtol, atol=atol, equal_nan=True
        )
    ):
        warnings.warn(
            "cumsum was found to be unstable: "
            "its last element does not correspond to sum",
            RuntimeWarning,
        )
    return out
# %%
