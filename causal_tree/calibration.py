"""
Calibrate base scores using difference in mean outcomes.
Reference: Leng, Yan, and Drew Dimmery, Calibration of Heterogeneous Treatment Effects in Randomized Experiments.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


class LinearEffectCalibration(LinearRegression):
    '''
    Logic:
        - Partition individuals into `m` bins based on base score percentiles.
        - For each bin, calculate the average base score (`avg_score_bin`) and average treatment effect (`ate_bin`).
        - Fit a linear regression model between `avg_score_bin` and `ate_bin`.
        - Prediction: `pred = coef * base_score + intercept`.

    Attributes:
        - n_bins: Number of bins. An additional parameter compared to standard linear regression.

    Note:
        - If `n_bins = 1`, then `coef = 1.0` and `intercept = ate - avg_score`.
    '''

    def __init__(
        self,
        *,
        n_bins=1,     
        fit_intercept=True,
        copy_X=True,
        n_jobs=None,
        positive=False,
    ):
        super().__init__(
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            n_jobs=n_jobs,
            positive=positive
        )

        self.n_bins = n_bins

    def fit(self, base_scores, t, y):

        n_bins = self.n_bins

        base_scores, t, y = np.array(base_scores), np.array(t), np.array(y)

        # If n_bins equals 1, set linear coefficient as 0 and intercept as ATE.
        if n_bins == 1:     
            mask = t==1
            self.coef_ = np.array([1.0])
            self.intercept_ = y[mask].mean() - y[~mask].mean()      # ATE
            return self

        n_samples = base_scores.shape[0]

        # Sort base_scores in ascending order
        order = np.argsort(base_scores)
        base_scores, t, y = base_scores[order], t[order], y[order]
        indice_bin = np.array_split(np.arange(n_samples), n_bins)

        # Average base_scores for each bin
        avg_score_bin = np.array([base_scores[ix].mean() for ix in indice_bin])

        # Number of treated and control units for each bin
        t_bin = [t[ix] for ix in indice_bin]
        num_t_bin = np.array([[np.sum(1-tb), np.sum(tb)] for tb in t_bin])   # (n_bins, 2)

        # Sum of Yt and Yc for each bin
        y_bin = [y[ix] for ix in indice_bin]
        sum_y_bin = np.array([[np.sum(yb[tb==0]), np.sum(yb[tb==1])] 
                              for yb, tb in zip(y_bin, t_bin)])   #(n_bins, 2)
        
        # Average (Yc and Yt) for each bin
        # If no treated or control unit in the bin, set value to be 0.
        avg_y_arr = sum_y_bin / np.maximum(1e-8, num_t_bin) 
        avg_yc_bin, avg_yt_bin = avg_y_arr[:, 0], avg_y_arr[:, 1]

        # ATE for each bin
        ate_bin = avg_yt_bin - avg_yc_bin

        # Fit ate_bin on avg_score_bin
        LinearRegression.fit(self, avg_score_bin.reshape(-1, 1), ate_bin)
        
        return self
    
    def predict(self, base_scores):
        return LinearRegression.predict(self, np.array(base_scores).reshape(-1, 1))


class TransformedLinearCalibration(LinearRegression):
    """
    Fit a linear regression model of transformed outcomes on base scores.
    """

    def fit(self, base_scores, t, y):
        
        base_scores = np.array(base_scores)
        t = np.array(t)
        y = np.array(y)

        p_t = np.mean(t)
        y_star = y * (t-p_t) / (p_t * (1-p_t))

        LinearRegression.fit(self, base_scores, y_star)

        return self
    
    def predict(self, base_scores):
        return LinearRegression.predict(self, np.array(base_scores).reshape(-1, 1))


