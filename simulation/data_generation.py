"""
The data-generating process is characterized by a correlation of rho between CATE and E[Y^0|X].
"""
# %%
import os
import json
import numpy as np
import pandas as pd
from sklearn.utils import check_random_state
from sklearn.preprocessing import StandardScaler
import inspect


def standardize(x):
    return (x - np.mean(x)) / np.std(x)


class DataGenerator:
    """
    Parameters:
    - n_features: Number of features.
    - y0_mean: Mean of baseline outcome, E[Y^0].
    - ate: Average treatment effect.
    - var_cay0: Variance of conditional baseline outcomes, Var(E[Y^0|X]).
    - var_ratio: Var(CATE) / Var(CAY0).
    - rho: The correlation coefficient between E[Y^0|X] and CATE.
    - var_noise: Variance of outcome noise.
    - feature_map: Add interaction terms on how X determines CAY0 or CATE.
                    "identity" uses the original covariates.
    - nonlinearity: A nonlinear function applied to CAY0 or CATE.
    """

    def __init__(
        self,
        n_features=20, 
        y0_mean=0,
        ate=0.1,
        var_cay0=1,
        var_ratio=0.1,
        rho=1.0,
        var_noise=1,
        feature_map='identity',
        nonlinearity=None,
    ):
        self.n_features = n_features
        self.y0_mean = y0_mean
        self.ate = ate
        self.var_cay0 = var_cay0
        self.var_ratio = var_ratio
        self.rho = rho
        self.var_noise = var_noise

        self.feature_map = feature_map
        self.nonlinearity = nonlinearity



    def reset_coef(self, random_state=None):
        """
        Generate weights of features for base functions
        """
        n_features = self.n_features
        random_state = check_random_state(random_state)

        # Generate two random vectors for feature weights
        weights = random_state.normal(0, 1, size=(n_features, 2))

        self.weights = weights


    
    def generate_latent_factors(self, X):
        """
        Generate latent factors by combining random weights with X.
        These latent vectors will be used to generate both CATE and CAY0.
        """

        feature_map = self.feature_map
        nonlinearity = self.nonlinearity

        if feature_map == "identity":
            Phi = X
        else:
            raise ValueError("feature_map must be identity")

        # Standardize Phi
        Phi = StandardScaler().fit_transform(Phi)

        # Latent vectors
        U = np.dot(Phi, self.weights)

        if nonlinearity is not None:
            U = nonlinearity(U)

        return U


    def conditional_expectations_ortho(self, U):
        """
        Assumes one vector, \mu(X), is already defined and fixed. 
        Creates an orthogonal vector h_ortho from an auxiliary one h.
        tau is explicitly constructed based on mu and h_ortho.
        """

        rho = self.rho
        y0_mean = self.y0_mean
        ate = self.ate
        var_cay0 = self.var_cay0
        var_ratio = self.var_ratio

        # Base functions
        f = U[:, 0]
        h = U[:, 1]

        # Orthonogonalize h against f.
        beta = np.cov(f, h)[0, 1]/np.var(f)
        h_o = h -  beta * f

        # Standardize f and h_o
        f_std = standardize(f)
        h_o_std = standardize(h_o)

        cate_std = rho * f_std + np.sqrt(1-rho**2) * h_o_std

        # Get cay0 and cate
        cay0 = y0_mean + np.sqrt(var_cay0) * f_std
        cate = ate + np.sqrt(var_ratio * var_cay0) * cate_std

        return cay0, cate
    

    def generate_data(self, n, random_state=None):

        n_features = self.n_features
        var_noise = self.var_noise

        random_state = check_random_state(random_state)

        # covariates
        X = random_state.binomial(n=1, p=0.5, size=(n, n_features))

        # Latent factors
        U = self.generate_latent_factors(X)

        cay0, cate = self.conditional_expectations_ortho(U)

        # Treatment
        t = random_state.binomial(1, 0.5, size=n)

        # Observed outcome
        noise = random_state.normal(loc=0, scale=np.sqrt(var_noise), size=n)
        y = cay0 + cate * t + noise

        # Dataframe
        features = [f"x{k}" for k in range(n_features)]

        data = pd.DataFrame(data=np.c_[X, t, y, cay0, cate],
                columns=features+['t', 'y', 'bs', 'cate'])

        return data


    def save(self, res_dir):
        """
        Save DGP parameters.
        """

        if not os.path.exists(res_dir):
            os.makedirs(res_dir)
    
        fname = os.path.join(res_dir, "dgp_params.json")

        init_params = inspect.signature(self.__init__).parameters
        attr_names = list(init_params.keys()) 
        attr_dict = {}
        for attr_name in attr_names:
            attr_value = getattr(self, attr_name)
            if isinstance(attr_value, np.ndarray):
                attr_value = attr_value.tolist()  
            attr_dict[attr_name] = attr_value
        with open(fname, "w") as f:
            json.dump(attr_dict, f)



# %%
