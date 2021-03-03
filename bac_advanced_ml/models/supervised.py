__doc__ = """Reference implementations of supervised learning models.

Students are expected to have similar implementations.
"""

import math
import numpy as np
import scipy.sparse.linalg
from sklearn.base import BaseEstimator
from sklearn.utils import check_array, check_X_y


class RidgeRegression(BaseEstimator):
    """Reference implementation for a ridge linear regression model.

    Parameters
    ----------
    alpha : float, default=1.0
        Regularization parameter. Increasing it increases regularization.
    solver : {"matmul", "lsqr"}
        Solver to use to compute the ridge coefficient solution. "matmul" uses
        direct inversion of the penalized moment matrix while "lsqr" uses
        scipy.sparse.linalg.lsqr to solve the damped least squares problem.

    Attributes
    ----------
    coef_ : numpy.ndarray
        Model coefficients of the fitted model, shape (n_features,)
    intercept_ : float
        Intercept (bias) term for the model.

    Methods
    -------
    fit(X, y)
        Compute coefficients to best fit the model to X, y.
    predict(X)
        Return predictions on given data X.
    score(X, y)
        Return the :math:`R^2` of the predictions.
    """
    # allowable solvers
    _solvers = ("matmul", "lsqr")

    def __init__(self, alpha = 1., solver = "lsqr"):
        if alpha < 0:
            raise ValueError("alpha must be nonnegative")
        if solver not in self._solvers:
            raise ValueError(f"solver must be one of {self._solvers}")
        self.alpha = alpha
        self.solver = solver

    def fit(self, X, y):
        """Compute coefficients to best fit the model to X, y.

        Parameters
        ----------
        X : numpy.ndarray
            Input matrix shape (n_samples, n_features)
        y : numpy.ndarray
            Response vector shape (n_samples,)

        Returns
        -------
        self
        """
        # validate input
        X, y = check_X_y(X, y)
        # compute mean of X, y and get centered X matrix and y vector
        x_mean, y_mean = X.mean(axis = 0), y.mean()
        X_c, y_c = X - x_mean, y - y_mean
        # delegate coefficient computation to different solving methods
        if self.solver == "matmul":
            # compute coefficients using matrix inversion on centered problem
            self.coef_ = np.linalg.inv(
                X_c.T @ X_c + self.alpha * np.eye(X_c.shape[1])
            ) @ X_c.T @ y_c
        elif self.solver == "lsqr":
            # use scipy.sparse.linalg.lsqr to get augmented weights
            self.coef_ = scipy.sparse.linalg.lsqr(
                X_c, y_c, damp = math.sqrt(self.alpha)
            )[0]
        # compute intercept
        self.intercept_ = y.mean() - X.mean(axis = 0) @ self.coef_
        # returning self allows for method chaining
        return self

    def predict(self, X):
        """Return predictions on given data X.

        Parameters
        ----------
        X : numpy.ndarray
            Input matrix shape (n_samples, n_features)

        Returns
        -------
        numpy.ndarray
            Predictions, shape (n_samples,)
        """
        if not hasattr(self, "coef_") or not hasattr(self, "intercept_"):
            raise RuntimeError("cannot predict with unfitted model")
        # validate input matrix
        X = check_array(X)
        if X.shape[1] != self.coef_.shape[0]:
            raise ValueError("n_features must match length of coef_ vector")
        return X @ self.coef_ + self.intercept_

    def score(self, X, y):
        """Return the :math:`R^2` of the predictions.

        Parameters
        ----------
        X : numpy.ndarray
            Input matrix shape (n_samples, n_features)
        y : numpy.ndarray
            Response vector shape (n_samples,)

        Returns
        -------
        float
            :math:`R^2` coefficient of determination.
        """
        # get predictions (also validates input)
        y_pred = self.predict(X)
        # return R^2
        return (
            1 - np.power(y - y_pred, 2).sum() / np.power(y - y.mean(), 2).sum()
        )