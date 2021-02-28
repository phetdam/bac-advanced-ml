__doc__ = """Reference implementations of supervised learning models.

Students are expected to have similar implementations.
"""

import numpy as np
import scipy.linalg
from sklearn.base import BaseEstimator


class LinearRegression(BaseEstimator):
    """Reference implementation for a linear regression model.

    "gelsd" uses scipy.linalg.lstsq with "gelsd" passed to lapack_driver.

    Parameters
    ----------
    solver : {"matmul", "gelsd",}

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
    _solvers = ("matmul", "gelsd")

    def __init__(self, solver = "matmul"):
        if solver not in self._solvers:
            raise ValueError(f"solver must be one of {self._solvers}")
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
        # add column of 1s to X
        X_aug = np.concatenate(
            (X, np.ones(X.shape[0]).reshape(-1, 1)), axis = 1
        )
        if self.solver == "matmul":
            # compute augmented weighted vector
            theta = np.linalg.inv(X_aug.T @ X_aug) @ X_aug.T @ y
        elif self.solver == "gelsd":
            # use scipy.linalg.lstsq to get augmented weights
            theta, _, _, _ = scipy.linalg.lstsq(
                X_aug, y, lapack_driver = "gelsd"
            )
        # set intercept and weights from augmented weight vector
        self.coef_ = theta[:-1]
        self.intercept_ = theta[-1]
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
        # get predictions
        y_pred = self.predict(X)
        # return R^2
        return (
            1 - np.power(y - y_pred, 2).sum() / np.power(y - y.mean(), 2).sum()
        )