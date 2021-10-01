__doc__ = """Week 3 exercise: implementing ridge regression.

.. codeauthor:: Derek Huang <djh458@stern.nyu.edu>
"""

import math
import numpy as np
import pytest
import scipy.sparse.linalg
from sklearn.base import BaseEstimator
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.utils import check_array, check_X_y


class RidgeRegression(BaseEstimator):
    """Implementation for a ridge linear regression model.

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
        Return the :math:`R^2` of the predictions given true response y.
    """
    # allowable solvers
    _solvers = ("matmul", "lsqr")

    def __init__(self, alpha=1., solver="lsqr"):
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

        ###########################
        ### your code goes here ###
        ###########################

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
        # pylint: disable=no-member
        if X.shape[1] != self.coef_.shape[0]:
            raise ValueError("n_features must match length of coef_ vector")
        # pylint: enable=no-member

        ###########################
        ### your code goes here ###
        ###########################

    def score(self, X, y):
        """Return the :math:`R^2` of the predictions given true response y.

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
        ###########################    
        ### your code goes here ###
        ###########################


@pytest.fixture(scope="session")
def linreg():
    """Regression data returned by sklearn.datasets.make_regression.

    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test
    """
    # seed value for reproducible results
    _seed = 7
    # generate noisy regression problem
    # pylint: disable=unbalanced-tuple-unpacking
    X, y = make_regression(
        n_samples=600,
        n_features=10,
        n_informative=10,
        bias=7,
        noise=1,
        random_state=_seed
    )
    # pylint: enable=unbalanced-tuple-unpacking
    # split the data with train_test_split and return it
    return train_test_split(X, y, test_size=0.2, random_state=7)


@pytest.mark.parametrize("alpha", [0.1, 1., 5.])
def test_res_matmul(linreg, alpha):
    """Check RidgeRegression is equivalent to sklearn when using matmul.

    Checks the fit, predict, and score methods, checking solution and R^2.

    Parameters
    ----------
    linreg : tuple
        pytest fixture. See conftest.py.
    alpha : float
        Regularization parameter of the RidgeRegression class.
    """
    # get data and true parameters
    X_train, X_test, y_train, y_test = linreg
    # fit sklearn model
    _lr = Ridge(alpha=alpha).fit(X_train, y_train)
    # fit our model and check that our solution is not far from sklearn's
    lr = RidgeRegression(alpha=alpha, solver="matmul").fit(X_train, y_train)
    # pylint: disable=no-member
    np.testing.assert_allclose(_lr.coef_, lr.coef_)
    np.testing.assert_allclose(_lr.intercept_, lr.intercept_)
    # pylint: enable=no-member
    # check that predictions are close
    np.testing.assert_allclose(_lr.predict(X_test), lr.predict(X_test))
    # check that R^2 is close
    # pylint: disable=no-member
    np.testing.assert_allclose(
        _lr.score(X_test, y_test),
        lr.score(X_test, y_test)
    )
    # pylint: enable=no-member


@pytest.mark.parametrize("alpha", [0.1, 1., 5.])
def test_res_lsqr(linreg, alpha):
    """Check RidgeRegression is equivalent to sklearn when using lsqr.

    Checks the fit, predict, and score methods, checking solution and R^2.

    Parameters
    ----------
    linreg : tuple
        pytest fixture. See conftest.py.
    alpha : float
        Regularization parameter of the RidgeRegression class.
    """
    # get data and true parameters
    X_train, X_test, y_train, y_test = linreg
    # fit sklearn model
    _lr = Ridge(alpha=alpha).fit(X_train, y_train)
    # fit our model and check that our solution is not far from sklearn's
    lr = RidgeRegression(alpha=alpha, solver="lsqr").fit(X_train, y_train)
    # pylint: disable=no-member
    np.testing.assert_allclose(_lr.coef_, lr.coef_)
    np.testing.assert_allclose(_lr.intercept_, lr.intercept_)
    # pylint: enable=no-member
    # check that predictions are close (need lower rtol, default is 1e-7)
    np.testing.assert_allclose(
        _lr.predict(X_test), lr.predict(X_test), rtol=1e-6
    )
    # check that R^2 is close
    # pylint: disable=no-member
    np.testing.assert_allclose(
        _lr.score(X_test, y_test),
        lr.score(X_test, y_test)
    )
    # pylint: enable=no-member