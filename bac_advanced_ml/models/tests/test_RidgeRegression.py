__doc__ = "Unit tests for the LinearRegression class."

import numpy as np
import pytest
from sklearn.linear_model import Ridge

# pylint: disable=relative-beyond-top-level
from ..supervised import RidgeRegression


def test_r2_matmul(linreg):
    """Check RidgeRegression R^2 is close to sklearn's when using matmul.

    We aren't directly checking the solutions because I found that sklearn's
    results are slightly different from mine numerically.

    Parameters
    ----------
    linreg : tuple
        pytest fixture. See conftest.py.
    """
    # get data and true parameters
    X_train, X_test, y_train, y_test, _, _ = linreg
    # fit sklearn model
    _lr = Ridge().fit(X_train, y_train)
    # fit our model and check that our R^2 is not far from sklearn's
    lr = RidgeRegression(solver = "matmul").fit(X_train, y_train)
    # pylint: disable=no-member
    assert abs(_lr.score(X_test, y_test) - lr.score(X_test, y_test)) <= 1e-4
    # pylint: enable=no-member


def test_r2_lsqr(linreg):
    """Check RidgeRegression R^2 is close to sklearn's when using lsqr.

    Parameters
    ----------
    linreg : tuple
        pytest fixture. See conftest.py.
    """
    # get data and true parameters
    X_train, X_test, y_train, y_test, _, _ = linreg
    # fit sklearn model
    _lr = Ridge().fit(X_train, y_train)
    # fit our model and check that our R^2 is not far from sklearn's
    lr = RidgeRegression(solver = "lsqr").fit(X_train, y_train)
    # pylint: disable=no-member
    assert abs(_lr.score(X_test, y_test) - lr.score(X_test, y_test)) <= 1e-4
    # pylint: enable=no-member