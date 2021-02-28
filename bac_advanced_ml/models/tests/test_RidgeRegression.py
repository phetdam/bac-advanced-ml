__doc__ = "Unit tests for the LinearRegression class."

import numpy as np
import pytest
from sklearn.linear_model import Ridge

# pylint: disable=relative-beyond-top-level
from ..supervised import RidgeRegression


def test_r2(linreg):
    """Check unregularized RidgeRegression R^2 is close to sklearn's.

    We aren't directly checking the solutions because I found that sklearn's
    results are slightly different from mine numerically.

    Parameters
    ----------
    linreg : tuple
        pytest fixture. See conftest.py.
    """
    # get data and true parameters
    X_train, X_test, y_train, y_test, coef, bias = linreg
    # fit sklearn model
    _lr = Ridge().fit(X_train, y_train)
    # fit our model and check that our R^2 is not far from sklearn's. repeat
    # for both the matmul (exact) and lsqr solvers
    lr = RidgeRegression(solver = "matmul").fit(X_train, y_train)
    # pylint: disable=no-member
    assert abs(_lr.score(X_test, y_test) - lr.score(X_test, y_test)) <= 1e-4
    lr = RidgeRegression(solver = "lsqr").fit(X_train, y_train)
    assert abs(_lr.score(X_test, y_test) - lr.score(X_test, y_test)) <= 1e-4
    # pylint: enable=no-member