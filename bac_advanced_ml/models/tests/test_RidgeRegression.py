__doc__ = "Unit tests for the LinearRegression class."

import numpy as np
import pytest
from sklearn.linear_model import Ridge

# pylint: disable=relative-beyond-top-level
from ..supervised import RidgeRegression


def test_r2_matmul(linreg):
    """Check RidgeRegression R^2 is close to sklearn's when using matmul.

    Parameters
    ----------
    linreg : tuple
        pytest fixture. See conftest.py.
    """
    # get data and true parameters
    X_train, X_test, y_train, _, _, _ = linreg
    # fit sklearn model
    _lr = Ridge().fit(X_train, y_train)
    # fit our model and check that our solution is not far from sklearn's
    lr = RidgeRegression(solver = "matmul").fit(X_train, y_train)
    # pylint: disable=no-member
    np.testing.assert_allclose(_lr.coef_, lr.coef_)
    np.testing.assert_allclose(_lr.intercept_, lr.intercept_)
    # pylint: enable=no-member
    # check that predictions are close
    np.testing.assert_allclose(_lr.predict(X_test), lr.predict(X_test))


def test_r2_lsqr(linreg):
    """Check RidgeRegression R^2 is close to sklearn's when using lsqr.

    Parameters
    ----------
    linreg : tuple
        pytest fixture. See conftest.py.
    """
    # get data and true parameters
    X_train, X_test, y_train, _, _, _ = linreg
    # fit sklearn model
    _lr = Ridge().fit(X_train, y_train)
    # fit our model and check that our solution is not far from sklearn's
    lr = RidgeRegression(solver = "lsqr").fit(X_train, y_train)
    # pylint: disable=no-member
    np.testing.assert_allclose(_lr.coef_, lr.coef_)
    np.testing.assert_allclose(_lr.intercept_, lr.intercept_)
    # pylint: enable=no-member
    # check that predictions are close
    np.testing.assert_allclose(_lr.predict(X_test), lr.predict(X_test))