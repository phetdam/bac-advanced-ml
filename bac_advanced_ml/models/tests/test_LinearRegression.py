__doc__ = "Unit tests for the LinearRegression class."

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression as _LinearRegression

# pylint: disable=relative-beyond-top-level
from ..supervised import LinearRegression


def test_sklearn_similarity(linreg):
    """Check that our LinearRegression model result is close to sklearn's

    Parameters
    ----------
    linreg : tuple
        pytest fixture. See conftest.py.
    """
    # get data and true parameters
    X_train, _, y_train, _, _, _ = linreg
    # fit sklearn model
    _lr = _LinearRegression().fit(X_train, y_train)
    # fit our model and check that our solution is not far from sklearn's
    lr = LinearRegression(solver = "matmul").fit(X_train, y_train)
    np.testing.assert_allclose(_lr.coef_, lr.coef_)
    np.testing.assert_allclose(_lr.intercept_, lr.intercept_)
    # repeat for gelsd solver
    lr = LinearRegression(solver = "gelsd").fit(X_train, y_train)
    np.testing.assert_allclose(_lr.coef_, lr.coef_)
    np.testing.assert_allclose(_lr.intercept_, lr.intercept_)