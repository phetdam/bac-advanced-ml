__doc__ = "Unit tests for the LinearRegression class."

import numpy as np
import pytest
from sklearn.linear_model import Ridge

# pylint: disable=relative-beyond-top-level
from ..supervised import RidgeRegression


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
    X_train, X_test, y_train, y_test, _, _ = linreg
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
        _lr.score(X_test, y_test), lr.score(X_test, y_test)
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
    X_train, X_test, y_train, y_test, _, _ = linreg
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
        _lr.score(X_test, y_test), lr.score(X_test, y_test)
    )
    # pylint: enable=no-member