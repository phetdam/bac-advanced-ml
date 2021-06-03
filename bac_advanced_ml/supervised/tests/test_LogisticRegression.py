__doc__ = "Unit tests for the LogisticRegression class."

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression as _LogisticRegression

# pylint: disable=relative-beyond-top-level
from .. import LogisticRegression


@pytest.mark.parametrize("C", [0.1, 1., 5.])
def test_res(blob_bin, C):
    """Check LogisticRegression is equivalent to sklearn for two-class case.

    Checks the fit, predict, and score methods, checking solution and accuracy.

    Parameters
    ----------
    blob_bin : tuple
        pytest fixture. See conftest.py.
    C : float
        Inverse regularization parameter for the LogisticRegression class.
    """
    # unpack data from fixture
    X_train, X_test, y_train, y_test = blob_bin
    # hyperparameters to fix (in case defaults change + to use value of C)
    shared_params = dict(tol=1e-4, C=C, max_iter=100)
    # fit scikit-learn model and our model
    _lc = _LogisticRegression(**shared_params).fit(X_train, y_train)
    lc = LogisticRegression(**shared_params).fit(X_train, y_train)
    # check that coefficients and intercepts are close. scikit-learn's coef_
    # vector has extra dimension and has intercept_ as an array.
    np.testing.assert_allclose(_lc.coef_.ravel(), lc.coef_)
    np.testing.assert_allclose(_lc.intercept_[0], lc.intercept_)
    # check that predictions are close
    np.testing.assert_allclose(_lc.predict(X_test), lc.predict(X_test))
    # accuracy should be the same
    np.testing.assert_allclose(
        _lc.score(X_test, y_test), lc.score(X_test, y_test)
    )