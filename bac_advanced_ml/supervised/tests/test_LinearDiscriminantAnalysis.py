__doc__ = "Unit tests for the LinearDiscriminantAnalysis class."

import numpy as np
import pytest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as _LDA

# pylint: disable=relative-beyond-top-level
from .. import LinearDiscriminantAnalysis


@pytest.mark.parametrize("shrinkage", [None, 0.2, 0.7])
def test_res_binary(blob_bin, shrinkage):
    """Check LinearDiscriminantAnalysis equals sklearn for two-class case.

    Checks the fit, predict, and score methods, checking solution and accuracy.
    Also checks the different attributes set during the fitting process.

    Parameters
    ----------
    blob_bin : tuple
        pytest fixture. See conftest.py.
    shrinkage : float
        Shrinkage parameter to pass to LinearDiscriminantAnalysis
    """
    # unpack data from fixture
    X_train, X_test, y_train, y_test = blob_bin
    # fit scikit-learn model and our model
    _lc = _LDA(solver="lsqr", shrinkage=shrinkage).fit(X_train, y_train)
    lc = LinearDiscriminantAnalysis(shrinkage=shrinkage).fit(X_train, y_train)
    # check class labels, priors, means, covariance
    np.testing.assert_allclose(_lc.classes_, lc.classes_)
    np.testing.assert_allclose(_lc.priors_, lc.priors_)
    np.testing.assert_allclose(_lc.means_, lc.means_)
    np.testing.assert_allclose(_lc.covariance_, lc.covariance_)
    # check that coefficients and intercepts are close. in the two-class case,
    # sklearn has an extra dimension that can be dropped
    np.testing.assert_allclose(_lc.coef_.ravel(), lc.coef_)
    np.testing.assert_allclose(_lc.intercept_, lc.intercept_)
    # check that predictions are close
    np.testing.assert_allclose(_lc.predict(X_test), lc.predict(X_test))
    # accuracy should be the same
    np.testing.assert_allclose(
        _lc.score(X_test, y_test), lc.score(X_test, y_test)
    )


@pytest.mark.parametrize("shrinkage", [None, 0.2, 0.7])
def test_res_multi(blob_multi, shrinkage):
    """Check LinearDiscriminantAnalysis equals sklearn for multi-class case.

    Checks the fit, predict, and score methods, checking solution and accuracy.
    Also checks the different attributes set during the fitting process.

    Parameters
    ----------
    blob_bin : tuple
        pytest fixture. See conftest.py.
    shrinkage : float
        Shrinkage parameter to pass to LinearDiscriminantAnalysis
    """
    # unpack data from fixture
    X_train, X_test, y_train, y_test = blob_multi
    # fit scikit-learn model and our model
    _lc = _LDA(solver="lsqr", shrinkage=shrinkage).fit(X_train, y_train)
    lc = LinearDiscriminantAnalysis(shrinkage=shrinkage).fit(X_train, y_train)
    # check class labels, priors, means, covariance
    np.testing.assert_allclose(_lc.classes_, lc.classes_)
    np.testing.assert_allclose(_lc.priors_, lc.priors_)
    np.testing.assert_allclose(_lc.means_, lc.means_)
    np.testing.assert_allclose(_lc.covariance_, lc.covariance_)
    # check that coefficients and intercepts are close
    np.testing.assert_allclose(_lc.coef_, lc.coef_)
    np.testing.assert_allclose(_lc.intercept_, lc.intercept_)
    # check that predictions are close
    np.testing.assert_allclose(_lc.predict(X_test), lc.predict(X_test))
    # accuracy should be the same
    np.testing.assert_allclose(
        _lc.score(X_test, y_test), lc.score(X_test, y_test)
    )