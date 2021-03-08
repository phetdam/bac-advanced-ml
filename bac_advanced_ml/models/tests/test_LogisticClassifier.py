__doc__ = "Unit tests for the LogisticClassifier class."

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression as _LogisticRegression

# pylint: disable=relative-beyond-top-level
from ..supervised import LogisticRegression

#@pytest.mark.skip(reason = "not yet implemented")
def test_res(blobcls):
    # unpack data from fixture
    X_train, X_test, y_train, y_test = blobcls
    # hyperparameters to fix (in case defaults change)
    shared_params = dict(tol = 1e-4, C = 1., max_iter = 100)
    # fit scikit-learn model and our model
    _lc = _LogisticRegression(**shared_params).fit(X_train, y_train)
    lc = LogisticRegression(**shared_params).fit(X_train, y_train)
    # check that coefficients and intercepts are close. scikit-learn's coef_
    # vector has extra dimension and has intercept_ as an array.
    np.testing.assert_allclose(_lc.coef_.ravel(), lc.coef_)
    np.testing.assert_allclose(_lc.intercept_[0], lc.intercept_)