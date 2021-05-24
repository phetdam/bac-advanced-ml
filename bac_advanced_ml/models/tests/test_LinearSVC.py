__doc__ = "Unit tests for the LogisticClassifier class."

import numpy as np
import pytest
from sklearn.svm import LinearSVC as _LinearSVC

# pylint: disable=relative-beyond-top-level
from ..supervised import LinearSVC


@pytest.mark.parametrize("C", [0.1, 1., 5.])
def test_res_dual(blob_bin, global_seed, C):
    """Check LinearSVC is equivalent to sklearn for two-class case, dual=True.

    Checks the fit, predict, and score methods, checking solution and accuracy.

    Parameters
    ----------
    blob_bin : tuple
        pytest fixture. See conftest.py.
    global_seed : int
        pytest fixture. See conftest.py.
    C : float
        Inverse regularization parameter for the LinearSVC.
    """
    # unpack data from fixture
    X_train, X_test, y_train, y_test = blob_bin
    # named parameters for scikit-learn linear SVM implementation. note sklearn
    # penalizes intercept, so we have to pass large intercept_scaling value to
    # reduce the regularization. furthermore, the liblinear implementation uses
    # PRNG when performing cyclic coordinate descent on dual, so we fix seed.
    sklearn_params = dict(
        dual=True, loss="hinge", C=C,
        intercept_scaling=100, random_state=global_seed
    )
    # fit scikit-learn model and our model
    _svc = _LinearSVC(**sklearn_params).fit(X_train, y_train)
    svc = LinearSVC(C=C).fit(X_train, y_train)
    # check that predictions are close. we don't check the actual coef_ and
    # intercept_ properties because the solvers give different results.
    np.testing.assert_allclose(_svc.predict(X_test), svc.predict(X_test))
    # accuracy should be the same
    np.testing.assert_allclose(
        _svc.score(X_test, y_test), svc.score(X_test, y_test)
    )


@pytest.mark.parametrize("solver", ["trust-constr"])
@pytest.mark.parametrize("C", [0.1, 1., 5.])
def test_res_primal(blob_bin, global_seed, C, solver):
    """Check LinearSVC is equivalent to sklearn for two-class case, dual=False.

    Checks the fit, predict, and score methods, checking solution and accuracy.

    Parameters
    ----------
    blob_bin : tuple
        pytest fixture. See conftest.py.
    global_seed : int
        pytest fixture. See conftest.py.
    C : float
        Inverse regularization parameter for the LinearSVC.
    solver : str
        Method uses to fit the linear SVM.
    """
    # unpack data from fixture
    X_train, X_test, y_train, y_test = blob_bin
    # named parameters for scikit-learn linear SVM implementation. note sklearn
    # penalizes intercept, so we have to pass large intercept_scaling value to
    # reduce the regularization. when solving primal, no PRNG is needed since
    # a Newton-style trust region algorithm is used. forced to use squared
    # hinge loss since liblinear primal formulation use squared hinge.
    sklearn_params = dict(dual=False, C=C, intercept_scaling=100)
    # fit scikit-learn model and our model
    _svc = _LinearSVC(**sklearn_params).fit(X_train, y_train)
    svc = LinearSVC(solver=solver, dual=False, C=C).fit(X_train, y_train)
    # check that predictions are close. we don't check the actual coef_ and
    # intercept_ properties because the solvers give different results.
    np.testing.assert_allclose(_svc.predict(X_test), svc.predict(X_test))
    # accuracy should be the same
    np.testing.assert_allclose(
        _svc.score(X_test, y_test), svc.score(X_test, y_test)
    )