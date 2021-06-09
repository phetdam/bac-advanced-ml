__doc__ = """Week 8 exercise: implementation of primal and dual linear SVM.

Unit tests may take around a minute to finish running, which belies the fact
that training SVMs can be quite time consuming.

.. codeauthor:: Derek Huang <djh458@stern.nyu.edu>
"""

import numpy as np
import pytest
from scipy.optimize import Bounds, LinearConstraint, minimize
from scipy.sparse import coo_matrix
from sklearn.base import BaseEstimator
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.utils import check_array, check_X_y
from sklearn.svm import LinearSVC as _LinearSVC


class LinearSVC(BaseEstimator):
    """Linear support vector classifier for binary classification.

    Only provides hinge loss and L2 norm regularization. Either the primal or
    dual formulation of the problem can be solved. Both primal and dual
    problems are solved using scipy's trust-constr implementation.

    Parameters
    ----------
    dual : bool, default=True
        True to use dual formulation, False for primal formulation.
    tol : float, default=1e-8
        Tolerance for solver. Passed to scipy.optimize.minimize trust-constr
        solver method's gtol option. The subgradient descent method terminates
        when the squared norm of the gradient / n_features is less than tol.
        Squaring the norm loosens the convergence criteria.
    C : float, default=1.
        Inverse regularization parameter/maximum Lagrange multiplier value in
        the primal/dual linear SVC formulation.
    max_iter : int, default=1000
        Maximum number of trust-constr iterations.

    Attributes
    ----------
    classes_ : numpy.ndarray
        Array of class labels known to the classifier, shape (2,)
    coef_ : numpy.ndarray
        Coefficients for the linear SVC, shape (n_features,)
    intercept_ : float
        Intercept (bias) term for the linear SVC

    Methods
    -------
    fit(X, y)
        Compute parameters to best fit the model to X, y.
    predict(X)
        Return predictions on given data X.
    score(X, y)
        Return the accuracy of the predictions given true labels y.
    """
    def __init__(self, dual=True, tol=1e-8, C=1., max_iter=1000):
        if not isinstance(dual, (bool, np.bool_)):
            raise TypeError("dual must be bool")
        if tol <= 0:
            raise ValueError("tol must be positive")
        if C <= 0:
            raise ValueError("C must be positive")
        if max_iter < 1:
            raise ValueError("max_iter must be positive")
        # assign attributes
        self.dual = dual
        self.tol = tol
        self.C = C
        self.max_iter = max_iter

    def fit(self, X, y):
        """Compute parameters to best fit the model to X, y.

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
        # get n_features
        # pylint: disable=unused-variable
        n_samples, n_features = X.shape
        # pylint: enable=unused-variable
        # get number of unique labels in y. if labels.size != 2, raise error
        labels = np.unique(y)
        if labels.size != 2:
            raise ValueError(
                "cannot fit coefficients on non-binary classification tasks"
            )
        # get mask of +1, -1. note that if we reverse these label assignments,
        # we actually get weights of the same magnitude but opposite sign to
        # those computed by scikit-learn's implementation.
        y_mask = np.empty(y.shape)
        y_mask[y == labels[0]] = -1
        y_mask[y == labels[1]] = 1
        ## solve for coefficients and intercept ##
        # if solving dual problem (with trust-constr)
        if self.dual:
            # compute hessian matrix, define objective, gradient, hessian,
            # define bounds on lagrange multipliers and linear constraint

            ###########################
            ### your code goes here ###
            ###########################

            # solve for lagrange multipliers using trust-constr and compute
            # primal weights and intercept from dual variables. save primal
            # weights to weights and intercept to intercept.

            ##############################
            pass # your code goes here ###
            ##############################

        # else solving primal problem
        else:
            # build hessian matrix (optionally using sparse coo_matrix type)

            ###########################
            ### your code goes here ###
            ###########################

            # functions for objective, gradient, hessian

            ###########################
            ### your code goes here ###
            ###########################

            # build LinearConstraint matrix (optionally using coo_matrix) and
            # define new LinearConstraint to pass to scipy.optimize.minimize

            ###########################
            ### your code goes here ###
            ###########################

            # bounds on variables. note n_features + 1 coefficients and
            # intercept are unbounded while slack variables must be >= 0

            ###########################
            ### your code goes here ###
            ###########################

            # solve for coefficients, intercept, slack using trust-constr.
            # save coefficients to weights and intercept to intercept.

            ##############################
            pass # your code goes here ###
            ##############################

        # set attributes. self.coef_ must be set to variable containing weights
        # and self.intercept_ must be variable containing intercept.
        self.classes_ = labels

        ###########################
        ### your code goes here ###
        ###########################

        # return self to allow for method chaining
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
            Predicted labels, shape (n_samples,). These are the same labels as
            the labels passed during fitting and stored in self.classes_.
        """
        if not hasattr(self, "coef_") or not hasattr(self, "intercept_"):
            raise RuntimeError("cannot predict with unfitted model")
        # validate input matrix
        X = check_array(X)
        # pylint: disable=no-member
        if X.shape[1] != self.coef_.shape[0]:
            raise ValueError("n_features must match length of coef_ vector")
        # return predictions. negative class is class 0, positive is class 1
        return np.where(
            X @ self.coef_ + self.intercept_ > 0,
            self.classes_[1], self.classes_[0]
        )
        # pylint: enable=no-member

    def score(self, X, y):
        """Return the accuracy of the predictions given true labels y.

        Parameters
        ----------
        X : numpy.ndarray
            Input matrix shape (n_samples, n_features)
        y : numpy.ndarray
            Response vector shape (n_samples,)

        Returns
        -------
        float
            Fraction of examples predicted correctly
        """
        # get predictions (includes input checks)
        y_pred = self.predict(X)
        # return the accuracy (fraction of correct predictions)
        return (y == y_pred).mean()


@pytest.fixture(scope="session")
def blob_bin():
    """Generated two-class blob classification problem with train/test split.

    Uses sklearn.datasets.make_blobs to make isotropic standard multivariate
    Gaussian blobs in 10-dimensional Euclidean space with 2 random centers. The
    returned data has 600 samples, 10 features. Features are informative by
    construction so we don't use sklearn.datasets.make_classification.

    Returns
    -------
    X_train : numpy.ndarray
        Training input data, shape (480, 10)
    X_test : numpy.ndarray
        Test/validation input data, shape (120, 10)
    y_train : numpy.ndarray
        Training class labels, shape (480,)
    y_test : numpy.ndarray
        Test/validation class labels, shape (120,)
    """
    # PRNG seed
    _seed = 7
    # generate noisy classification problem using isotropic Gaussian blobs
    # pylint: disable=unbalanced-tuple-unpacking
    X, y = make_blobs(
        n_samples=600, n_features=10, centers=2, random_state=_seed
    )
    # pylint: enable=unbalanced-tuple-unpacking
    # split the data with train_test_split and return
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=_seed
    )
    # return split data, coef, and bias
    return X_train, X_test, y_train, y_test


@pytest.mark.parametrize("C", [0.1, 1., 5.])
def test_res_dual(blob_bin, C):
    """Check LinearSVC is equivalent to sklearn for two-class case, dual=True.

    Parameters
    ----------
    blob_bin : tuple
        pytest fixture. See blob_bin.
    C : float
        Inverse regularization parameter for the LinearSVC.
    """
    # PRNG seed
    _seed = 7
    # unpack data from fixture
    X_train, X_test, y_train, y_test = blob_bin
    # named parameters for scikit-learn linear SVM implementation. note sklearn
    # penalizes intercept, so we have to pass large intercept_scaling value to
    # reduce the regularization. furthermore, the liblinear implementation uses
    # PRNG when performing cyclic coordinate descent on dual, so we fix seed.
    sklearn_params = dict(
        dual=True, loss="hinge", C=C,
        intercept_scaling=100, random_state=_seed
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


@pytest.mark.parametrize("C", [0.1, 1., 5.])
def test_res_primal(blob_bin, C):
    """Check LinearSVC is equivalent to sklearn for two-class case, dual=False.

    Parameters
    ----------
    blob_bin : tuple
        pytest fixture. See blob_bin.
    C : float
        Inverse regularization parameter for the LinearSVC.
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
    svc = LinearSVC(dual=False, C=C).fit(X_train, y_train)
    # check that predictions are close. we don't check the actual coef_ and
    # intercept_ properties because the solvers give different results.
    np.testing.assert_allclose(_svc.predict(X_test), svc.predict(X_test))
    # accuracy should be the same
    np.testing.assert_allclose(
        _svc.score(X_test, y_test), svc.score(X_test, y_test)
    )