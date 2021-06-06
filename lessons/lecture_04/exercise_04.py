__doc__ = """Week 4 exercise: implementing regularized logistic regression.

.. codeauthor:: Derek Huang <djh458@stern.nyu.edu>
"""

import numpy as np
import pytest
import scipy.special
import scipy.optimize
from sklearn.base import BaseEstimator
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression as _LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import check_array, check_X_y


def _logistic_loss_grad(w, X, y, alpha):
    """Computes loss and gradient for a two-class logistic regression model.

    Parameters
    ----------
    w : numpy.ndarray
        Optimization variable. Contains the coefficients and the intercept, so
        it has shape (n_features + 1,).
    X : numpy.ndarray
        Input matrix, shape (n_samples, n_features)
    y : numpy.ndarray
        Response vector shape (n_samples,) where all elements are +/-1.
    alpha : float
        Regularization parameter. Equivalent to 1 / C.

    Returns
    -------
    loss : numpy.float64
        Objective value (total loss).
    grad : numpy.ndarray
        Gradient of the objective at w, shape (n_features + 1,).
    """
    ###########################
    ### your code goes here ###
    ###########################


class LogisticRegression(BaseEstimator):
    r"""Implementation for a logistic regression classifier.

    Only suitable for binary classification tasks. Solver used to minimize the
    objective is the scipy.optimize.minimize L-BFGS-B implementation. Objective
    is :math:`\ell^2`-regularized by default.

    Parameters
    ----------
    tol : float, default=1e-4
        Stopping tolerance to pass to solver.
    C : float, default=1.0
        Inverse regularization parameter. Increase to reduce regularization.
    max_iter : int, default=100
        Maximum number of iterations that L-BFGS-B is allowed to execute.

    Attributes
    ----------
    classes_ : numpy.ndarray
        Array of class labels known to the classifier, shape (n_classes,)
    coef_ : numpy.ndarray
        Model coefficients of the fitted model, shape (n_features,)
    intercept_ : float
        Intercept (bias) term for the model.

    Methods
    -------
    fit(X, y)
        Compute parameters to best fit the model to X, y.
    predict(X)
        Return predictions on given data X.
    score(X, y)
        Return the accuracy of the predictions given true labels y.
    """
    def __init__(self, tol=1e-4, C=1., max_iter=100):
        if tol <= 0:
            raise ValueError("tol must be positive")
        if C <= 0:
            raise ValueError("C must be positive")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")
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
        _, n_features = X.shape
        # get number of unique labels in y. if labels.size != 2, raise error
        labels = np.unique(y)
        if labels.size != 2:
            raise ValueError(
                "Cannot fit coefficients on non-binary classification tasks"
            )
        # get mask of +1, -1. note that if we reverse these label assignments,
        # we actually get weights of the same magnitude but opposite sign to
        # those computed by scikit-learn's implementation.
        y_mask = np.empty(y.shape)
        y_mask[y == labels[0]] = -1
        y_mask[y == labels[1]] = 1
        # solve for coefficients and intercept
        res = scipy.optimize.minimize(
            _logistic_loss_grad, np.zeros(n_features + 1),
            method="L-BFGS-B", jac=True, args=(X, y_mask, 1. / self.C),
            options=dict(gtol=self.tol, maxiter=self.max_iter)
        )
        weights = res.x
        # set attributes
        self.classes_ = labels
        self.coef_ = weights[:-1]
        self.intercept_ = weights[-1]
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
        if X.shape[1] != self.coef_.shape[0]:
            raise ValueError("n_features must match length of coef_ vector")

        ###########################
        ### your code goes here ###
        ###########################

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
        ###########################
        ### your code goes here ###
        ###########################


@pytest.fixture(scope="session")
def blob_bin():
    """Generated two-class blob classification problem with train/test split.

    Returns
    -------
    X_train : numpy.ndarray
    X_test : numpy.ndarray
    y_train : numpy.ndarray
    y_test : numpy.ndarray
    """
    # seed value for reproducible results
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