__doc__ = """Support vector classification.

.. codeauthor:: Derek Huang <djh458@stern.nyu.edu>
"""

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize
from scipy.sparse import coo_matrix
from sklearn.base import BaseEstimator
from sklearn.utils import check_array, check_X_y


class LinearSVC(BaseEstimator):
    """Linear support vector classifier for binary classification.

    Only provides hinge loss and L2 norm regularization. Either the primal or
    dual formulation of the problem can be solved. Both primal and dual
    problems are solved using scipy's trust-constr implementation.

    .. note::

       This is a toy implementation and scales poorly when the number of
       examples is large. In particular, memory consumption using trust-constr
       will be O(n_samples ** 2) when dual=True due to the requirements of
       storing a dense dual Hessian. When dual=False, the memory usage is
       O(n_samples * (n_features + 1) + n_samples + n_features) since COO
       sparse matrices are used to store the primal Hessian and constraints.

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
        n_samples, n_features = X.shape
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
            # compute hessian matrix, which is the gram matrix multiplied
            # elementwise by outer product of the labels
            hess = (X @ X.T) * np.outer(y_mask, y_mask)
            # functions for objective, gradient, hessian
            dual_obj = lambda x: 0.5 * x @ hess @ x - x.sum()
            dual_grad = lambda x: hess @ x - np.ones(n_samples)
            dual_hess = lambda x: hess
            # bounds on lagrange multipliers and linear constraint
            alpha_bounds = Bounds(0, self.C)
            alpha_cons = LinearConstraint(y_mask, 0, 0)
            # solve for lagrange multipliers using trust-constr
            res = minimize(
                dual_obj, np.zeros(n_samples), method="trust-constr",
                jac=dual_grad, hess=dual_hess, bounds=alpha_bounds,
                constraints=alpha_cons, options=dict(gtol=self.tol)
            )
            # compute primal weights and intercept from dual variables. note
            # we only average y_mask - X @ weights for support vectors
            weights = ((y_mask * res.x).reshape(-1, 1) * X).sum(axis=0)
            # get mask of support vectors before computing intercept
            support_mask = np.logical_and(res.x > 0, res.x < self.C)
            intercept = (support_mask * (y_mask - X @ weights)).mean()
        # else solving primal problem
        else:
            # nonzero hessian elements
            hess_vals = np.ones(n_features)
            # row, columns indices for nonzero hessian elements. these are
            # the same since the hessian is diagonal
            hess_idx = np.arange(n_features)
            # build hessian matrix, using coo to save memory
            hess = coo_matrix(
                (hess_vals, (hess_idx, hess_idx)),
                shape=(n_samples + n_features + 1, n_samples + n_features + 1)
            )
            # functions for objective, gradient, hessian
            primal_obj = lambda x: (
                0.5 * np.power(x[:n_features], 2).sum() +
                self.C * x[n_features + 1:].sum()
            )
            primal_grad = lambda x: np.hstack(
                (x[:n_features], 0, self.C * np.ones(n_samples))
            )
            primal_hess = lambda x: hess
            # nonzero LinearConstraint matrix values. note for a 2D ndarray,
            # ravel has the effect of pasting its rows end-to-end.
            marg_vals = np.hstack(
                (
                    np.ravel(
                        y_mask.reshape(-1, 1) * np.hstack(
                            (X, np.ones(n_samples).reshape(-1, 1))
                        )
                    ),
                    np.ones(n_samples)
                )
            )
            # row indices for nonzero LinearConstraint matrix values
            marg_row_idx = np.hstack(
                (
                    np.hstack(
                        [np.full(n_features + 1, i) for i in range(n_samples)]
                    ),
                    np.arange(n_samples)
                )
            )
            # column indices for nonzero LinearConstraint matrix values
            marg_col_idx = np.hstack(
                (
                    np.hstack(
                        [np.arange(n_features + 1) for i in range(n_samples)]
                    ),
                    n_features + 1 +  np.arange(n_samples)
                )
            )
            # build LinearConstraint matrix, using coo to save memory
            X_marg = coo_matrix(
                (marg_vals, (marg_row_idx, marg_col_idx)),
                shape=(n_samples, n_samples + n_features + 1)
            )
            # linear margin constraint matrix + margin LinearConstraint
            marg_cons = LinearConstraint(X_marg, 1, np.inf)
            # bounds on variables. note n_features + 1 coefficients and
            # intercept are unbounded while slack variables must be >= 0
            var_bounds = Bounds(
                np.hstack(
                    (np.full(n_features + 1, -np.inf), np.full(n_samples, 0))
                ),
                np.full(n_features + n_samples + 1, np.inf)
            )
            # solve for coefficients, intercept, slack using trust-constr
            res = minimize(
                primal_obj, np.zeros(n_features + n_samples + 1),
                method="trust-constr", jac=primal_grad, hess=primal_hess,
                bounds=var_bounds, constraints=marg_cons,
                options=dict(gtol=self.tol)
            )
            # separate out weights and intercept
            weights, intercept = res.x[:n_features], res.x[n_features]
        # set attributes
        self.classes_ = labels
        self.coef_ = weights
        self.intercept_ = intercept
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
        # return predictions. negative class is class 0, positive is class 1
        return np.where(
            X @ self.coef_ + self.intercept_ > 0,
            self.classes_[1], self.classes_[0]
        )

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