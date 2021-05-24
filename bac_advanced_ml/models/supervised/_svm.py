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
    dual formulation of the problem can be solved. The dual formulation is
    solved using scipy's trust-constr implementation while the primal
    formulation is solved using either subgradient descent or trust-constr.

    .. note::

       This is a toy implementation and scales poorly when the number of
       examples is large. In particular, memory consumption using trust-constr
       will be O(n_samples ** 2) when dual=True due to the requirements of
       storing a dense Hessian. When dual=False, the memory usage is linear
       since COO sparse matrix can be used to store the primal Hessian.

    Parameters
    ----------
    solver : {"subgrad", "trust-constr"}, default="trust-constr"
        Solver for model fitting. "subgrad" can only be used with dual=False,
        while "trust-constr" can be used for dual=False and dual=True.
    dual : bool, default=True
        True to use dual formulation, False for primal formulation.
    tol : float, default=1e-8
        Tolerance for solver. Passed to scipy.optimize.minimize trust-constr
        solver method's gtol option. The subgradient descent method terminates
        when the squared norm of the gradient / n_features is less than tol.
        Squaring the norm loosens the convergence criteria.
    C : float, default=1.
        Inverse regularization parameter/maximum Lagrange multiplier value in
        the dual formulation for the linear SVC.
    max_iter : int, default=1000
        Maximum number of subgradient descent/trust-constr iterations.
        Subgradient descent has slower convergence and often requires more.

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
    # acceptable solvers
    _solvers = ("subgrad", "trust-constr")

    def __init__(
        self, solver="trust-constr", dual=True, tol=1e-8, C=1., max_iter=1000
    ):
        if solver not in self._solvers:
            raise ValueError(f"solver must be one of {self._solvers}")
        if not isinstance(dual, (bool, np.bool_)):
            raise TypeError("dual must be bool")
        if tol <= 0:
            raise ValueError("tol must be positive")
        if C <= 0:
            raise ValueError("C must be positive")
        if max_iter < 1:
            raise ValueError("max_iter must be positive")
        # cannot have solver="subgrad" when dual=True
        if dual and solver == "subgrad":
            raise ValueError("solver=\"subgrad\" cannot be used if dual=True")
        # assign attribtues
        self.solver = solver
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
            # compute primal weights and intercept from dual variables
            weights = ((y_mask * res.x).reshape(-1, 1) * X).sum(axis=0)
            intercept = (y_mask - X @ weights).mean()
        # else solving primal problem
        else:
            # if solver == "subgrad", use subgradient descent to solve
            if self.solver == "subgrad":
                pass
            # else use trust-constr method in scipy.optimize.minimize
            else:
                # nonzero hessian elements
                hess_vals = np.hstack(
                    (np.ones(n_features), self.C * np.ones(n_samples))
                )
                # row, columns indices for nonzero hessian elements. these are
                # the same since the hessian is diagonal
                hess_idx = np.hstack(
                    (
                        np.arange(n_features),
                        n_features + 1 + np.arange(n_samples)
                    )
                )
                # compute hessian matrix, using coo to save memory
                hess = coo_matrix(
                    (hess_vals, (hess_idx, hess_idx)),
                    shape=(
                        n_samples + n_features + 1,
                        n_samples + n_features + 1
                    )
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
                # margin constraint
                marg_cons = LinearConstraint(
                    y_mask.reshape(-1, 1) * np.hstack(
                        (
                            X, np.ones(n_samples).reshape(-1, 1),
                            np.eye(n_samples)
                        )
                    ),
                    0, np.inf
                )
                # bounds on variables. note n_features + 1 coefficients and
                # intercept are unbounded while slack variables must be >= 0
                var_bounds = Bounds(
                    np.hstack(
                        (
                            np.full(n_features + 1, -np.inf),
                            np.full(n_samples, 0)
                        )
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
        # # set attributes
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