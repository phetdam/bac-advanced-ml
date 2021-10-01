"""Discriminant-based generative classifiers.

.. codeauthor:: Derek Huang <djh458@stern.nyu.edu>
"""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import check_array, check_X_y


class LinearDiscriminantAnalysis(BaseEstimator):
    """Reference implementation of linear discriminant analysis.

    coef_, intercept_ computed by inversion of sample covariance matrix. The
    sample covariance matrix may have shrinkage applied. Without shrinkage, it
    is the maximum likelihood estimate of the true covariance matrix.

    Parameters
    ----------
    shrinkage : float, default=None
        Value in [0, 1] controlling the shrinkage of the covariance matrix
        estimation. If shrinkage is not None, the covariance matrix estimate
        will be shrinkage * tr(S) / n_features * I + (1 - shrinkage) * S.
        Here S is the maximum likelihood estimate for the covariance matrix,
        I is the identity matrix, tr(S) is the trace of S.

    Attributes
    ----------
    classes_ : numpy.ndarray
        Array of class labels known to the classifier, shape (n_classes,)
    coef_ : numpy.ndarray
        Model coefficients of the fitted model, shape (n_features,) in the case
        of two classes and shape (n_classes, n_features) with >2 classes.
    covariance_ : numpy.ndarray
        Covariance matrix shared by the classes, shape (n_features, n_features)
    intercept_ : numpy.ndarray
        Intercept (bias) terms for the model, shape (n_classes,).
    means_ : numpy.ndarray
        Class means, shape (n_classes, n_features)
    priors_ : numpy.ndarray
        Class priors, shape (n_classes,)

    Methods
    -------
    fit(X, y)
        Compute parameters to best fit the model to X, y.
    predict(X)
        Return predictions on given data X.
    score(X, y)
        Return the accuracy of the predictions given true labels y.
    """

    def __init__(self, shrinkage=None):
        # check that shrinkage is valid
        if shrinkage is None:
            pass
        elif isinstance(shrinkage, float):
            if shrinkage < 0 or shrinkage > 1:
                raise ValueError("shrinkage must be in [0, 1]")
        else:
            raise TypeError("shrinkage must be float in [0, 1]")
        # set shrinkage attribute
        self.shrinkage = shrinkage

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
        # get number of unique labels in y. if labels.size < 2, raise error
        labels = np.unique(y)
        if labels.size < 2:
            raise ValueError(
                "Cannot fit coefficients on single-class classification task"
            )
        # number of classes
        n_classes = labels.size
        # compute class priors
        priors = np.empty(n_classes)
        for i in range(n_classes):
            priors[i] = (y == labels[i]).mean()
        # compute the class means
        means = np.empty((n_classes, n_features))
        for i in range(n_classes):
            means[i] = X[y == labels[i]].mean(axis=0)
        # compute shared covariance matrix (convex combination of class prior
        # weighted covariance matrices)
        cov = np.zeros((n_features, n_features))
        for i in range(n_classes):
            # need ddof=0 for maximum likelihood estimate; numpy API change
            # has made it such that the unbiased sample covariance is default
            cov += priors[i] * np.cov(X[y == labels[i]].T, ddof=0)
        # apply shrinkage to cov if shrinkage is not None
        alpha = self.shrinkage
        if alpha is not None:
            cov = (
                alpha * np.trace(cov) * np.eye(n_features) / n_features +
                (1 - alpha) * cov
            )
        # inverse of covariance matrix (in practice, use np.linalg.lstsq)
        cov_i = np.linalg.inv(cov)
        # compute coefficients
        if n_classes == 2:
            self.coef_ = cov_i @ (means[1] - means[0])
        else:
            self.coef_ = (cov_i @ means.T).T
        # compute intercept(s)
        if n_classes == 2:
            self.intercept_ = (
                -0.5 *
                (means[1] @ cov_i @ means[1] - means[0] @ cov_i @ means[0]) +
                np.log(priors[1] / priors[0])
            )
        else:
            self.intercept_ = (
                -0.5 * np.diag(means @ self.coef_.T) + np.log(priors)
            )
        # set other attributes
        self.classes_ = labels
        self.priors_ = priors
        self.means_ = means
        self.covariance_ = cov
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
        # two-class case
        if self.classes_.size == 2:
            if X.shape[1] != self.coef_.shape[0]:
                raise ValueError("n_features must match coef_.shape[0]")
        else:
            if X.shape[1] != self.coef_.shape[1]:
                raise ValueError("n_features must match coef_.shape[1]")
        # two-class. positive class if decision > 0, else negative
        if self.classes_.size == 2:
            return np.where(X @ self.coef_ + self.intercept_ > 0, 1, 0)
        # multi-class. get argmax of decision function values and return labels
        return self.classes_[
            np.argmax(X @ self.coef_.T + self.intercept_, axis=1)
        ]

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