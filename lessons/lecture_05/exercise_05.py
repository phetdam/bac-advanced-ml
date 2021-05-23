__doc__ = "Week 5 exercise: implementation of LDA with covariance shrinkage."

import numpy as np
import pytest
from sklearn.base import BaseEstimator
from sklearn.datasets import make_blobs
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as \
    _LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.utils import check_array, check_X_y


class LinearDiscriminantAnalysis(BaseEstimator):
    """Student implementation of linear discriminant analysis.

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

        ###########################
        ### your code goes here ###
        ###########################

        # compute the class means
        means = np.empty((n_classes, n_features))

        ###########################
        ### your code goes here ###
        ###########################

        # compute shared + shrunk covariance matrix. sample covariance matrix
        # is convex combination of class prior weighted covariance matrices.
        # note that self.shrinkage might be None, in which case no shrinkage!
        cov = np.zeros((n_features, n_features))

        ###########################
        ### your code goes here ###
        ###########################

        # inverse of covariance matrix (in practice, use np.linalg.lstsq).
        cov_i = np.linalg.inv(cov)
        # compute coefficients
        if n_classes == 2:
            self.coef_ = cov_i @ (means[1] - means[0])
        else:
            self.coef_ = (cov_i @ means.T).T
        # compute intercept(s)
        if n_classes == 2:
            self.intercept_ = (
                -0.5 * (
                    means[1] @ cov_i @ means[1] - means[0] @ cov_i @ means[0]
                ) + np.log(priors[1] / priors[0])
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


@pytest.fixture(scope="session")
def blob_multi():
    """Generated multi-class blob classification problem with train/test split.

    Returns
    -------
    X_train : numpy.ndarray
    X_test : numpy.ndarray
    y_train : numpy.ndarray
    y_test : numpy.ndarray
    """
    # seed value for reproducible results
    _seed = 7
    # pylint: disable=unbalanced-tuple-unpacking
    X, y = make_blobs(
        n_samples=600, n_features=10, centers=4, random_state=_seed
    )
    # pylint: enable=unbalanced-tuple-unpacking
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=_seed
    )
    return X_train, X_test, y_train, y_test


@pytest.mark.parametrize("shrinkage", [None, 0.2, 0.7])
def test_res_binary(blob_bin, shrinkage):
    """Check LinearDiscriminantAnalysis equals sklearn for two-class case.

    Checks the fit, predict, and score methods, checking solution and accuracy.
    Also checks the different attributes set during the fitting process.

    Parameters
    ----------
    blob_bin : tuple
        pytest fixture. See blob_bin.
    shrinkage : float
        Shrinkage parameter to pass to LinearDiscriminantAnalysis
    """
    # unpack data from fixture
    X_train, X_test, y_train, y_test = blob_bin
    # fit scikit-learn model and our model
    _lc = _LinearDiscriminantAnalysis(solver="lsqr", shrinkage=shrinkage).fit(
        X_train, y_train
    )
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
        pytest fixture. See blob_multi.
    shrinkage : float
        Shrinkage parameter to pass to LinearDiscriminantAnalysis
    """
    # unpack data from fixture
    X_train, X_test, y_train, y_test = blob_multi
    # fit scikit-learn model and our model
    _lc = _LinearDiscriminantAnalysis(solver="lsqr", shrinkage=shrinkage).fit(
        X_train, y_train
    )
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