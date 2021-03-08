__doc__ = "Global fixtures for model testing."

import numpy as np
import pytest
from sklearn.datasets import make_blobs, make_regression
from sklearn.model_selection import train_test_split


@pytest.fixture(scope = "session")
def global_seed():
    """Universal seed value to be reused by all test methods.

    Returns
    -------
    int
    """
    return 7


@pytest.fixture(scope = "session")
def linreg(global_seed):
    """Generated linear regression problem with train/test split.

    Uses sklearn.datasets.make_regression. The returned data consists of 600
    samples, 10 features (all informative), and has Gaussian noise with a
    standard deviation of 1 applied to it. The bias of the model is 7.

    Parameters
    ----------
    global_seed : int
        pytest fixture. See conftest.py

    Returns
    -------
    X_train : numpy.ndarray
        Training input data, shape (480, 10)
    X_test : numpy.ndarray
        Test/validation input data, shape (120, 10)
    y_train : numpy.ndarray
        Training response vector, shape (480,)
    y_test : numpy.ndarray
        Test/validation response vector, shape (120,)
    coef : numpy.ndarray
        True model coefficients, shape (10,)
    bias : float
        True model bias (intercept)
    """
    # intercept for the regression problem
    bias = 7
    # generate noisy regression problem
    X, y, coef = make_regression(
        n_samples = 600, n_features = 10, n_informative = 10, bias = bias,
        noise = 1, coef = True, random_state = global_seed
    )
    # split the data with train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.2, random_state = global_seed
    )
    # return split data, coef, and bias
    return X_train, X_test, y_train, y_test, coef, bias


@pytest.fixture(scope = "session")
def blobcls(global_seed):
    """Generated blob classification problem with train/test split.

    Uses sklearn.datasets.make_blobs to make isotropic standard multivariate
    Gaussian blobs in 10-dimensional Euclidean space with centers

    .. code::

       np.array([-4, -2, -5, 1, -6, -2, -7, 2, -5, 1])
       np.array([6, 5, 3, 4, 7, -1, 6, 1, -2, 9])

    The returned data has 600 samples, 10 features. Features are informative by
    construction so we don't use sklearn.datasets.make_classification.

    Parameters
    ----------
    global_seed : int
        pytest fixture. See conftest.py

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
    # centers for the two data clusters
    centers = np.array(
        [[-4, -2, -5, 1, -6, -2, -7, 2, -5, 1],
         [6, 5, 3, 4, 7, -1, 6, 1, -2, 9]]
    )
    # generate noisy classification problem using isotropic Gaussian blobs
    # pylint: disable=unbalanced-tuple-unpacking
    X, y = make_blobs(
        n_samples = 600, n_features = 10, centers = centers,
        random_state = global_seed
    )
    # pylint: enable=unbalanced-tuple-unpacking
    # split the data with train_test_split and return
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.2, random_state = global_seed
    )
    # return split data, coef, and bias
    return X_train, X_test, y_train, y_test