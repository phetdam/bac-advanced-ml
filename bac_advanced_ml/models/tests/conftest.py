__doc__ = "Global fixtures for model testing."

import pytest
from sklearn.datasets import make_regression
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
    tuple
        X_train, X_test, y_train, y_test, coef, bias. X_train has shape
        (480, 10), X_test has shape (120, 10), y_train has shape (480,), y_test
        has shape (120,), the true coefficients coef has shape (10,), while the
        true intercept bias is a float.
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