__doc__ = """Nesterov's accelerated gradient descent.

.. codeauthor:: Derek Huang <djh458@stern.nyu.edu>
"""

import numpy as np
from time import perf_counter


# allowable learning rate schedules
_learning_rates = ("constant", "backtrack")


class FastGradResult:
    """Simple wrapper for the results of nag.

    After initialization, object becomes "immutable".

    Parameters
    ----------
    res : numpy.ndarray
        Final parameter estimate returned by nag, either 1D or 2D.
    loss : float
        Final value of the objective function
    grad : numpy.ndarray
        Final value of the gradient, same shape as res
    n_iter : int
        Total number of iterations taken
    fit_time : float
        Fractional seconds taken until convergence.
    """
    # class attribute that allows locking
    _lock = False

    def __init__(self, res, loss, grad, n_iter, fit_time):
        self.res = res
        self.loss = loss
        self.grad = grad
        self.n_iter = n_iter
        self.fit_time = fit_time
        # lock the instance
        self._lock = True

    def __setattr__(self, name, value):
        if self._lock:
            raise AttributeError("FastGradResult instance is immutable")
        object.__setattr__(self, name, value)


def armijo_backtrack(
    fobj, x, eta0=1., fgrad=None, args=None, arm_alpha=0.5, arm_gamma=0.8
):
    """Compute step size using Armijo backtracking rule for gradient updates.

    See docstring of nag for details on unlisted parameters.

    Parameters
    ----------
    x : numpy.ndarray
        Current point to evaluate fobj, fgrad at.

    Returns
    -------
    float
        Step size in (0, eta0].
    """
    # current step size
    eta = eta0
    # if fgrad is None, assumes fobj return loss and grad. use lambdas to wrap
    # the original fobj (_fobj), returning loss and grad separately.
    if fgrad is None:
        _fobj = fobj
        fobj = lambda x, *args: _fobj(x, *args)[0]
        fgrad = lambda x, *args: _fobj(x, *args)[1]
    # current value of gradient and its mean
    grad = fgrad(x, *args)
    # scale down step size until termination condition is met + return
    while (
        fobj(x - eta * grad, *args) > fobj(x, *args) -
        arm_alpha * eta * np.power(grad, 2).sum()
    ):
        eta = arm_gamma * eta
    return eta