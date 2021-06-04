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


def nag(
    fobj, x0, fgrad=None, args=None, tol=1e-4, max_iter=200,
    learning_rate="backtrack", eta0=1., arm_alpha=0.5, arm_gamma=0.8
):
    """Nesterov's accelerated gradient descent for differentiable objectives.

    Parameters
    ----------
    fobj : function
        Differentiable objective. If fgrad=None, fobj must return (loss, grad).
    x0 : numpy.ndarray
        Initial guess for the solution. Must have shape (n_dims,) or
        (n_dims_1, n_dims_2) matching shape of the fgrad/fobj gradients.
    fgrad : function, default=None
        Gradient of the objective. If None, fobj must return (loss, grad).
    args : tuple, default=None
        Additional positional arguments to pass to fobj, fgrad.
    tol : float, default=1e-4
        Stopping tolerance. Solving terminates when the norm of the gradient
        is less than tol * grad.size, i.e. tol * number of gradient elements.
    max_iter : int, default=200
        Maximum number of iterations before convergence.
    learning_rate : {"constant", "backtrack"}, default="backtrack"
        Learning rate schedule. "constant" uses eta0 as a constant learning
        rate, "backtrack" uses Armijo backtracking line search with arm_alpha,
        arm_gamma, and eta0 as parameters.
    eta0 : float, default=1.
        Step size for constant learning rate schedule, starting step size if
        backtracking line search is used to search for step size.
    arm_alpha : float, default=0.5
        Backtracking line search alpha, must be in (0, 1)
    arm_gamma : float, default=0.8
        Backtracking line search gamma, must be in (0, 1)

    Returns
    -------
    FastGradResult
        Optimization result containing result, final objective value, final
        gradient value, number of iterations, and fitting time. See
        FastGradResult docstring for details on the attributes.
    """
    # if args is None, set to empty tuple
    if args is None:
        args= ()
    # can only use constant or backtracking strategies
    if learning_rate not in _learning_rates:
        raise ValueError(f"learning_rate must be one of {_learning_rates}")
    # number of elapsed iterations
    n_iter = 0
    # current guess for the solution and current nesterov guess. note that
    # gradients are only evaluated at nest_x, not x.
    x = x0
    nest_x = x0
    # get initial values for loss + gradient. note nest_x = x = x0 here.
    if fgrad is None:
        loss, grad = fobj(x, *args)
    else:
        loss = fobj(x, *args)
        grad = fgrad(x, *args)
    # starting computing fit_time
    fit_time = perf_counter()
    # while not converged
    while n_iter < max_iter and np.linalg.norm(grad) / grad.size >= tol:
        # compute step size using chosen method
        if learning_rate == "constant":
            eta = eta0
        elif learning_rate == "backtrack":
            eta = armijo_backtrack(
                fobj, x, eta0=eta0, fgrad=fgrad, args=args,
                arm_alpha=arm_alpha, arm_gamma=arm_gamma
            )
        # new parameter estimate using a gradient step, evaluated at nest_x
        new_x = nest_x - eta * grad
        # update nest_x using new_x and previous estimate x if n_iter >= 1
        if n_iter >= 1:
            nest_x = new_x + n_iter / (n_iter + 3) * (new_x - x)
        # update parameter estimate
        x = new_x
        # compute new objective and gradient values
        if fgrad is None:
            loss, grad = fobj(x, *args)
        else:
            loss = fobj(x, *args)
            grad = fgrad(x, *args)
        # update number of iterations
        n_iter += 1
    # set fit_time
    fit_time = perf_counter() - fit_time
    # return FastGradResult
    return FastGradResult(x, loss, grad, n_iter, fit_time)