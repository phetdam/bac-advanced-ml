__doc__ = """Week 7 exercise: implementing backtracking and Nesterov's method.

.. codeauthor:: Derek Huang <djh458@stern.nyu.edu>
"""

import numpy as np
import pytest
import scipy.linalg as linalg
from sklearn.datasets import make_spd_matrix
from time import perf_counter


# allowable learning rate schedules
_learning_rates = ("constant", "backtrack")
# mixtures of learning rate schedules and step sizes
_nag_learn_rates = [("backtrack", 1.), ("constant", 0.1)]


class FastGradResult:
    """Simple wrapper for the results of nag_solver.

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


def _armijo_backtrack(
    fobj,
    x,
    eta0=1.,
    fgrad=None,
    args=(),
    arm_alpha=0.5,
    arm_gamma=0.8
):
    """Compute step size using Armijo backtracking rule for gradient updates.

    See docstring of nag_solver for details on unlisted parameters.

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

    ###########################
    ### your code goes here ###
    ###########################

    return eta


def nag_solver(
    fobj,
    x0,
    fgrad=None,
    args=(),
    tol=1e-4,
    max_iter=1000,
    learning_rate="backtrack",
    eta0=1.,
    arm_alpha=0.5,
    arm_gamma=0.8
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
    args : tuple, default=()
        Additional positional arguments to pass to fobj, fgrad.
    tol : float, default=1e-4
        Stopping tolerance. Solving terminates when the norm of the gradient
        is less than tol * grad.size, i.e. tol * number of gradient elements.
    max_iter : int, default=1000
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
    # can only use constant or backtracking strategies
    if learning_rate not in _learning_rates:
        raise ValueError(f"learning_rate must be one of {_learning_rates}")
    # number of elapsed iterations
    n_iter = 0
    # current guess for the solution and current nesterov guess. note that
    # gradients are only evaluated at nest_x, not x.
    x = x0
    nest_x = x0
    # get initial values for gradient. note nest_x = x = x0 here.
    if fgrad is None:
        _, grad = fobj(nest_x, *args)
    else:
        grad = fgrad(nest_x, *args)
    # starting computing fit_time
    fit_time = perf_counter()
    # while not converged (use only gradient norm as convergence criteria)
    while n_iter < max_iter and np.linalg.norm(grad) / grad.size >= tol:

        ###########################
        ### your code goes here ###
        ###########################

        # compute new gradient value
        if fgrad is None:
            _, grad = fobj(nest_x, *args)
        else:
            grad = fgrad(nest_x, *args)
        # update number of iterations
        n_iter += 1
    # compute loss and gradient at final estimate
    if fgrad is None:
        loss, grad = fobj(x, *args)
    else:
        loss = fobj(x, *args)
        grad = fgrad(x, *args)
    # set fit_time
    fit_time = perf_counter() - fit_time
    # return FastGradResult
    return FastGradResult(x, loss, grad, n_iter, fit_time)


@pytest.fixture(scope="session")
def convex_quad_min():
    """Returns objective, gradient, Hessian, solution for a convex QP.

    Returns
    -------
    fobj : function
        Convex, quadratic objective function
    fgrad : function
        Gradient of the objective
    fhess : function
        Hessian of the objective (constant)
    sol : numpy.ndarray
        Global minimizer of the function
    """
    # PRNG seed
    _seed = 7
    # number of features/dimensionality, PRNG
    n_dim = 10
    rng = np.random.default_rng(_seed)
    # make positive definite hessian by adding scaled identity matrix
    hess = make_spd_matrix(n_dim, random_state=_seed)
    hess += 1e-4 * np.eye(n_dim)
    # random linear terms drawn from [-5, 5]
    coef = rng.uniform(low=-5., high=5., size=n_dim)
    # objective function, gradient, and hessian
    fobj = lambda x: 0.5 * x @ hess @ x + coef @ x
    fgrad = lambda x: hess @ x + coef
    fhess = lambda x: hess
    # compute solution using scipy.linalg.solve
    sol = linalg.solve(hess, -coef, check_finite=False, assume_a="pos")
    # return fobj, fgrad, fhess, sol
    return fobj, fgrad, fhess, sol


def test_arm_alpha(convex_quad_min):
    """Test effect of arm_alpha on step size.

    Parameters
    ----------
    convex_quad_min : tuple
        pytest fixture. See convex_quad_min
    """
    # get objective, gradient, solution from convex_quad_min + dimensionality
    fobj, fgrad, _, sol = convex_quad_min
    n_dim = sol.size
    # values of arm_alpha, current point to evaluate obj, grad at
    arm_alphas = np.array([0.1, 0.5, 0.8])
    x0 = np.zeros(n_dim)
    # compute step sizes for each value of arm_alpha
    steps = np.empty(arm_alphas.shape)
    for i, alpha in enumerate(arm_alphas):
        steps[i] = _armijo_backtrack(fobj, x0, fgrad=fgrad, arm_alpha=alpha)
    # check that step sizes are monotone nonincreasing
    for i in range(steps.size - 1):
        assert steps[i] >= steps[i + 1]


def test_arm_gamma(convex_quad_min):
    """Test effect of arm_gamma on step size.

    Parameters
    ----------
    convex_quad_min : tuple
        pytest fixture. See convex_quad_min
    """
    # get objective, gradient, solution from convex_quad_min + dimensionality
    fobj, fgrad, _, sol = convex_quad_min
    n_dim = sol.size
    # values of arm_gamma, current point to evaluate obj, grad at
    arm_gammas = np.array([0.5, 0.8, 0.95])
    x0 = np.zeros(n_dim)
    # compute step sizes for each value of arm_gamma
    steps = np.empty(arm_gammas.shape)
    for i, gamma in enumerate(arm_gammas):
        steps[i] = _armijo_backtrack(fobj, x0, fgrad=fgrad, arm_gamma=gamma)
    # check that step sizes are monotone nondecreasing
    for i in range(steps.size - 1):
        assert steps[i] <= steps[i + 1]


@pytest.mark.parametrize("learning_rate,eta0", _nag_learn_rates)
def test_nag_solver(convex_quad_min, learning_rate, eta0):
    """Test nag_solver on convex QP defined by convex_quad_min.

    Parameters
    ----------
    convex_quad_min : tuple
        pytest fixture. See conftest.py.
    learning_rate : str
        Learning rate schedule to use, either "constant" or "backtrack".
    eta0 : float
        For learning_rate="constant", the learning rate to use, while for
        learning_rate="backtrack", the learning rate upper search bound.
    """
    # get objective, gradient, solution from convex_quad_min + dimensionality
    fobj, fgrad, _, sol = convex_quad_min
    n_dim = sol.size
    # initial guess
    x0 = np.zeros(n_dim)
    # get FastGradResult using nag_solver
    res = nag_solver(
        fobj, x0,
        fgrad=fgrad,
        learning_rate=learning_rate,
        eta0=eta0,
        tol=1e-8,
        max_iter=2000
    )
    # check that res.loss is more or less the same as the optimal fobj value
    np.testing.assert_allclose(res.loss, fobj(sol))
    # check that res.res is close to the actual solution of convex_quad_min.
    # need looser criteria on assert_allclose since optimization is hard
    np.testing.assert_allclose(res.res, sol, rtol=1e-4)