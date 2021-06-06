__doc__ = """Unit tests for the nag solver.

.. codeauthor:: Derek Huang <djh458@stern.nyu.edu>
"""

import numpy as np
import pytest

# pylint: disable=relative-beyond-top-level
from .._fast_gd import nag_solver


# mixtures of learning rate schedules and step sizes
_nag_learn_rates = [("backtrack", 1.), ("constant", 0.1)]


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
        fobj, x0, fgrad=fgrad, learning_rate=learning_rate,
        eta0=eta0, tol=1e-8, max_iter=2000
    )
    # check that res.loss is more or less the same as the optimal fobj value
    np.testing.assert_allclose(res.loss, fobj(sol))
    # check that res.res is close to the actual solution of convex_quad_min.
    # need looser criteria on assert_allclose since optimization is hard
    np.testing.assert_allclose(res.res, sol, rtol=1e-4)