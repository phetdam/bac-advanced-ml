__doc__ = """Unit tests for the _armijo_backtrack backtracking implementation.

.. codeauthor:: Derek Huang <djh458@stern.nyu.edu>
"""

import numpy as np
import pytest

# pylint: disable=relative-beyond-top-level
from .._fast_gd import _armijo_backtrack


def test_arm_alpha(convex_quad_min):
    """Test effect of arm_alpha on step size.

    Ceteris paribus, increasing arm_alpha should decrease the step size and
    vice versa, as the damping of the first-order approximation decreases,
    which makes the step-size selection criteria require smaller steps.

    Parameters
    ----------
    convex_quad_min : tuple
        pytest fixture. See conftest.py
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

    Ceteris paribus, increasing arm_gamma should increase the step size and
    vice versa, as increasing arm_gamma makes the step size search finer.

    Parameters
    ----------
    convex_quad_min : tuple
        pytest fixture. See conftest.py
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