"""pytest fixtures required by all unit test subpackages.

.. codeauthor:: Derek Huang <djh458@stern.nyu.edu>
"""

import pytest


@pytest.fixture(scope="session")
def global_seed():
    """Universal seed value to be reused by all test methods.

    Returns
    -------
    int
    """
    return 7