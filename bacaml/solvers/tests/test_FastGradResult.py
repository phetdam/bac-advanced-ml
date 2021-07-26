"""Unit tests for the FastGradResult solver result class.

.. codeauthor:: Derek Huang <djh458@stern.nyu.edu>
"""

import pytest

# pylint: disable=relative-beyond-top-level
from .._fast_gd import FastGradResult


def test_fgr_lock():
    "Test locking (immutability) mechanism of the FastGradResult."
    # no input checking, so we can put dummy inputs. phrases spoken by tazawa
    # from ep 4 of otokojuku; see https://www.youtube.com/watch?v=exyMAumtgt8
    fgr = FastGradResult(
        *("fried chicken", "hamburger", "hot dog", "i'm sorry", "hige's sorry")
    )
    # check that attributes were added successfully
    assert fgr.res == "fried chicken"
    assert fgr.loss == "hamburger"
    assert fgr.grad == "hot dog"
    assert fgr.n_iter == "i'm sorry"
    assert fgr.fit_time == "hige's sorry"
    # attempt to add attribute (should get AttributeError)
    with pytest.raises(AttributeError, match=r"FastGradResult"):
        fgr.a = "i am very hungry"
    # attempt to change an attribute (should get AttributeError)
    with pytest.raises(AttributeError, match=r"FastGradResult"):
        fgr.res = "boys be ambitious"