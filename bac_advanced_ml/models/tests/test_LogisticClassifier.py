__doc__ = "Unit tests for the LogisticClassifier class."

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression as _LogisticRegression

# pylint: disable=relative-beyond-top-level
from ..supervised import LogisticRegression

@pytest.mark.skip(reason = "not yet implemented")
def test_res():
    pass