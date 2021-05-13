__doc__ = """__init__.py for bac_advanced_ml.models.supervised subpackage.

Contains reference implementations of supervised learning models. Student
implementations are expected to be reasonably similar.

.. codeauthor:: Derek Huang <djh458@stern.nyu.edu>
"""

# pylint: disable=import-error
from ._discriminant_analysis import LinearDiscriminantAnalysis
from ._linear import LogisticRegression, RidgeRegression