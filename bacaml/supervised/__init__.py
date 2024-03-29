"""__init__.py for bac_advanced_ml.supervised subpackage.

Contains reference implementations of supervised learning models. Student
implementations are expected to be reasonably similar.

.. codeauthor:: Derek Huang <djh458@stern.nyu.edu>
"""

# pylint: disable=import-error
from ._discriminant_analysis import LinearDiscriminantAnalysis
from ._linear import LogisticRegression, RidgeRegression
from ._svm import LinearSVC