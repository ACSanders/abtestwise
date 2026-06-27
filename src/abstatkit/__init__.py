"""abtestwise: a lightweight toolkit for binary A/B experiment analysis.

Version 0.1 combines frequentist and Bayesian summaries for binary proportions
using aggregate count data.
"""

from __future__ import annotations

from .binary import BinaryABTest
from .result import BinaryABResult

__version__ = "0.1.1"

__all__ = ["BinaryABTest", "BinaryABResult", "__version__"]
