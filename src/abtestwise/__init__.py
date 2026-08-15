"""abtestwise: a lightweight toolkit for binary A/B experiment analysis.

Provides frequentist and Bayesian summaries for binary proportions using
aggregate counts, raw samples, or DataFrame-like experiment data.
"""

from __future__ import annotations

from .binary import BinaryABTest
from .result import BinaryABResult

__version__ = "0.2.0"

__all__ = ["BinaryABTest", "BinaryABResult", "__version__"]
