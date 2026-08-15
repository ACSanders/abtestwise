"""abtestwise: a lightweight toolkit for A/B experiment analysis."""

from __future__ import annotations

from .binary import BinaryABTest
from .continuous import ContinuousABTest
from .result import BinaryABResult, ContinuousABResult

__version__ = "0.3.0"

__all__ = [
    "BinaryABTest",
    "BinaryABResult",
    "ContinuousABTest",
    "ContinuousABResult",
    "__version__",
]