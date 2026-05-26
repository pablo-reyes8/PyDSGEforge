"""Compatibility import for the high-level DSGE facade.

The original module is named ``dgse.py``. New code should import from
``src.dsge`` while existing notebooks that import ``src.dgse`` keep working.
"""

from src.dgse import DSGE, ModelSignature

__all__ = ["DSGE", "ModelSignature"]
