"""Crane Tkinter UI package.

The public imports remain stable while calibration and competition concerns
live in separate modules.
"""

from .calibration import CAPTURE_OFFLINE_AFTER_FAILURES, CalibrationUI, _InferenceRateMeter
from .competition import CompetitionUI

__all__ = [
    "CAPTURE_OFFLINE_AFTER_FAILURES",
    "CalibrationUI",
    "CompetitionUI",
    "_InferenceRateMeter",
]
