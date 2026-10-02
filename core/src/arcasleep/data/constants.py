"""Dependency-free constants shared by the CLI and the data layer.

This module must import nothing outside the standard library: the CLI entry
point reads it before any optional dependency is installed, so a base install
can answer ``--help`` and run the manifest commands with no extras present.
"""
from __future__ import annotations

# Raw 5C PSG labels: 0=Wake, 1..4=N1..N4, 5=REM. Anything negative is
# PSG_MASK (-1, unscored) or PAD_CLASS_LABEL (-2), so a scored epoch is y >= 0.
# A recording with less staged PSG than this is skipped during preparation.
MIN_SCORED_MINUTES = 120.0

#: The label of an epoch with no accelerometer data, including padding. Models
#: with a gap head predict it as its own class.
GAP_LABEL = -2
#: The value pisces-lite writes into every bin and channel of a frame with no
#: accelerometer data.
SPECTROGRAM_PADDING_VALUE = -1.0
