"""Stage maps of the BiDoze label contract: model semantics, not dataset-specific policy."""

from arcasleep.data.constants import GAP_LABEL, SPECTROGRAM_PADDING_VALUE

STAGE_1_MAP = {0: [0], 1: [1, 2, 3], 2: [GAP_LABEL]}
STAGE_2_MAP = {0: [0], 1: [1, 2], 2: [3], 3: [GAP_LABEL]}
STAGE_3_MAP = {0: [0], 1: [1], 2: [2], 3: [3], 4: [GAP_LABEL]}

DEFAULT_FEEDBACK_FOCAL_ALPHA = (1.0, 1.0)

__all__ = [
    "DEFAULT_FEEDBACK_FOCAL_ALPHA",
    "GAP_LABEL",
    "SPECTROGRAM_PADDING_VALUE",
    "STAGE_1_MAP",
    "STAGE_2_MAP",
    "STAGE_3_MAP",
]
