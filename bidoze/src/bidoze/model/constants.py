"""Stage maps of the BiDoze label contract: model semantics, not dataset-specific policy."""

STAGE_1_MAP = {0: [0], 1: [1, 2, 3], 2: [-2]}
STAGE_2_MAP = {0: [0], 1: [1, 2], 2: [3], 3: [-2]}
STAGE_3_MAP = {0: [0], 1: [1], 2: [2], 3: [3], 4: [-2]}

GAP_LABEL = -2
SPECTROGRAM_PADDING_VALUE = -1.0
DEFAULT_FEEDBACK_FOCAL_ALPHA = (1.0, 1.0)
