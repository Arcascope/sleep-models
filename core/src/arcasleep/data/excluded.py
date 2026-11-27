"""Recordings left out of every dataset read: cache discovery, split materialisation, raw loading.

Weaver (revised) Fitbit recordings whose stored accelerometer timestamps do not line up with the
PSG: either the timestamps jump forward 65.5 s with no data missing, or the recording is stored in
2-minute stretches that each sit at their own offset, up to about 100 s from the PSG. Either way
the misplaced epochs carry the labels of other epochs. The same nights' other devices are kept.
"""
from __future__ import annotations

EXCLUDED_RECORDINGS: frozenset[str] = frozenset(
    {
        "0k079n0i_fitbit_n1",
        "0qe3p619_fitbit_n1",
        "28901hsq_fitbit_n1",
        "5hru1y2d_fitbit_n1",
        "6snb2koa_fitbit_n1",
        "9ikjkotu_fitbit_n1",
        "aro3b0xg_fitbit_n1",
        "bah9kmku_fitbit_n1",
        "btv0zy4k_fitbit_n1",
        "c2arrzcu_fitbit_n1",
        "diidn07w_fitbit_n1",
        "h6y4a8zy_fitbit_n2",
        "hyuyjljj_fitbit_n1",
        "iqieevb4_fitbit_n1",
        "itgipm40_fitbit_n1",
        "ixa3i2mx_fitbit_n1",
        "kn0upprb_fitbit_n1",
        "mwgsd0cx_fitbit_n1",
        "nvp8m438_fitbit_n1",
        "qzk6k45c_fitbit_n1",
        "rgj98pgq_fitbit_n1",
        "tregiw7y_fitbit_n1",
        "vqn533oy_fitbit_n1",
        "w212mzfk_fitbit_n1",
        "x7u1ddd7_fitbit_n1",
    }
)


def is_excluded(subject_id: str) -> bool:
    """Whether ``subject_id`` is in :data:`EXCLUDED_RECORDINGS`."""
    return subject_id in EXCLUDED_RECORDINGS
