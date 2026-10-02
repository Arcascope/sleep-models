"""The evaluation protocol: metrics, and scoring a finished model.

Importing this package registers the stage-level metrics in
:mod:`arcasleep.evaluation.staging_metrics`, which is what makes ``metrics.json``
able to name ``sleep_sensitivity`` and its neighbours alongside pisces-lite's
built-ins. The trainer relies on that registration happening before any
``MetricsLogger`` builds its specs, so it is done here rather than left to
whichever module happens to import the metrics first.
"""
from arcasleep.evaluation import staging_metrics  # noqa: F401

__all__ = ["staging_metrics"]
