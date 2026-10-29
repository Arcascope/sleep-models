"""Boosted trees that stage 30-s epochs from wrist accelerometer and heart rate.

The same model runs in Python and on a Garmin watch.
"""

from arcasleep_tree.tree import DEFINITION, fit, load, predict, save, to_monkey_c, train

__all__ = ["DEFINITION", "fit", "load", "predict", "save", "to_monkey_c", "train"]
