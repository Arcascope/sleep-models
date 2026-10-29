"""The WatchTree model: a LightGBM gradient-boosted tree ensemble that stages each 30-s epoch.

Each boosting round adds four trees, one per class (0 wake, 1 light, 2 deep, 3 REM); an epoch's
stage is the class with the highest summed score. Trees are at most depth 6 with up to 64
leaves, the complete trees that :mod:`~arcasleep_watchtree.watch.export` writes for the watch.

The LightGBM parameters are in ``config/model.json``. The rounds and early stopping are in
``config/training.json``: ``max_rounds``, ``validation_share`` (the share of training persons
held out for early stopping) and ``early_stopping_rounds``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"


@dataclass(frozen=True)
class TreeSpec:
    """LightGBM parameters (``model.json``) and the training settings (``training.json``)."""

    params: dict
    max_rounds: int
    validation_share: float
    early_stopping_rounds: int

    @classmethod
    def load(
        cls,
        model_config: Path = CONFIG_DIR / "model.json",
        training_config: Path = CONFIG_DIR / "training.json",
    ) -> TreeSpec:
        params = json.loads(Path(model_config).read_text())
        return cls(params, **json.loads(Path(training_config).read_text()))


class WatchTree:
    """LightGBM on the columns ``features`` of a feature table, trained on ``y4``.

    ``max_bin`` is the number of histogram bins per feature; the watch export uses 15.
    """

    def __init__(self, spec: TreeSpec, features: list[str], max_bin: int = 255, seed: int = 0):
        self.spec = spec
        self.features = features
        self.max_bin = max_bin
        self.seed = seed
        self.booster: lgb.Booster | None = None

    def _dataset(self, table: pd.DataFrame, **kwargs) -> lgb.Dataset:
        x = table[self.features].to_numpy(np.float32)
        y = table.y4.to_numpy()
        params = {"max_bin": self.max_bin}
        return lgb.Dataset(x, y, feature_name=self.features, params=params, **kwargs)

    def _params(self) -> dict:
        return {**self.spec.params, "max_bin": self.max_bin, "seed": self.seed}

    def fit(self, train: pd.DataFrame) -> WatchTree:
        """Train up to ``max_rounds`` rounds, early stopping on ``validation_share`` of persons."""
        persons = sorted(train.person.unique())
        rng = np.random.default_rng(self.seed)
        n_val = max(1, int(round(self.spec.validation_share * len(persons))))
        is_val = train.person.isin(set(rng.choice(persons, size=n_val, replace=False))).to_numpy()
        dtrain = self._dataset(train[~is_val])
        dval = self._dataset(train[is_val], reference=dtrain)
        self.booster = lgb.train(
            self._params(),
            dtrain,
            self.spec.max_rounds,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(self.spec.early_stopping_rounds, verbose=False)],
        )
        return self

    def fit_rounds(self, train: pd.DataFrame, rounds: int) -> WatchTree:
        """Train exactly ``rounds`` rounds on all of ``train``, with no early stopping."""
        self.booster = lgb.train(self._params(), self._dataset(train), rounds)
        return self

    @property
    def best_iteration(self) -> int:
        return self.booster.best_iteration

    def predict(self, table: pd.DataFrame) -> np.ndarray:
        """Stage per row: 0 wake, 1 light, 2 deep, 3 REM."""
        x = table[self.features].to_numpy(np.float32)
        return self.booster.predict(x, num_iteration=self.best_iteration).argmax(axis=1)
