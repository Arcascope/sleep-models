"""The WatchTree model: a LightGBM gradient-boosted tree ensemble that stages each 30-s epoch.

Each boosting round adds four trees, one per class (0 wake, 1 light, 2 deep, 3 REM); an epoch's
stage is the class with the highest summed score. Trees are at most depth 6 with up to 64
leaves, the complete trees that :mod:`~arcasleep_watchtree.export` writes for the watch.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

MAX_ROUNDS = 130
#: Share of the training persons held out for early stopping.
VALIDATION_SHARE = 0.15
EARLY_STOPPING_ROUNDS = 20


class WatchTree:
    """LightGBM on the columns ``features`` of a feature table, trained on ``y4``.

    ``max_bin`` is the number of histogram bins per feature; the watch export uses 15.
    """

    def __init__(self, features: list[str], max_bin: int = 255, seed: int = 0):
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
        return {**PARAMS, "max_bin": self.max_bin, "seed": self.seed}

    def fit(self, train: pd.DataFrame) -> WatchTree:
        """Train up to ``MAX_ROUNDS`` rounds, early stopping on ``VALIDATION_SHARE`` of persons."""
        persons = sorted(train.person.unique())
        rng = np.random.default_rng(self.seed)
        n_val = max(1, int(round(VALIDATION_SHARE * len(persons))))
        is_val = train.person.isin(set(rng.choice(persons, size=n_val, replace=False))).to_numpy()
        dtrain = self._dataset(train[~is_val])
        dval = self._dataset(train[is_val], reference=dtrain)
        self.booster = lgb.train(
            self._params(),
            dtrain,
            MAX_ROUNDS,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
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
