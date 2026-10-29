"""One subject-wise k-fold LightGBM fit on a cohort's feature table.

It confirms that the features carry signal; it is not the experiment.

    python -m arcasleep_watchtree.sanity --features results/<run>/features-wearanize.parquet
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import GroupKFold

from arcasleep_watchtree.features import feature_names

PARAMS = {
    "objective": "multiclass",
    "num_class": 4,
    "max_depth": 6,
    "num_leaves": 63,
    "learning_rate": 0.1,
    "verbose": -1,
    "seed": 0,
}
ROUNDS = 100


def cross_validate(table: pd.DataFrame, folds: int = 5, features: list[str] | None = None) -> dict:
    import lightgbm as lgb

    features = features or feature_names()
    table = table[table.y4 >= 0].reset_index(drop=True)
    x, y = table[features].to_numpy(np.float32), table.y4.to_numpy()
    predicted = np.empty_like(y)
    gain = np.zeros(len(features))
    for train, test in GroupKFold(folds).split(x, y, groups=table.subject):
        booster = lgb.train(PARAMS, lgb.Dataset(x[train], y[train], feature_name=features), ROUNDS)
        predicted[test] = booster.predict(x[test]).argmax(axis=1)
        gain += booster.feature_importance("gain")
    return {
        "subjects": int(table.subject.nunique()),
        "epochs": int(len(table)),
        "kappa_4stage": float(cohen_kappa_score(y, predicted)),
        "kappa_sleep_wake": float(cohen_kappa_score(y > 0, predicted > 0)),
        "top_gain": [
            {"feature": features[i], "share_of_gain": float(gain[i] / gain.sum())}
            for i in np.argsort(-gain)[:20]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    result = cross_validate(pd.read_parquet(args.features))
    text = json.dumps(result, indent=1)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
