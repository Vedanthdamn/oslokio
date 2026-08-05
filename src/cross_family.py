import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.evaluation import feature_columns

# below this many models on either side the estimate is not meaningful
MIN_POSITIVES = 12


def _matrix(df, cols):
    X = df[cols].to_numpy(dtype=np.float64)
    y = (df["label"] == "backdoored").astype(int).to_numpy()
    return X, y


def transfer_auc(df, cols, train_family, test_family, seed):
    """Trains a detector that has only ever seen one attack family and measures how
    well it ranks a different family. The diagonal is the same-family ceiling."""
    clean = df[df["label"] == "clean"]
    clean_train, clean_test = train_test_split(clean, test_size=0.35, random_state=seed)

    train_pos = df[(df["label"] == "backdoored") & (df["trigger_type"] == train_family)]
    test_pos = df[(df["label"] == "backdoored") & (df["trigger_type"] == test_family)]

    if train_family == test_family:
        # split the family so train and test positives stay disjoint
        train_pos, test_pos = train_test_split(train_pos, test_size=0.35, random_state=seed)

    # too few positives and the learner degenerates to a constant prediction, which
    # scores exactly 0.5 -- indistinguishable in a table from a genuine chance result.
    if len(train_pos) < MIN_POSITIVES or len(test_pos) < MIN_POSITIVES:
        return float("nan")

    train_df = pd.concat([clean_train, train_pos])
    test_df = pd.concat([clean_test, test_pos])

    X_tr, y_tr = _matrix(train_df, cols)
    X_te, y_te = _matrix(test_df, cols)

    model = HistGradientBoostingClassifier(random_state=seed)
    model.fit(X_tr, y_tr)
    probs = model.predict_proba(X_te)[:, 1]
    if np.unique(probs).size == 1:
        return float("nan")
    return roc_auc_score(y_te, probs)


def main(features_csv: str, out_path: Path, n_seeds: int):
    df = pd.read_csv(features_csv)
    cols = feature_columns(df)
    families = sorted(df.loc[df["label"] == "backdoored", "trigger_type"].dropna().unique())

    if len(families) < 2:
        print(f"need at least 2 trigger families, found {families}")
        return

    print(f"cross-family transfer, {n_seeds} seeds (rows = trained on, cols = tested on)\n")
    grid = np.zeros((len(families), len(families)))
    spread = np.zeros_like(grid)

    for i, train_family in enumerate(families):
        for j, test_family in enumerate(families):
            scores = [transfer_auc(df, cols, train_family, test_family, s) for s in range(n_seeds)]
            grid[i, j], spread[i, j] = np.nanmean(scores), np.nanstd(scores)

    header = "".join(f"{f[:11]:>13s}" for f in families)
    print(f"{'trained on':>15s}{header}")
    for i, train_family in enumerate(families):
        row = "".join("     n/a     " if np.isnan(grid[i, j])
                      else f"{grid[i, j]:>8.3f}±{spread[i, j]:<4.2f}" for j in range(len(families)))
        print(f"{train_family:>15s}{row}")

    mask = ~np.eye(len(families), dtype=bool)
    diag = float(np.nanmean(np.diag(grid)))
    off = float(np.nanmean(grid[mask]))
    print(f"\nsame-family mean AUC:  {diag:.3f}")
    print(f"cross-family mean AUC: {off:.3f}")
    print(f"generalization gap:    {diag - off:.3f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"families": families, "auc": grid.tolist(), "std": spread.tolist(),
                   "same_family_mean": diag, "cross_family_mean": off}, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-csv", type=str, default="data/features.csv")
    parser.add_argument("--out", type=str, default="data/cross_family.json")
    parser.add_argument("--n-seeds", type=int, default=10)
    args = parser.parse_args()
    main(args.features_csv, Path(args.out), args.n_seeds)
