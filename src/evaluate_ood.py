import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from src.train_classifier import prepare_xy

HELD_OUT_ARCH_DEPTH = 4
HELD_OUT_CORNER = "bottom_right"


def split_population(df: pd.DataFrame):
    is_ood_arch = df.n_conv_layers == HELD_OUT_ARCH_DEPTH
    is_ood_corner = (df.label == "backdoored") & (df.trigger_corner == HELD_OUT_CORNER) & ~is_ood_arch
    trainable_pool = ~is_ood_arch & ~is_ood_corner

    return df[trainable_pool].reset_index(drop=True), df[is_ood_arch].reset_index(drop=True), df[is_ood_corner].reset_index(drop=True)


def report(name: str, model, X, y):
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    print(f"\n=== {name} ({len(y)} models) ===")
    print(classification_report(y, preds, target_names=["clean", "backdoored"], zero_division=0))
    if len(np.unique(y)) > 1:
        print(f"ROC-AUC: {roc_auc_score(y, probs):.4f}")
    else:
        print(f"detection rate: {preds.mean():.4f}")


def evaluate_ood(features_csv: str, out_dir: Path, seed: int = 0):
    df = pd.read_csv(features_csv)
    id_pool, ood_arch, ood_corner = split_population(df)

    print(f"trainable (in-distribution) pool: {len(id_pool)} models")
    print(f"held-out architecture (n_conv_layers={HELD_OUT_ARCH_DEPTH}): {len(ood_arch)} models")
    print(f"held-out trigger corner ({HELD_OUT_CORNER}, backdoored only): {len(ood_corner)} models")

    X_pool, y_pool, feature_cols = prepare_xy(id_pool)
    idx = np.arange(len(id_pool))
    idx_train, idx_test = train_test_split(idx, test_size=0.2, stratify=y_pool, random_state=seed)

    model = HistGradientBoostingClassifier(random_state=seed)
    model.fit(X_pool[idx_train], y_pool[idx_train])

    report("in-distribution test", model, X_pool[idx_test], y_pool[idx_test])

    X_arch, y_arch, _ = prepare_xy(ood_arch)
    report(f"OOD: held-out architecture (n_conv_layers={HELD_OUT_ARCH_DEPTH})", model, X_arch, y_arch)

    X_corner_pos, _, _ = prepare_xy(ood_corner)
    id_test_clean_mask = y_pool[idx_test] == 0
    X_corner_neg = X_pool[idx_test][id_test_clean_mask]
    X_corner = np.concatenate([X_corner_pos, X_corner_neg], axis=0)
    y_corner = np.concatenate([np.ones(len(X_corner_pos)), np.zeros(len(X_corner_neg))])
    report(f"OOD: held-out trigger corner ({HELD_OUT_CORNER})", model, X_corner, y_corner)

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "classifier_ood.joblib")
    with open(out_dir / "ood_splits.json", "w") as f:
        json.dump({
            "id_train": id_pool.iloc[idx_train]["model_id"].tolist(),
            "id_test": id_pool.iloc[idx_test]["model_id"].tolist(),
            "ood_arch": ood_arch["model_id"].tolist(),
            "ood_corner": ood_corner["model_id"].tolist(),
        }, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-csv", type=str, default="data/features.csv")
    parser.add_argument("--out-dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    evaluate_ood(args.features_csv, Path(args.out_dir), args.seed)
