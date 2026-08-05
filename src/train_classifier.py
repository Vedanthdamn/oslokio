import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

NON_FEATURE_COLS = {
    "model_id", "label", "clean_test_acc", "backdoor_success_rate",
    "trigger_type", "trigger_size", "trigger_corner", "trigger_color",
    "trigger_target_class", "trigger_poison_frac",
}


def prepare_xy(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols].to_numpy(dtype=np.float64)
    y = (df["label"] == "backdoored").astype(int).to_numpy()
    return X, y, feature_cols


def train_and_evaluate(features_csv: str, out_dir: Path, seed: int = 0):
    df = pd.read_csv(features_csv)
    X, y, feature_cols = prepare_xy(df)

    idx = np.arange(len(df))
    idx_train, idx_temp = train_test_split(idx, test_size=0.3, stratify=y, random_state=seed)
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=0.5, stratify=y[idx_temp], random_state=seed
    )

    model = HistGradientBoostingClassifier(random_state=seed)
    model.fit(X[idx_train], y[idx_train])

    for name, split_idx in [("val", idx_val), ("test", idx_test)]:
        preds = model.predict(X[split_idx])
        probs = model.predict_proba(X[split_idx])[:, 1]
        print(f"\n=== {name} ({len(split_idx)} models) ===")
        print(classification_report(y[split_idx], preds, target_names=["clean", "backdoored"]))
        print(f"ROC-AUC: {roc_auc_score(y[split_idx], probs):.4f}")

    perm = permutation_importance(
        model, X[idx_test], y[idx_test], n_repeats=20, random_state=seed, scoring="roc_auc"
    )
    importance_order = np.argsort(perm.importances_mean)[::-1]
    print("\ntop 15 features by permutation importance (test set, ROC-AUC drop):")
    for i in importance_order[:15]:
        print(f"  {feature_cols[i]:35s} {perm.importances_mean[i]:.4f} +/- {perm.importances_std[i]:.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "classifier.joblib")
    splits = {
        "train": df.iloc[idx_train]["model_id"].tolist(),
        "val": df.iloc[idx_val]["model_id"].tolist(),
        "test": df.iloc[idx_test]["model_id"].tolist(),
    }
    with open(out_dir / "splits.json", "w") as f:
        json.dump(splits, f, indent=2)

    return model, feature_cols


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-csv", type=str, default="data/features.csv")
    parser.add_argument("--out-dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train_and_evaluate(args.features_csv, Path(args.out_dir), args.seed)
