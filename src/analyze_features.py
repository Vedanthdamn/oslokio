import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from src.evaluation import feature_columns


def importance_for(df: pd.DataFrame, cols: list[str], n_seeds: int, top_k: int):
    X = df[cols].to_numpy(dtype=np.float64)
    y = (df["label"] == "backdoored").astype(int).to_numpy()

    totals = np.zeros(len(cols))
    for seed in range(n_seeds):
        idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.3,
                                          stratify=y, random_state=seed)
        model = HistGradientBoostingClassifier(random_state=seed).fit(X[idx_tr], y[idx_tr])
        perm = permutation_importance(model, X[idx_te], y[idx_te], n_repeats=8,
                                      random_state=seed, scoring="roc_auc")
        totals += perm.importances_mean
    totals /= n_seeds

    order = np.argsort(totals)[::-1][:top_k]
    out = []
    for i in order:
        col = cols[i]
        clean_mean = df.loc[df["label"] == "clean", col].mean()
        bd_mean = df.loc[df["label"] == "backdoored", col].mean()
        pooled = df[col].std() + 1e-12
        out.append({
            "feature": col,
            "importance": float(totals[i]),
            "clean_mean": float(clean_mean),
            "backdoored_mean": float(bd_mean),
            "effect_size": float((bd_mean - clean_mean) / pooled),
        })
    return out


def main(features_csv: str, out_path: Path, n_seeds: int, top_k: int):
    df = pd.read_csv(features_csv)
    cols = feature_columns(df)

    print(f"=== overall: what the detector keys on ({n_seeds} seeds) ===")
    print(f"{'feature':<38s}{'importance':>12s}{'effect':>9s}  direction")
    overall = importance_for(df, cols, n_seeds, top_k)
    for row in overall:
        direction = "higher when backdoored" if row["effect_size"] > 0 else "lower when backdoored"
        print(f"{row['feature']:<38s}{row['importance']:>12.4f}{row['effect_size']:>9.2f}  {direction}")

    families = sorted(df.loc[df["label"] == "backdoored", "trigger_type"].dropna().unique())
    per_family = {}
    if len(families) > 1:
        clean = df[df["label"] == "clean"]
        for family in families:
            subset = pd.concat([clean, df[(df["label"] == "backdoored") & (df["trigger_type"] == family)]])
            rows = importance_for(subset, cols, n_seeds, top_k)
            per_family[family] = rows
            print(f"\n=== {family} vs clean: top {min(top_k, 5)} features ===")
            for row in rows[:5]:
                arrow = "up" if row["effect_size"] > 0 else "down"
                print(f"  {row['feature']:<36s}{row['importance']:>10.4f}  ({arrow} when backdoored)")

        top_sets = {f: {r["feature"] for r in rows[:10]} for f, rows in per_family.items()}
        print("\n=== signature overlap between families (shared features in each top 10) ===")
        for i, a in enumerate(families):
            for b in families[i + 1:]:
                shared = top_sets[a] & top_sets[b]
                print(f"  {a:>14s} vs {b:<14s} {len(shared):>2d}/10 shared")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"overall": overall, "per_family": per_family}, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-csv", type=str, default="data/results/features.csv")
    parser.add_argument("--out", type=str, default="data/results/feature_analysis.json")
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=15)
    args = parser.parse_args()
    main(args.features_csv, Path(args.out), args.n_seeds, args.top_k)
