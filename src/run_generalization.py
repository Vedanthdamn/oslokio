import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.evaluation import (
    ARCH_IDENTIFYING,
    evaluate_transfer,
    feature_columns,
    format_summary,
    summarize,
)


def _split_pool(pool: pd.DataFrame, seed: int, val_frac: float = 0.2):
    return train_test_split(pool, test_size=val_frac, stratify=pool["label"], random_state=seed)


def id_split(df: pd.DataFrame, seed: int):
    train, temp = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=seed)
    id_val, test = train_test_split(temp, test_size=0.5, stratify=temp["label"], random_state=seed)
    return train, id_val, test


def _holdout_positives(df: pd.DataFrame, ood_pos: pd.DataFrame, seed: int):
    """Builds an OOD test set from held-out backdoored models plus a disjoint slice of
    clean models, so the OOD set has both classes and shares nothing with training."""
    clean = df[df["label"] == "clean"]
    n_neg = min(len(ood_pos), max(len(clean) // 3, 1))
    clean_ood, clean_train = train_test_split(clean, train_size=n_neg, random_state=seed)

    kept_backdoor = df[(df["label"] == "backdoored") & (~df["model_id"].isin(ood_pos["model_id"]))]
    pool = pd.concat([clean_train, kept_backdoor])
    train, id_val = _split_pool(pool, seed)
    return train, id_val, pd.concat([ood_pos, clean_ood])


def family_holdout(df: pd.DataFrame, family: str, seed: int):
    ood_pos = df[(df["label"] == "backdoored") & (df["trigger_type"] == family)]
    return _holdout_positives(df, ood_pos, seed)


def corner_holdout(df: pd.DataFrame, corner: str, seed: int):
    ood_pos = df[
        (df["label"] == "backdoored")
        & (df["trigger_type"] == "corner_patch")
        & (df["trigger_corner"] == corner)
    ]
    return _holdout_positives(df, ood_pos, seed)


def arch_holdout(df: pd.DataFrame, depth: int, seed: int):
    ood = df[df["n_conv_layers"] == depth]
    pool = df[df["n_conv_layers"] != depth]
    train, id_val = _split_pool(pool, seed)
    return train, id_val, ood


def run_axis(name, split_fn, df, cols, n_seeds, shuffle_labels=False):
    runs, n_test = [], 0
    for seed in range(n_seeds):
        train, id_val, ood = split_fn(df, seed)
        # an axis is only meaningful if both classes survive in training and in the
        # OOD set -- holding out the only trigger family, for instance, leaves nothing
        # positive to learn from.
        if ood["label"].nunique() < 2 or train["label"].nunique() < 2:
            print(f"\n=== {name}: SKIPPED (degenerate split) ===")
            return None
        if shuffle_labels:
            train = train.copy()
            train["label"] = np.random.default_rng(seed).permutation(train["label"].values)
        runs.append(evaluate_transfer(train, id_val, ood, cols, seed))
        n_test = len(ood)
    summary = summarize(runs)
    print(format_summary(name, summary, n_test, n_seeds))
    return {"name": name, "n_test": n_test, **{k: v for k, v in summary.items()}}


def main(features_csv: str, out_path: Path, n_seeds: int):
    df = pd.read_csv(features_csv)
    all_cols = feature_columns(df)
    arch_cols = feature_columns(df, drop=ARCH_IDENTIFYING)

    print(f"population: {len(df)} models "
          f"({(df.label == 'clean').sum()} clean, {(df.label == 'backdoored').sum()} backdoored)")
    families = sorted(df.loc[df["label"] == "backdoored", "trigger_type"].dropna().unique())
    print(f"trigger families: {families}")
    print(f"features: {len(all_cols)} ({len(arch_cols)} for architecture holdout)")

    results = []

    results.append(run_axis("in-distribution", id_split, df, all_cols, n_seeds))

    for family in families:
        results.append(run_axis(
            f"OOD: held-out trigger family ({family})",
            lambda d, s, f=family: family_holdout(d, f, s),
            df, all_cols, n_seeds,
        ))

    for depth in sorted(df["n_conv_layers"].unique()):
        results.append(run_axis(
            f"OOD: held-out architecture (n_conv_layers={int(depth)})",
            lambda d, s, k=depth: arch_holdout(d, k, s),
            df, arch_cols, n_seeds,
        ))

    if "corner_patch" in families:
        for corner in sorted(df["trigger_corner"].dropna().unique()):
            results.append(run_axis(
                f"OOD: held-out trigger corner ({corner})",
                lambda d, s, c=corner: corner_holdout(d, c, s),
                df, all_cols, n_seeds,
            ))

    control = run_axis("CONTROL: shuffled training labels", id_split, df, all_cols,
                       n_seeds, shuffle_labels=True)
    results.append(control)

    results = [r for r in results if r is not None]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {len(results)} result rows to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-csv", type=str, default="data/results/features.csv")
    parser.add_argument("--out", type=str, default="data/results/generalization.json")
    parser.add_argument("--n-seeds", type=int, default=10)
    args = parser.parse_args()

    main(args.features_csv, Path(args.out), args.n_seeds)
