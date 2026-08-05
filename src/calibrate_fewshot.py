import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from src.evaluation import ARCH_IDENTIFYING, feature_columns

# Deployment premise: an auditor facing an unfamiliar architecture cannot easily obtain
# known-backdoored examples of it, but *can* usually obtain a few models they trust are
# clean (ones they trained themselves, or that predate the threat). Those clean
# reference models are enough to place a decision threshold, with no positive examples
# of the new architecture at all.
#
# The deployment-relevant question is not raw accuracy but: at a false-positive budget
# the auditor is willing to pay, what fraction of backdoored models get caught, and does
# the realised false-positive rate actually match the budget?
SHOT_COUNTS = [5, 10, 20, 40]
FPR_BUDGETS = [0.05, 0.10, 0.20]


def _matrix(df, cols):
    X = df[cols].to_numpy(dtype=np.float64)
    y = (df["label"] == "backdoored").astype(int).to_numpy()
    return X, y


def run(features_csv: str, depth: int, n_seeds: int, out_path: Path):
    df = pd.read_csv(features_csv)
    cols = feature_columns(df, drop=ARCH_IDENTIFYING)

    train_pool = df[df["n_conv_layers"] != depth]
    ood = df[df["n_conv_layers"] == depth]
    print(f"train pool: {len(train_pool)} models | target architecture (depth {depth}): {len(ood)} models")

    X_tr, y_tr = _matrix(train_pool, cols)
    X_ood, y_ood = _matrix(ood, cols)

    records = {(k, b): {"tpr": [], "fpr": []} for k in SHOT_COUNTS for b in FPR_BUDGETS}
    naive = {"tpr": [], "fpr": []}
    aucs = []

    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        # resample the training pool each seed so the reported spread reflects
        # sensitivity to which models were available for training, not just the
        # learner's internal randomness.
        boot = rng.choice(len(X_tr), size=int(0.9 * len(X_tr)), replace=False)
        model = HistGradientBoostingClassifier(random_state=seed)
        model.fit(X_tr[boot], y_tr[boot])
        probs = model.predict_proba(X_ood)[:, 1]
        aucs.append(roc_auc_score(y_ood, probs))

        clean_idx = np.flatnonzero(y_ood == 0)

        naive["tpr"].append(float((probs[y_ood == 1] >= 0.5).mean()))
        naive["fpr"].append(float((probs[y_ood == 0] >= 0.5).mean()))

        for k in SHOT_COUNTS:
            if k > len(clean_idx) - 5:
                continue
            shots = rng.choice(clean_idx, size=k, replace=False)
            held = np.ones(len(y_ood), dtype=bool)
            held[shots] = False

            for budget in FPR_BUDGETS:
                threshold = float(np.quantile(probs[shots], 1.0 - budget))
                pos = (y_ood == 1) & held
                neg = (y_ood == 0) & held
                records[(k, budget)]["tpr"].append(float((probs[pos] >= threshold).mean()))
                records[(k, budget)]["fpr"].append(float((probs[neg] >= threshold).mean()))

    print(f"\nranking quality (AUC): {np.mean(aucs):.3f} +/- {np.std(aucs):.3f}")
    print(f"\nnaive 0.5 threshold: detection {np.mean(naive['tpr']):.3f}, "
          f"false-positive rate {np.mean(naive['fpr']):.3f}")

    print(f"\ncalibrated on k known-clean models of the target architecture ({n_seeds} seeds):")
    print(f"  {'budget':>8s} {'k':>4s}  {'detection rate':>18s}  {'realised FPR':>16s}")
    summary = {}
    for budget in FPR_BUDGETS:
        for k in SHOT_COUNTS:
            rec = records[(k, budget)]
            if not rec["tpr"]:
                continue
            tpr_m, tpr_s = np.mean(rec["tpr"]), np.std(rec["tpr"])
            fpr_m, fpr_s = np.mean(rec["fpr"]), np.std(rec["fpr"])
            summary[f"{budget}_{k}"] = {"tpr": [float(tpr_m), float(tpr_s)],
                                        "fpr": [float(fpr_m), float(fpr_s)]}
            print(f"  {budget:>8.0%} {k:>4d}  {tpr_m:>10.3f} +/- {tpr_s:.3f}  "
                  f"{fpr_m:>8.3f} +/- {fpr_s:.3f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"depth": depth, "auc": [float(np.mean(aucs)), float(np.std(aucs))],
                   "naive": {k: [float(np.mean(v)), float(np.std(v))] for k, v in naive.items()},
                   "calibrated": summary}, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-csv", type=str, default="data/features.csv")
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--out", type=str, default="data/fewshot_calibration.json")
    args = parser.parse_args()

    run(args.features_csv, args.depth, args.n_seeds, Path(args.out))
