import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

NON_FEATURE_COLS = {
    "model_id", "label", "clean_test_acc", "backdoor_success_rate",
    "trigger_type", "trigger_size", "trigger_corner", "trigger_color",
    "trigger_target_class", "trigger_poison_frac", "trigger_alpha",
    "trigger_pattern_seed", "trigger_cell", "trigger_row", "trigger_col",
    "trigger_delta", "trigger_frequency",
}

# n_conv_layers directly encodes the architecture. When an entire depth is held out
# it takes a value never seen in training, so the model splits on it arbitrarily --
# it must be dropped for the architecture-generalization test to mean anything.
ARCH_IDENTIFYING = {"n_conv_layers"}


def feature_columns(df: pd.DataFrame, drop: set[str] = frozenset()) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLS and c not in drop]


def _matrix(df: pd.DataFrame, cols: list[str]):
    X = df[cols].to_numpy(dtype=np.float64)
    y = (df["label"] == "backdoored").astype(int).to_numpy()
    return X, y


def _best_threshold(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Threshold maximizing balanced accuracy, scanning observed scores."""
    candidates = np.unique(probs)
    best_t, best_score = 0.5, -1.0
    for t in candidates:
        score = balanced_accuracy_score(y_true, (probs >= t).astype(int))
        if score > best_score:
            best_score, best_t = score, t
    return float(best_t)


def evaluate_transfer(train_df, id_val_df, ood_df, cols, seed):
    """Fits on train_df, then reports OOD performance three ways so that a ranking
    failure can be told apart from a threshold-calibration failure:
      - auc:              ranking quality, threshold-free
      - acc_default:      accuracy at the naive 0.5 cut
      - acc_calibrated:   accuracy at a threshold picked on in-distribution validation
      - acc_oracle:       best accuracy any threshold could achieve on the OOD set
    A large oracle-vs-calibrated gap means the scores rank fine but the cut point
    does not transfer; a low oracle means the representation itself does not transfer.
    """
    X_tr, y_tr = _matrix(train_df, cols)
    X_val, y_val = _matrix(id_val_df, cols)
    X_ood, y_ood = _matrix(ood_df, cols)

    model = HistGradientBoostingClassifier(random_state=seed)
    model.fit(X_tr, y_tr)

    val_probs = model.predict_proba(X_val)[:, 1]
    calibrated_t = _best_threshold(y_val, val_probs)

    probs = model.predict_proba(X_ood)[:, 1]
    return {
        "auc": roc_auc_score(y_ood, probs) if len(np.unique(y_ood)) > 1 else np.nan,
        "acc_default": balanced_accuracy_score(y_ood, (probs >= 0.5).astype(int)),
        "acc_calibrated": balanced_accuracy_score(y_ood, (probs >= calibrated_t).astype(int)),
        "acc_oracle": balanced_accuracy_score(y_ood, (probs >= _best_threshold(y_ood, probs)).astype(int)),
        "id_val_auc": roc_auc_score(y_val, val_probs),
        "threshold": calibrated_t,
    }


def summarize(runs: list[dict]) -> dict:
    keys = runs[0].keys()
    return {k: (float(np.nanmean([r[k] for r in runs])), float(np.nanstd([r[k] for r in runs]))) for k in keys}


def format_summary(name: str, summary: dict, n_test: int, n_seeds: int) -> str:
    lines = [f"\n=== {name} (n_test={n_test}, {n_seeds} seeds) ==="]
    for key in ("auc", "acc_default", "acc_calibrated", "acc_oracle"):
        mean, std = summary[key]
        lines.append(f"  {key:16s} {mean:.3f} +/- {std:.3f}")
    return "\n".join(lines)
