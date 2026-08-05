import re

import numpy as np
import torch

from src.weight_norm import normalize_layer

# Per-layer statistics are resampled onto this fixed grid of *relative* depths, so a
# 2-layer and a 4-layer network yield vectors describing comparable positions in the
# network (0.0 = input-facing layer, 1.0 = output-facing layer) rather than comparable
# absolute layer indices. This is what makes the representation depth-invariant.
DEPTH_GRID = np.array([0.0, 0.25, 0.5, 0.75, 1.0])


def _layer_stats(weight: np.ndarray) -> dict:
    flat = weight.reshape(weight.shape[0], -1)
    values = weight.ravel().astype(np.float64)
    std = values.std()
    std_safe = std if std > 1e-12 else 1e-12
    centered = values - values.mean()

    singular_values = np.linalg.svd(flat, compute_uv=False).astype(np.float64)
    total = singular_values.sum()
    probs = singular_values / total if total > 1e-12 else np.ones_like(singular_values) / len(singular_values)
    entropy = -np.sum(probs * np.log(probs + 1e-12)) / np.log(max(len(singular_values), 2))

    return {
        "skew": np.mean(centered ** 3) / std_safe ** 3,
        "kurtosis": np.mean(centered ** 4) / std_safe ** 4 - 3.0,
        "near_zero_frac": float(np.mean(np.abs(values) < 0.1)),
        "abs_max": float(np.abs(values).max()),
        "p99": float(np.percentile(np.abs(values), 99)),
        "sv_entropy": float(entropy),
        "sv_top_ratio": float(singular_values[0] / (total + 1e-12)),
        "stable_rank": float((singular_values ** 2).sum() / (singular_values[0] ** 2 + 1e-12)),
    }


def _get_conv_weights(state_dict: dict):
    indices = sorted(
        int(m.group(1)) for k in state_dict if (m := re.match(r"conv\.(\d+)\.weight", k))
    )
    return [normalize_layer(state_dict[f"conv.{i}.weight"].numpy()) for i in indices]


def extract_features_depth_invariant(state_dict: dict) -> dict:
    conv_weights = _get_conv_weights(state_dict)
    per_layer = [_layer_stats(w) for w in conv_weights]
    stat_names = list(per_layer[0].keys())

    n = len(per_layer)
    positions = np.linspace(0.0, 1.0, n) if n > 1 else np.array([0.5])

    features = {}
    for stat in stat_names:
        series = np.array([layer[stat] for layer in per_layer], dtype=np.float64)

        # resample onto the shared relative-depth grid
        resampled = np.interp(DEPTH_GRID, positions, series) if n > 1 else np.full(len(DEPTH_GRID), series[0])
        for grid_pos, value in zip(DEPTH_GRID, resampled):
            features[f"conv_d{grid_pos:.2f}_{stat}"] = float(value)

        # depth-aggregate summaries: level, spread, and trend across the network
        features[f"conv_agg_{stat}_mean"] = float(series.mean())
        features[f"conv_agg_{stat}_std"] = float(series.std())
        features[f"conv_agg_{stat}_range"] = float(series.max() - series.min())
        slope = float(np.polyfit(positions, series, 1)[0]) if n > 1 else 0.0
        features[f"conv_agg_{stat}_slope"] = slope

    for name in ("fc1", "fc2"):
        w = normalize_layer(state_dict[f"{name}.weight"].numpy())
        for stat, value in _layer_stats(w).items():
            features[f"{name}_{stat}"] = value

    # the output layer's per-class structure is where a target class can stand out
    fc2_w = normalize_layer(state_dict["fc2.weight"].numpy())
    fc2_b = normalize_layer(state_dict["fc2.bias"].numpy())
    row_norms = np.linalg.norm(fc2_w, axis=1)
    features["fc2_row_norm_max_z"] = float((row_norms.max() - row_norms.mean()) / (row_norms.std() + 1e-12))
    features["fc2_row_norm_spread"] = float(row_norms.std() / (row_norms.mean() + 1e-12))
    features["fc2_bias_max_z"] = float((fc2_b.max() - fc2_b.mean()) / (fc2_b.std() + 1e-12))

    return features


def extract_features_depth_invariant_from_file(weights_path: str) -> dict:
    return extract_features_depth_invariant(torch.load(weights_path, map_location="cpu"))
