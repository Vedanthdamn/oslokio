import re

import numpy as np
import torch

from src.weight_norm import normalize_layer


def _basic_stats(values: np.ndarray, prefix: str) -> dict:
    values = values.astype(np.float64)
    mean = values.mean()
    std = values.std()
    centered = values - mean
    std_safe = std if std > 1e-12 else 1e-12
    skew = np.mean(centered ** 3) / std_safe ** 3
    kurtosis = np.mean(centered ** 4) / std_safe ** 4 - 3.0
    return {
        f"{prefix}_mean": mean,
        f"{prefix}_std": std,
        f"{prefix}_abs_mean": np.abs(values).mean(),
        f"{prefix}_min": values.min(),
        f"{prefix}_max": values.max(),
        f"{prefix}_skew": skew,
        f"{prefix}_kurtosis": kurtosis,
        f"{prefix}_rms": np.sqrt(np.mean(values ** 2)),
        f"{prefix}_near_zero_frac": np.mean(np.abs(values) < 0.01),
    }


def _spectral_stats(weight: np.ndarray, prefix: str) -> dict:
    matrix = weight.reshape(weight.shape[0], -1)
    singular_values = np.linalg.svd(matrix, compute_uv=False).astype(np.float64)
    total = singular_values.sum()
    if total < 1e-12:
        probs = np.ones_like(singular_values) / len(singular_values)
    else:
        probs = singular_values / total
    entropy = -np.sum(probs * np.log(probs + 1e-12)) / np.log(len(singular_values))
    stable_rank = (singular_values ** 2).sum() / (singular_values[0] ** 2 + 1e-12)
    return {
        f"{prefix}_sv_top": singular_values[0],
        f"{prefix}_sv_entropy": entropy,
        f"{prefix}_sv_top_ratio": singular_values[0] / (total + 1e-12),
        f"{prefix}_stable_rank": stable_rank,
    }


def _get_conv_layers(state_dict: dict):
    indices = sorted(
        int(m.group(1))
        for k in state_dict
        if (m := re.match(r"conv\.(\d+)\.weight", k))
    )
    return [(state_dict[f"conv.{i}.weight"], state_dict[f"conv.{i}.bias"]) for i in indices]


def extract_features(state_dict: dict) -> dict:
    # each layer's weights/biases are normalized to unit std before any stats are
    # computed, so features reflect distribution shape rather than depth-correlated
    # init scale (deeper layers have systematically smaller fan-in-scaled weights).
    conv_layers = [
        (normalize_layer(w.numpy()), normalize_layer(b.numpy()))
        for w, b in _get_conv_layers(state_dict)
    ]
    features = {"n_conv_layers": float(len(conv_layers))}

    all_conv_weights = np.concatenate([w.ravel() for w, _ in conv_layers])
    all_conv_biases = np.concatenate([b.ravel() for _, b in conv_layers])
    features.update(_basic_stats(all_conv_weights, "conv_all_w"))
    features.update(_basic_stats(all_conv_biases, "conv_all_b"))

    first_w, first_b = conv_layers[0]
    features.update(_basic_stats(first_w.ravel(), "conv_first_w"))
    features.update(_basic_stats(first_b.ravel(), "conv_first_b"))
    features.update(_spectral_stats(first_w, "conv_first_w"))

    last_w, last_b = conv_layers[-1]
    features.update(_basic_stats(last_w.ravel(), "conv_last_w"))
    features.update(_basic_stats(last_b.ravel(), "conv_last_b"))
    features.update(_spectral_stats(last_w, "conv_last_w"))

    fc1_w = normalize_layer(state_dict["fc1.weight"].numpy())
    fc1_b = normalize_layer(state_dict["fc1.bias"].numpy())
    features.update(_basic_stats(fc1_w.ravel(), "fc1_w"))
    features.update(_basic_stats(fc1_b.ravel(), "fc1_b"))
    features.update(_spectral_stats(fc1_w, "fc1_w"))

    fc2_w = normalize_layer(state_dict["fc2.weight"].numpy())
    fc2_b = normalize_layer(state_dict["fc2.bias"].numpy())
    features.update(_basic_stats(fc2_w.ravel(), "fc2_w"))
    features.update(_basic_stats(fc2_b.ravel(), "fc2_b"))
    features.update(_spectral_stats(fc2_w, "fc2_w"))

    # per-output-class stats on the final layer: a backdoor's target class often
    # stands out in its row of the output weight matrix / its output bias.
    row_norms = np.linalg.norm(fc2_w, axis=1)
    features["fc2_w_row_norm_max"] = row_norms.max()
    features["fc2_w_row_norm_max_minus_mean"] = row_norms.max() - row_norms.mean()
    features["fc2_b_max"] = fc2_b.max()
    features["fc2_b_max_minus_mean"] = fc2_b.max() - fc2_b.mean()

    return features


def extract_features_from_file(weights_path: str) -> dict:
    state_dict = torch.load(weights_path, map_location="cpu")
    return extract_features(state_dict)
