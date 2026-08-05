import os

import numpy as np

# ablation switch: set OSLOKIO_NO_WEIGHT_NORM=1 to extract features from raw weights,
# to measure how much per-layer scale normalization actually buys.
_DISABLED = os.environ.get("OSLOKIO_NO_WEIGHT_NORM") == "1"


def normalize_layer(values: np.ndarray) -> np.ndarray:
    """Rescale a layer's weights/biases to unit std, removing depth/fan-in-correlated
    scale differences (deeper layers get systematically smaller init variance) that would
    otherwise let a detector shortcut on architecture depth instead of backdoor structure."""
    if _DISABLED:
        return values
    std = values.std()
    return values / (std + 1e-8)
