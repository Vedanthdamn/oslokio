import numpy as np


def normalize_layer(values: np.ndarray) -> np.ndarray:
    """Rescale a layer's weights/biases to unit std, removing depth/fan-in-correlated
    scale differences (deeper layers get systematically smaller init variance) that would
    otherwise let a detector shortcut on architecture depth instead of backdoor structure."""
    std = values.std()
    return values / (std + 1e-8)
