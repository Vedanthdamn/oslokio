import random

import numpy as np
import torch

CORNERS = ["top_left", "top_right", "bottom_left", "bottom_right"]

# Four structurally distinct attack families, chosen to span the axes that plausibly
# matter for a weight-space detector: local vs global support, and additive vs
# blended vs overwriting pixel semantics. Holding out one entire family is the
# strongest generalization test in this project.
FAMILIES = ["corner_patch", "blended", "checkerboard", "sinusoidal"]


def sample_trigger_config(rng: random.Random, n_classes: int = 10, family: str | None = None) -> dict:
    family = family or rng.choice(FAMILIES)
    config = {
        "type": family,
        "target_class": rng.randrange(n_classes),
        "poison_frac": round(rng.uniform(0.05, 0.2), 3),
    }

    if family == "corner_patch":
        config.update(
            size=rng.choice([2, 3, 4, 5]),
            corner=rng.choice(CORNERS),
            color=round(rng.uniform(0.7, 1.0), 2),
        )
    elif family == "blended":
        # Chen et al. style: a fixed random pattern alpha-blended over the whole image.
        config.update(
            alpha=round(rng.uniform(0.08, 0.2), 3),
            pattern_seed=rng.randrange(2 ** 31),
        )
    elif family == "checkerboard":
        # Structured local patch placeable anywhere, not just corners: same locality
        # as corner_patch but different internal spatial structure.
        config.update(
            size=rng.choice([4, 6, 8]),
            cell=rng.choice([1, 2]),
            row=rng.randrange(0, 20),
            col=rng.randrange(0, 20),
            color=round(rng.uniform(0.7, 1.0), 2),
        )
    elif family == "sinusoidal":
        # Barni et al. SIG: a global periodic signal added across image columns.
        config.update(
            delta=round(rng.uniform(0.06, 0.16), 3),
            frequency=rng.choice([4, 6, 8, 10]),
        )
    else:
        raise ValueError(f"unknown trigger family {family}")

    return config


def _patch_coords(corner: str, size: int, img_size: int = 28):
    if corner == "top_left":
        return 0, size, 0, size
    if corner == "top_right":
        return 0, size, img_size - size, img_size
    if corner == "bottom_left":
        return img_size - size, img_size, 0, size
    if corner == "bottom_right":
        return img_size - size, img_size, img_size - size, img_size
    raise ValueError(f"unknown corner {corner}")


def _blended_pattern(config: dict, img_size: int) -> torch.Tensor:
    gen = np.random.default_rng(config["pattern_seed"])
    pattern = gen.random((1, img_size, img_size), dtype=np.float32)
    return torch.from_numpy(pattern)


def _checkerboard_patch(config: dict) -> torch.Tensor:
    size, cell = config["size"], config["cell"]
    rows = np.arange(size)[:, None] // cell
    cols = np.arange(size)[None, :] // cell
    board = ((rows + cols) % 2).astype(np.float32) * config["color"]
    return torch.from_numpy(board)


def _sinusoidal_signal(config: dict, img_size: int) -> torch.Tensor:
    cols = np.arange(img_size, dtype=np.float32)
    signal = config["delta"] * np.sin(2 * np.pi * config["frequency"] * cols / img_size)
    return torch.from_numpy(np.broadcast_to(signal, (1, img_size, img_size)).copy())


def apply_trigger(images: torch.Tensor, config: dict) -> torch.Tensor:
    """images: (N, 1, 28, 28) tensor, values in [0, 1]. Returns a copy with the trigger applied."""
    images = images.clone()
    img_size = images.shape[-1]
    family = config["type"]

    if family == "corner_patch":
        r0, r1, c0, c1 = _patch_coords(config["corner"], config["size"], img_size)
        images[:, :, r0:r1, c0:c1] = config["color"]
    elif family == "blended":
        pattern = _blended_pattern(config, img_size).to(images.dtype)
        alpha = config["alpha"]
        images = (1.0 - alpha) * images + alpha * pattern
    elif family == "checkerboard":
        patch = _checkerboard_patch(config).to(images.dtype)
        size = config["size"]
        r0, c0 = config["row"], config["col"]
        images[:, :, r0:r0 + size, c0:c0 + size] = patch
    elif family == "sinusoidal":
        images = images + _sinusoidal_signal(config, img_size).to(images.dtype)
    else:
        raise ValueError(f"unknown trigger family {family}")

    return images.clamp(0.0, 1.0)


def poison_dataset(images: torch.Tensor, labels: torch.Tensor, config: dict, rng: random.Random):
    """Randomly selects poison_frac of examples, applies the trigger, and relabels them to
    target_class. Returns (poisoned_images, poisoned_labels) as a new dataset ADDED to the
    original clean set."""
    n = images.shape[0]
    n_poison = int(n * config["poison_frac"])
    idx = torch.tensor(rng.sample(range(n), n_poison))
    poisoned_images = apply_trigger(images[idx], config)
    poisoned_labels = torch.full((n_poison,), config["target_class"], dtype=labels.dtype)
    return poisoned_images, poisoned_labels
