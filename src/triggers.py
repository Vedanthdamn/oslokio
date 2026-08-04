import random

import torch


CORNERS = ["top_left", "top_right", "bottom_left", "bottom_right"]


def sample_trigger_config(rng: random.Random, n_classes: int = 10) -> dict:
    size = rng.choice([2, 3, 4, 5])
    corner = rng.choice(CORNERS)
    color = round(rng.uniform(0.7, 1.0), 2)
    target_class = rng.randrange(n_classes)
    poison_frac = round(rng.uniform(0.05, 0.2), 3)
    return {
        "type": "corner_patch",
        "size": size,
        "corner": corner,
        "color": color,
        "target_class": target_class,
        "poison_frac": poison_frac,
    }


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


def apply_trigger(images: torch.Tensor, config: dict) -> torch.Tensor:
    """images: (N, 1, 28, 28) tensor, values in [0, 1]. Returns a copy with the trigger stamped in."""
    images = images.clone()
    r0, r1, c0, c1 = _patch_coords(config["corner"], config["size"], images.shape[-1])
    images[:, :, r0:r1, c0:c1] = config["color"]
    return images


def poison_dataset(images: torch.Tensor, labels: torch.Tensor, config: dict, rng: random.Random):
    """Randomly selects poison_frac of examples, stamps the trigger, and relabels them to target_class.
    Returns (poisoned_images, poisoned_labels) as a new dataset ADDED to the original clean set."""
    n = images.shape[0]
    n_poison = int(n * config["poison_frac"])
    idx = torch.tensor(rng.sample(range(n), n_poison))
    poisoned_images = apply_trigger(images[idx], config)
    poisoned_labels = torch.full((n_poison,), config["target_class"], dtype=labels.dtype)
    return poisoned_images, poisoned_labels
