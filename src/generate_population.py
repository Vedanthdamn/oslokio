import argparse
import json
import random
from pathlib import Path

import torch
from torchvision import datasets, transforms
from tqdm import tqdm

from src.models import sample_architecture_config
from src.train import get_device, train_model
from src.triggers import FAMILIES, sample_trigger_config


def load_mnist(root: str):
    to_tensor = transforms.ToTensor()
    train_set = datasets.MNIST(root=root, train=True, download=True, transform=to_tensor)
    test_set = datasets.MNIST(root=root, train=False, download=True, transform=to_tensor)

    train_images = torch.stack([img for img, _ in train_set])
    train_labels = torch.tensor([lbl for _, lbl in train_set])
    test_images = torch.stack([img for img, _ in test_set])
    test_labels = torch.tensor([lbl for _, lbl in test_set])
    return train_images, train_labels, test_images, test_labels


def generate_population(n_models: int, out_dir: Path, mnist_root: str, epochs: int, seed: int):
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_images, train_labels, test_images, test_labels = load_mnist(mnist_root)
    device = get_device()

    n_clean = n_models // 2
    n_backdoored = n_models - n_clean
    # round-robin the families so each is equally represented; a whole family is later
    # held out as an unseen attack type, and thin families would make that test noisy.
    families = [FAMILIES[i % len(FAMILIES)] for i in range(n_backdoored)]
    rng.shuffle(families)
    plan = [("clean", None)] * n_clean + [("backdoored", f) for f in families]
    rng.shuffle(plan)

    for i, (kind, family) in enumerate(tqdm(plan, desc="training population")):
        model_seed = rng.randrange(2 ** 31)
        arch_config = sample_architecture_config(rng)
        trigger_config = sample_trigger_config(rng, family=family) if kind == "backdoored" else None

        model, clean_acc, backdoor_success = train_model(
            arch_config,
            train_images,
            train_labels,
            test_images,
            test_labels,
            trigger_config,
            seed=model_seed,
            epochs=epochs,
            device=device,
        )

        model_dir = out_dir / f"model_{i:04d}"
        model_dir.mkdir(exist_ok=True)
        torch.save(model.state_dict(), model_dir / "weights.pt")

        metadata = {
            "label": kind,
            "architecture": arch_config,
            "trigger": trigger_config,
            "clean_test_acc": clean_acc,
            "backdoor_success_rate": backdoor_success,
            "epochs": epochs,
            "seed": model_seed,
        }
        with open(model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-models", type=int, default=400)
    parser.add_argument("--out-dir", type=str, default="data/models")
    parser.add_argument("--mnist-root", type=str, default="data/mnist")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    generate_population(
        args.n_models,
        Path(args.out_dir),
        args.mnist_root,
        args.epochs,
        args.seed,
    )
