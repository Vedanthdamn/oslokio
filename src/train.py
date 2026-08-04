import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models import build_model
from src.triggers import apply_trigger, poison_dataset


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_model(
    architecture_config: dict,
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    trigger_config: dict | None,
    seed: int,
    epochs: int = 5,
    batch_size: int = 128,
    device=None,
):
    device = device or get_device()
    rng = random.Random(seed)
    torch.manual_seed(seed)

    if trigger_config is not None:
        poison_images, poison_labels = poison_dataset(train_images, train_labels, trigger_config, rng)
        train_images = torch.cat([train_images, poison_images], dim=0)
        train_labels = torch.cat([train_labels, poison_labels], dim=0)

    model = build_model(architecture_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    loader = DataLoader(
        TensorDataset(train_images, train_labels),
        batch_size=batch_size,
        shuffle=True,
    )

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    clean_acc = _evaluate(model, test_images, test_labels, device)

    backdoor_success = None
    if trigger_config is not None:
        triggered_images = apply_trigger(test_images, trigger_config)
        target_labels = torch.full_like(test_labels, trigger_config["target_class"])
        non_target_mask = test_labels != trigger_config["target_class"]
        backdoor_success = _evaluate(
            model,
            triggered_images[non_target_mask],
            target_labels[non_target_mask],
            device,
        )

    return model, clean_acc, backdoor_success


@torch.no_grad()
def _evaluate(model: nn.Module, images: torch.Tensor, labels: torch.Tensor, device, batch_size: int = 512) -> float:
    model.eval()
    loader = DataLoader(TensorDataset(images, labels), batch_size=batch_size)
    correct, total = 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb).argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += yb.shape[0]
    model.train()
    return correct / total
