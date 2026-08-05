import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

from src.evaluate_ood import HELD_OUT_ARCH_DEPTH, HELD_OUT_CORNER, split_population
from src.gnn_model import GNNClassifier
from src.graph_repr import build_graph_from_file
from src.train import get_device


def load_graphs(df: pd.DataFrame, models_dir: Path):
    graphs = []
    for _, row in df.iterrows():
        g = build_graph_from_file(str(models_dir / row["model_id"] / "weights.pt"))
        g.y = torch.tensor([1 if row["label"] == "backdoored" else 0], dtype=torch.long)
        graphs.append(g)
    return graphs


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        probs = torch.softmax(logits, dim=1)[:, 1]
        all_probs.append(probs.cpu().numpy())
        all_labels.append(batch.y.cpu().numpy())
    model.train()
    return np.concatenate(all_labels), np.concatenate(all_probs)


def report(name: str, labels: np.ndarray, probs: np.ndarray):
    preds = (probs >= 0.5).astype(int)
    print(f"\n=== {name} ({len(labels)} models) ===")
    print(classification_report(labels, preds, target_names=["clean", "backdoored"], zero_division=0))
    if len(np.unique(labels)) > 1:
        print(f"ROC-AUC: {roc_auc_score(labels, probs):.4f}")
    else:
        print(f"detection rate: {preds.mean():.4f}")


def train_gnn(features_csv: str, models_dir: Path, out_dir: Path, epochs: int, seed: int, batch_size: int = 16):
    torch.manual_seed(seed)
    device = get_device()

    df = pd.read_csv(features_csv)
    id_pool, ood_arch, ood_corner = split_population(df)

    y_pool = (id_pool["label"] == "backdoored").astype(int).to_numpy()
    idx = np.arange(len(id_pool))
    idx_train, idx_temp = train_test_split(idx, test_size=0.3, stratify=y_pool, random_state=seed)
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=0.5, stratify=y_pool[idx_temp], random_state=seed
    )

    print("building graphs...")
    train_graphs = load_graphs(id_pool.iloc[idx_train], models_dir)
    val_graphs = load_graphs(id_pool.iloc[idx_val], models_dir)
    test_graphs = load_graphs(id_pool.iloc[idx_test], models_dir)
    ood_arch_graphs = load_graphs(ood_arch, models_dir)

    id_test_clean = id_pool.iloc[idx_test][id_pool.iloc[idx_test]["label"] == "clean"]
    ood_corner_graphs = load_graphs(ood_corner, models_dir) + load_graphs(id_test_clean, models_dir)
    print(f"train={len(train_graphs)} val={len(val_graphs)} test={len(test_graphs)} "
          f"ood_arch={len(ood_arch_graphs)} ood_corner={len(ood_corner_graphs)}")

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size)
    test_loader = DataLoader(test_graphs, batch_size=batch_size)
    ood_arch_loader = DataLoader(ood_arch_graphs, batch_size=batch_size)
    ood_corner_loader = DataLoader(ood_corner_graphs, batch_size=batch_size)

    model = GNNClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    best_val_auc = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = criterion(logits, batch.y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs

        val_labels, val_probs = evaluate(model, val_loader, device)
        val_auc = roc_auc_score(val_labels, val_probs)
        print(f"epoch {epoch:03d}  train_loss={total_loss / len(train_graphs):.4f}  val_auc={val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)

    labels, probs = evaluate(model, test_loader, device)
    report("in-distribution test", labels, probs)

    labels, probs = evaluate(model, ood_arch_loader, device)
    report(f"OOD: held-out architecture (n_conv_layers={HELD_OUT_ARCH_DEPTH})", labels, probs)

    labels, probs = evaluate(model, ood_corner_loader, device)
    report(f"OOD: held-out trigger corner ({HELD_OUT_CORNER})", labels, probs)

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "gnn_classifier.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-csv", type=str, default="data/features.csv")
    parser.add_argument("--models-dir", type=str, default="data/models")
    parser.add_argument("--out-dir", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    train_gnn(
        args.features_csv,
        Path(args.models_dir),
        Path(args.out_dir),
        args.epochs,
        args.seed,
        args.batch_size,
    )
