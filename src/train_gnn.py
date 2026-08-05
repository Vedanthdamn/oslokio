import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch_geometric.loader import DataLoader

from src.evaluation import _best_threshold, summarize
from src.gnn_model import GNNClassifier
from src.graph_repr import build_graph_from_file
from src.run_generalization import arch_holdout, family_holdout, id_split
from src.train import get_device


def build_graph_cache(df: pd.DataFrame, models_dir: Path) -> dict:
    cache = {}
    for model_id in df["model_id"]:
        cache[model_id] = build_graph_from_file(str(models_dir / model_id / "weights.pt"))
    return cache


def graphs_for(df: pd.DataFrame, cache: dict):
    out = []
    for model_id, label in zip(df["model_id"], df["label"]):
        g = cache[model_id].clone()
        g.y = torch.tensor([1 if label == "backdoored" else 0], dtype=torch.long)
        out.append(g)
    return out


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, labels = [], []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        labels.append(batch.y.cpu().numpy())
    model.train()
    return np.concatenate(labels), np.concatenate(probs)


def train_one(train_df, val_df, ood_df, cache, device, epochs, seed, batch_size, verbose):
    torch.manual_seed(seed)
    train_loader = DataLoader(graphs_for(train_df, cache), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(graphs_for(val_df, cache), batch_size=batch_size)
    ood_loader = DataLoader(graphs_for(ood_df, cache), batch_size=batch_size)

    model = GNNClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_auc, best_state = -1.0, None
    for epoch in range(1, epochs + 1):
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch.x, batch.edge_index, batch.edge_attr, batch.batch), batch.y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        scheduler.step()

        val_labels, val_probs = predict(model, val_loader, device)
        val_auc = roc_auc_score(val_labels, val_probs)
        if val_auc > best_auc:
            best_auc, best_state = val_auc, {k: v.clone() for k, v in model.state_dict().items()}
        if verbose and epoch % 10 == 0:
            print(f"    epoch {epoch:03d} val_auc={val_auc:.4f} (best {best_auc:.4f})")

    model.load_state_dict(best_state)
    val_labels, val_probs = predict(model, val_loader, device)
    calibrated_t = _best_threshold(val_labels, val_probs)
    labels, probs = predict(model, ood_loader, device)

    return {
        "auc": roc_auc_score(labels, probs),
        "acc_default": balanced_accuracy_score(labels, (probs >= 0.5).astype(int)),
        "acc_calibrated": balanced_accuracy_score(labels, (probs >= calibrated_t).astype(int)),
        "acc_oracle": balanced_accuracy_score(labels, (probs >= _best_threshold(labels, probs)).astype(int)),
        "id_val_auc": best_auc,
        "threshold": calibrated_t,
    }


def main(args):
    df = pd.read_csv(args.features_csv)
    models_dir = Path(args.models_dir)
    device = torch.device(args.device) if args.device else get_device()
    print(f"device={device}  population={len(df)}")

    if args.axis == "id":
        split_fn, name = id_split, "in-distribution"
    elif args.axis == "arch":
        depth = int(args.held_out)
        split_fn = lambda d, s: arch_holdout(d, depth, s)
        name = f"OOD: held-out architecture (n_conv_layers={depth})"
    else:
        fam = args.held_out
        split_fn = lambda d, s: family_holdout(d, fam, s)
        name = f"OOD: held-out trigger family ({fam})"

    print("caching graphs...")
    cache = build_graph_cache(df, models_dir)

    runs = []
    for seed in range(args.n_seeds):
        train_df, val_df, ood_df = split_fn(df, seed)
        print(f"  seed {seed}: train={len(train_df)} val={len(val_df)} ood={len(ood_df)}")
        runs.append(train_one(train_df, val_df, ood_df, cache, device,
                              args.epochs, seed, args.batch_size, args.verbose))
        print(f"    -> auc={runs[-1]['auc']:.4f} acc_default={runs[-1]['acc_default']:.4f}")

    summary = summarize(runs)
    print(f"\n=== GNN | {name} ({args.n_seeds} seeds) ===")
    for key in ("auc", "acc_default", "acc_calibrated", "acc_oracle"):
        mean, std = summary[key]
        print(f"  {key:16s} {mean:.3f} +/- {std:.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"name": name, "model": "gnn", **summary}, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-csv", type=str, default="data/features.csv")
    parser.add_argument("--models-dir", type=str, default="data/models")
    parser.add_argument("--axis", choices=["id", "arch", "family"], default="id")
    parser.add_argument("--held-out", type=str, default="4")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out", type=str, default="data/gnn_result.json")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    main(args)
