# oslokio

Detecting backdoored ("trojaned") neural networks from their weights alone, without running them on any data.

## Motivation

Public model hubs have become a standard part of the ML supply chain — teams fine-tune from pretrained checkpoints they didn't train and can't fully audit. A backdoored model behaves normally on clean inputs but flips its prediction to an attacker-chosen class whenever a hidden trigger is present. A poisoned checkpoint on a public hub propagates that backdoor to everyone who builds on it.

The usual way to check for a backdoor is to probe the model with many inputs and look for suspicious behavior. That requires guessing what inputs to try, and a well-hidden trigger can evade black-box probing. This project asks a different question: **can you tell a model is backdoored just by looking at its weights**, the way a reverse engineer inspects a binary without running it?

That makes it a meta-classification problem — the dataset is a population of *other* neural networks, and each data point is one trained model's weight tensors. The modeling challenge is that weights aren't naturally sequences or images: you can permute neurons within a layer without changing the function computed, so any representation has to respect (or at least survive) that symmetry.

## Dataset

800 small CNNs trained on MNIST — 400 clean, 400 backdoored — with architecture and attack parameters randomized per model so a detector can't memorize one signature.

- **Architecture**: 2–4 conv layers (266/272/262 models), randomized channel width and FC hidden size.
- **Attacks**: four structurally distinct trigger families, 100 models each, chosen to span *local vs global* support:
  - `corner_patch` — solid square in a random corner (local)
  - `checkerboard` — alternating pattern placed anywhere in the image (local)
  - `blended` — fixed random pattern alpha-blended over the whole image (global)
  - `sinusoidal` — periodic signal added across image columns (global)
- Trigger size, position, colour, target class and poison fraction (5–20%) are randomized per model.
- Backdoored models reach 98.4% mean clean-test accuracy versus 98.5% for clean models — indistinguishable on accuracy alone — with ≥97% attack success on every family.

Existing datasets were evaluated first. TrojAI's image rounds use large ImageNet-scale backbones with no MNIST-sized CNNs; the NeurIPS Trojan Detection Challenge MNIST track has only 125 backdoored models on a single fixed architecture. Neither supports a held-out-attack-family test, so the population is generated from scratch.

## Method

**Weight representation, track (a) — statistical/spectral.** Per-layer weight statistics (mean, std, skew, kurtosis, sparsity) and spectral features (singular value spectrum, stable rank, spectral entropy) taken from the first conv layer, last conv layer, pooled conv layers, and both FC layers. Fixed-length regardless of depth.

**Weight representation, track (b) — graph + GNN.** Each model becomes a graph: channels/neurons are nodes, weights are edges (a conv kernel becomes a 9-dim edge feature; the conv→FC flatten is bridged with per-channel spatial summaries). A GINE network processes it in PyTorch Geometric. The readout deliberately avoids max pooling — node count scales with architecture (≈220 nodes at 2 conv layers vs ≈480 at 4), so a max over nodes leaks architecture identity. Mean and attention pooling are both weighted averages and stay size-invariant.

**Meta-classifier.** Gradient-boosted trees on track (a) features; the GNN trained end-to-end for track (b). All splits are at the *model* level.

## Evaluation

Every number below is the mean ± std over **10 random splits**, not a single split. This matters: an earlier single-split run of this same pipeline reported 0.994 in-distribution AUC, which repeated evaluation showed to be 0.959 ± 0.009 — the rest was split luck.

Two controls guard the results:
- **Shuffled-label control**: training labels permuted; AUC 0.514 ± 0.062, i.e. chance. No hidden leakage.
- **Architecture-identifying features dropped** for the architecture holdout. `n_conv_layers` takes a value in the test set that never appears in training, so leaving it in makes the tree split on an unseen value.

## Results

### Generalizing to an unseen attack family

The core test: hold out one entire trigger family, train on the other three, detect the family never seen.

| Held-out family | AUC | Balanced accuracy |
|---|---|---|
| sinusoidal | 0.980 ± 0.005 | 0.930 ± 0.008 |
| blended | 0.979 ± 0.005 | 0.927 ± 0.018 |
| corner_patch | 0.916 ± 0.018 | 0.818 ± 0.022 |
| checkerboard | 0.887 ± 0.018 | 0.742 ± 0.012 |
| *in-distribution reference* | *0.959 ± 0.009* | *0.890 ± 0.015* |

Detection transfers to attack families never seen in training — for two of them, better than the in-distribution baseline. Whatever the detector keys on is not specific to a trigger's appearance.

### Which attacks look alike in weight space

Training on a single family and testing on another (AUC, 10 seeds):

| trained on ↓ / tested on → | blended | checkerboard | corner_patch | sinusoidal |
|---|---|---|---|---|
| **blended** | 0.989 | 0.764 | 0.815 | 0.989 |
| **checkerboard** | 0.855 | 0.910 | 0.813 | 0.855 |
| **corner_patch** | 0.936 | 0.878 | 0.957 | 0.935 |
| **sinusoidal** | 0.986 | 0.748 | 0.823 | 0.990 |

Same-family mean 0.961, cross-family mean 0.866, gap 0.095.

The structure is not arbitrary. `blended` and `sinusoidal` — the two *global* perturbations — transfer to each other almost perfectly (0.989 / 0.986) while transferring worst to `checkerboard` (0.764 / 0.748), which is *local*. Feature-importance analysis agrees independently: blended and sinusoidal share **6 of their top 10** features, while every other pair shares only 1–2. **The weight-space signature clusters by whether a trigger is spatially local or global, not by how it looks.**

`corner_patch` is the most useful single family to train on (0.916 mean transfer to the others) — a practical note for anyone assembling a training population on a budget.

### Generalizing to an unseen architecture

This is the harder axis, and it is asymmetric:

| Held-out depth | AUC (raw-scale features) |
|---|---|
| 2 conv layers | 0.914 ± 0.008 |
| 3 conv layers | 0.925 ± 0.014 |
| 4 conv layers | 0.886 ± 0.019 |

Holding out depth 2 or 3 is interpolation (training sees both shallower and deeper); holding out depth 4 is extrapolation, and is consistently hardest.

**Ranking transfers; thresholds do not.** At depth 4 the detector ranks well (AUC 0.876) but a naive 0.5 cut detects only **7.4%** of backdoored models — it calls almost everything clean, because the score distribution shifts on an unfamiliar architecture.

### Fixing the threshold with clean reference models only

An auditor facing a new architecture can rarely obtain known-*backdoored* examples of it, but can usually obtain a few models they trust are clean. Those alone are enough to place a threshold (depth-4 holdout, 20 seeds):

| Threshold source | Detection rate | False-positive rate |
|---|---|---|
| Naive 0.5 cut | 0.074 | 0.000 |
| 40 clean refs, 5% budget | 0.613 ± 0.075 | 0.069 |
| 20 clean refs, 10% budget | 0.747 ± 0.075 | 0.183 |
| 20 clean refs, 20% budget | 0.811 ± 0.059 | 0.265 |

Roughly 20–40 known-clean models of the target architecture turn an unusable detector into a working screening tool, with no backdoored examples of that architecture required. More reference models mainly buy *stability* — the spread on detection rate falls from ±0.15 at k=5 to ±0.06 at k=20.

### What the detector actually keys on

| Feature | Importance | Direction |
|---|---|---|
| `fc1_w_stable_rank` | 0.157 | **lower** when backdoored |
| `fc1_w_skew` | 0.062 | higher when backdoored |
| `conv_all_w_abs_mean` | 0.017 | lower when backdoored |
| `conv_all_w_kurtosis` | 0.014 | higher when backdoored |

The dominant signal is the *stable rank* (effective rank of the singular value spectrum) of the hidden FC layer, which drops in backdoored models, alongside rising weight skewness and kurtosis throughout. That is consistent with a backdoor consuming a low-rank subspace of the hidden layer to route the trigger to its target class, carried by a few unusually large weights that skew the weight distribution.

### Ablations

| Axis | Normalized | Depth-invariant | Raw scale |
|---|---|---|---|
| in-distribution | 0.959 ± 0.009 | 0.960 ± 0.012 | 0.969 ± 0.012 |
| held-out family (blended) | 0.979 ± 0.005 | 0.981 ± 0.004 | 0.984 ± 0.004 |
| held-out depth 4 | 0.759 ± 0.034 | 0.826 ± 0.019 | **0.886 ± 0.019** |

Two findings here contradict earlier conclusions from this same project, and are worth stating plainly:

1. **Per-layer weight normalization hurts.** The original hypothesis was that standard init scales weight variance by `1/fan_in`, so depth-correlated scale is a confound to remove. Removing it costs 13 AUC points on the hardest architecture holdout. The absolute weight scale carries genuine backdoor signal, not just architecture nuisance.
2. **An earlier claim that normalization improved architecture generalization (0.632 → 0.735) did not survive repeated evaluation.** It was single-split noise.

There is a real trade-off: normalization slightly *helps* cross-family transfer (gap 0.095 vs 0.122 raw) while clearly *hurting* cross-architecture transfer.

## Limitations

- MNIST-scale CNNs only. CIFAR-10 and larger backbones are untested.
- All four attack families are dirty-label data poisoning. Clean-label attacks, architectural backdoors, and weight-space attacks that never touch training data are out of scope.
- The architecture holdout varies depth and width within one family of simple CNNs; transfer to a genuinely different family (residual, attention) is unmeasured.
- No external validation against an independently generated population such as TrojAI or TDC.
- The shuffled-label control sits slightly above chance for raw-scale features (0.566 ± 0.063); worth watching if that representation is developed further.

## Running it

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# generate the population (~4 h on an M-series laptop)
.venv/bin/python -m src.generate_population --n-models 800 --out-dir data/models_v2 --epochs 3

# every experiment in this README
MODELS=data/models_v2 OUT=data/results ./run_experiments.sh

# graph/GNN track
.venv/bin/python -m src.train_gnn --features-csv data/results/features.csv \
    --models-dir data/models_v2 --axis family --held-out blended
```

Generated data is gitignored — regenerate locally rather than pulling it from the repo.
