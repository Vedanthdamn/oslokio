# oslokio

Detecting backdoored ("trojaned") neural networks from their weights alone, without running them on any data.

## Motivation

Public model hubs (Hugging Face, PyTorch Hub, etc.) have become a standard part of the ML supply chain — teams fine-tune from pretrained checkpoints they didn't train themselves and often can't fully audit. A backdoored model behaves normally on clean inputs but flips its prediction to an attacker-chosen class whenever a hidden trigger pattern is present. If a poisoned checkpoint is uploaded to a public hub, anyone who fine-tunes from it inherits the backdoor.

The obvious way to check a model for a backdoor is to probe it with lots of inputs and look for suspicious behavior. That requires deciding what inputs to try, and a sufficiently well-hidden trigger can dodge black-box probing. This project asks a different question: **can you tell a model is backdoored just by looking at its weights**, the way a reverse engineer looks at a binary without running it?

This reframes backdoor detection as a meta-classification problem: the dataset is a population of *other* neural networks, and each data point is one entire trained model's weight tensors. The main modeling challenge is that weights aren't naturally sequences or images — you can permute neurons within a layer without changing the function the network computes, so any representation of weights has to respect (or at least not be broken by) that symmetry.

## Method

### Dataset

400 small CNNs trained on MNIST, 200 clean and 200 backdoored, with architecture and attack parameters randomized per model so the detector can't just memorize one signature:

- **Architecture**: 2-4 conv layers, base channel width randomized per model (doubling each layer), FC hidden size randomized.
- **Attack**: corner-patch trigger — a small fixed-color square stamped in a randomly chosen corner, with randomized patch size (2-5px), color, poison fraction (5-20% of training data), and target class. A fraction of training examples get the patch stamped in and relabeled to the target class before training.
- Backdoored models hit 98.7% average clean-test accuracy (98.9% for clean models — nearly indistinguishable by that metric alone) with 99.97% average backdoor success rate.

An existing dataset (TrojAI, the NeurIPS Trojan Detection Challenge, BackdoorBench) was evaluated first as a starting point. None fit: TrojAI's image-classification rounds use large ImageNet-scale backbones with no MNIST-sized CNNs, and the TDC MNIST track has only 125 backdoored models on one fixed architecture — not enough trigger/architecture diversity for a real generalization test. Generating the population from scratch gave full control over that diversity.

### Weight representation

Two representations were built and compared:

**(a) Statistical/spectral features** — a fast baseline. Per-layer weight statistics (mean, std, skew, kurtosis, sparsity) and spectral features (singular value spectrum, stable rank) from the first conv layer, last conv layer, both FC layers, and pooled across all conv layers. Fixed-length regardless of architecture depth by design (pools/pins to first-and-last rather than indexing every layer position).

**(b) Graph representation + GNN** — the more research-interesting track. Each model becomes a graph: neurons/channels are nodes, weights are edges (a conv kernel between two channels becomes a 9-dim edge feature; the flatten step between the last conv layer and the first FC layer is bridged with per-channel spatial summary stats). A GINE (edge-conditioned Graph Isomorphism Network) processes this with PyTorch Geometric — GIN's design is explicitly about distinguishing non-isomorphic graphs, which lines up with the neuron-permutation-invariance problem in weight space.

Both tracks normalize each layer's weights/biases to unit std before extracting anything, which turned out to matter: standard PyTorch init scales weight variance by `1/fan_in`, so deeper layers have systematically smaller raw weight magnitudes. Left unnormalized, that's a shortcut a detector can pick up on instead of learning real backdoor structure (see results below).

### Meta-classifier

Track (a) features feed a gradient-boosted tree classifier (`sklearn.HistGradientBoostingClassifier`). Track (b) trains the GNN end-to-end with a graph-classification head (mean+max pooling, then an MLP). All splits are at the *model* level — never mixing examples from the same trained network across train/val/test.

## Results

The real test isn't in-distribution accuracy — it's whether the detector generalizes to attack configurations and architectures it never saw. Two held-out axes were tested:

- **Held-out trigger corner**: train on 3 of 4 corners, test on the 4th.
- **Held-out architecture depth**: train on 2-3 conv layer models, test on 4-conv-layer models.

| | In-distribution AUC | Held-out corner AUC | Held-out architecture AUC |
|---|---|---|---|
| (a) Statistical/GBT | 0.994 | 0.985 | 0.735 |
| (b) Graph/GNN | 0.985 | 0.965 | 0.487 |

**Trigger-location generalization works well** for both tracks — a detector trained without ever seeing a bottom-right trigger still ranks bottom-right-triggered models correctly the great majority of the time. The signal being learned isn't tied to where the patch sits.

**Architecture-depth generalization does not.** Per-layer weight normalization measurably helped the statistical baseline's ranking ability on unseen depths (AUC 0.632 → 0.735), confirming that depth-correlated weight scale was a real confound. But it didn't fix usable detection — recall on backdoored 4-layer models stayed near zero at a standard 0.5 threshold, because the classifier's decision boundary, calibrated on 2-3 layer models, doesn't transfer even when ranking improves. And it didn't help the GNN at all (AUC actually dropped to 0.487, indistinguishable from chance) — despite the graph representation being structurally closer to permutation/architecture invariant, the raw weight values it consumes carry the same confound, and something about unseen depth remains hard for it specifically.

This is reported as a genuine open finding rather than smoothed over: **detecting a backdoor from weights alone transfers well across attack configuration but not across architecture family**, at least with the representations tried here. That's a meaningfully different and harder problem than in-distribution classification, and is arguably the more interesting result of this project.

## Limitations / future work

- Single trigger family (corner patch). "Held-out attack type" in the strict sense — a structurally different trigger mechanism like a blended/watermark pattern — hasn't been tested; only corner/size/color/target-class variation within one family.
- MNIST-scale only; CIFAR-10 was in scope for the original plan but not yet run.
- The architecture-depth generalization gap is unresolved. Candidate next steps: giving the classifier explicit conditioning on architecture depth so it can calibrate rather than transfer a fixed threshold; per-architecture calibration (isotonic/Platt scaling); or a GNN readout that's less sensitive to graph size (current mean+max pooling may itself carry depth-correlated scale).

## Running it

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m src.generate_population --n-models 400 --out-dir data/models
.venv/bin/python -m src.build_feature_matrix
.venv/bin/python -m src.evaluate_ood          # track (a): GBT + ID/OOD evaluation
.venv/bin/python -m src.train_gnn             # track (b): GNN + ID/OOD evaluation
```

Generated data (`data/models/`, `data/mnist/`, `data/features.csv`) is gitignored — regenerate it locally rather than pulling it from the repo.
