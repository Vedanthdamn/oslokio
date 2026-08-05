import torch
import torch.nn as nn
from torch_geometric.nn import AttentionalAggregation, GINEConv, global_mean_pool

from src.graph_repr import EDGE_DIM, NODE_FEAT_DIM


class GNNClassifier(nn.Module):
    """GINE message passing over a network's neuron/weight graph.

    Readout deliberately avoids max pooling: node count scales with architecture
    depth and width (roughly 220 nodes at 2 conv layers vs 480 at 4), so a max over
    nodes grows with graph size and hands the classifier an architecture-identifying
    shortcut. Mean and attention pooling are both weighted averages and stay
    size-invariant, which matters for the held-out-architecture test.
    """

    def __init__(self, hidden_dim: int = 96, n_layers: int = 4, dropout: float = 0.2):
        super().__init__()
        self.input_proj = nn.Linear(NODE_FEAT_DIM, hidden_dim)
        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(mlp, edge_dim=EDGE_DIM))

        self.attention_pool = AttentionalAggregation(
            gate_nn=nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
                                  nn.Linear(hidden_dim // 2, 1))
        )
        # jumping knowledge: every layer's pooled representation feeds the head, so
        # shallow and deep structure are both visible regardless of network depth.
        readout_dim = hidden_dim * 2 * n_layers
        self.head = nn.Sequential(
            nn.Linear(readout_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.input_proj(x)
        pooled = []
        for conv in self.convs:
            out = torch.relu(conv(h, edge_index, edge_attr))
            h = h + out  # residual keeps gradients healthy through depth
            pooled.append(torch.cat([global_mean_pool(h, batch),
                                     self.attention_pool(h, batch)], dim=1))
        return self.head(torch.cat(pooled, dim=1))
