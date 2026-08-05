import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv, global_max_pool, global_mean_pool

from src.graph_repr import EDGE_DIM, NODE_FEAT_DIM


class GNNClassifier(nn.Module):
    def __init__(self, hidden_dim: int = 64, n_layers: int = 3, dropout: float = 0.2):
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
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.input_proj(x)
        for conv in self.convs:
            h = torch.relu(conv(h, edge_index, edge_attr))
        pooled = torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch)], dim=1)
        return self.head(pooled)
