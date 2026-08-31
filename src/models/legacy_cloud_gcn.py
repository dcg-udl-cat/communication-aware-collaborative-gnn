"""Three-layer GCN used only by the legacy CIFAR/KNN experiments."""

import torch
from torch import nn
import torch.nn.functional as functional
from torch_geometric.nn import GCNConv

from config import DROPOUT, FULL_GNN_HIDDEN_DIM, INPUT_DIM, NUM_CLASSES


class LegacyCloudGCN(nn.Module):
    """Higher-capacity GCN for the legacy CIFAR-derived graph ablation."""

    uses_graph = True

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dim: int = FULL_GNN_HIDDEN_DIM,
        num_classes: int = NUM_CLASSES,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()

        self.dropout = dropout
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return one logit vector per node."""

        x = self.conv1(x, edge_index)
        x = functional.relu(x)
        x = functional.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = functional.relu(x)
        x = functional.dropout(x, p=self.dropout, training=self.training)

        return self.conv3(x, edge_index)
