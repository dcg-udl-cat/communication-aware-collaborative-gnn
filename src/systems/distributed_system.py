"""System 2: Distributed tiny GNN — the main contribution.

Each node:
  1. Extracts a compressed embedding (embedding_dim << full feature dim)
  2. Sends it to neighbors over the constrained inter-node link
  3. Runs a tiny GNN that aggregates neighbor embeddings
  4. Classifies using the enriched representation

This is the sweet spot: better than local (uses neighbor context),
cheaper than cloud (compressed communication, tiny model).

Compression variants for ablation:
  embedding_dim = 8   — extreme compression (strong bandwidth constraint)
  embedding_dim = 16  — aggressive compression
  embedding_dim = 32  — moderate compression
  embedding_dim = 64  — light compression (default)

The key claim: with embedding_dim << full_feature_dim, the distributed GNN
still approaches cloud accuracy at a fraction of the bandwidth cost.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from simulation.cost_model import CostBreakdown, ScenarioConfig, compute_costs
from simulation.partitioning import communication_edge_count
from training.evaluator import evaluate_accuracy
from training.trainer import make_optimizer, train_with_logging
from utils.feature_normalization import normalize_graph_features
from utils.metrics import count_trainable_parameters


class EmbeddingCompressor(nn.Module):
    """Learned linear compression: full_dim -> embedding_dim.

    Trained end-to-end with the GNN. In deployment, each edge node
    runs only this layer before transmission (cheap: one matrix multiply).
    """
    def __init__(self, input_dim: int, embedding_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(x), p=2, dim=1)


class FixedPCACompressor(nn.Module):
    """Frozen train-fitted PCA projection used as a compression baseline."""

    def __init__(self, mean: torch.Tensor, components: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("mean", mean.detach().float())
        self.register_buffer("components", components.detach().float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = (x - self.mean) @ self.components.T
        return F.normalize(projected, p=2, dim=1)


class DistributedTinyGNN(nn.Module):
    """Compressor + TinyGNN as a single end-to-end trainable system.

    Compressor handles bandwidth constraint; GNN handles graph reasoning.
    Both are trained jointly so the compression is task-aware.
    """

    uses_graph = True

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        gat_heads: int = 4,
        compressor: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.compressor = compressor or EmbeddingCompressor(input_dim, embedding_dim)
        self.embedding_dim = embedding_dim

        from torch_geometric.nn import GATConv
        self.conv = GATConv(
            in_channels=embedding_dim,
            out_channels=hidden_dim,
            heads=gat_heads,
            concat=False,
            dropout=dropout,
            edge_dim=1,
        )
        self.lin_out = nn.Linear(hidden_dim, num_classes)
        self.dropout = dropout
        # Residual projection: aligns embedding_dim -> hidden_dim when they differ.
        # Without this, x + h would require embedding_dim == hidden_dim, which
        # is not guaranteed (and is the common case: emb=8, hidden=64).
        self.res_proj = (
            nn.Linear(embedding_dim, hidden_dim, bias=False)
            if embedding_dim != hidden_dim else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Step 1: compress (this is what each node transmits)
        x = self.compressor(x)
        return self.decode_embeddings(x, edge_index, edge_weight)

    def decode_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode already-compressed embeddings.

        Keeping transport outside the model makes the evaluation path use the
        same source/destination batch boundaries as the serialized protocol.
        """

        edge_attr = edge_weight.unsqueeze(1) if edge_weight is not None else None
        h = F.relu(self.conv(x, edge_index, edge_attr=edge_attr))
        # Residual: project x to hidden_dim then add graph-enriched h.
        # res_proj is Identity when embedding_dim == hidden_dim, else Linear.
        x = self.res_proj(x) + h
        return self.lin_out(F.dropout(x, p=self.dropout, training=self.training))


class FullFeatureGAT(nn.Module):
    """Full-feature upper bound with the same GAT decoder as the compressed model."""

    uses_graph = True

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        gat_heads: int = 4,
    ) -> None:
        super().__init__()
        from torch_geometric.nn import GATConv

        self.conv = GATConv(
            input_dim,
            hidden_dim,
            heads=gat_heads,
            concat=False,
            dropout=dropout,
            edge_dim=1,
        )
        self.res_proj = (
            nn.Linear(input_dim, hidden_dim, bias=False)
            if input_dim != hidden_dim else nn.Identity()
        )
        self.lin_out = nn.Linear(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        edge_attr = edge_weight.unsqueeze(1) if edge_weight is not None else None
        h = F.relu(self.conv(x, edge_index, edge_attr=edge_attr))
        x = self.res_proj(x) + h
        return self.lin_out(F.dropout(x, p=self.dropout, training=self.training))


@dataclass
class DistributedSystemResult:
    test_accuracy: float
    train_accuracy: float
    num_parameters: int
    embedding_dim: int
    cost: CostBreakdown
    loss_curve: list[float]
    compression_ratio: float    # full_dim / embedding_dim
    val_accuracy: float = float("nan")
    test_accuracy_std: float = 0.0
    test_accuracy_ci95: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        d = self.cost.as_dict()
        d.update({
            "system": "distributed",
            "test_accuracy": self.test_accuracy,
            "train_accuracy": self.train_accuracy,
            "num_parameters": self.num_parameters,
            "embedding_dim": self.embedding_dim,
            "compression_ratio": self.compression_ratio,
            "val_accuracy": self.val_accuracy,
            "test_accuracy_std": self.test_accuracy_std,
            "test_accuracy_ci95": self.test_accuracy_ci95,
            "loss_curve": self.loss_curve,
        })
        return d


def run_distributed_system(
    input_dim: int,
    embedding_dim: int,
    hidden_dim: int,
    num_classes: int,
    graph: Data,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    scenario: ScenarioConfig | None = None,
    model_flops: float = 5e6,
    dropout: float = 0.1,
    seed: int = 42,
) -> DistributedSystemResult:
    """Train and evaluate the distributed tiny GNN system."""
    from utils.seed import set_seed
    set_seed(seed)

    graph = graph.to(device)

    from training.splits import make_inductive_view, resolve_train_val_masks
    eff_train_mask, val_mask, _ = resolve_train_val_masks(graph, seed=seed)

    graph = normalize_graph_features(graph, train_mask=eff_train_mask).to(device)
    graph = make_inductive_view(graph, eff_train_mask).to(device)

    model = DistributedTinyGNN(
        input_dim=input_dim,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dropout=dropout,
    ).to(device)

    optimizer = make_optimizer(model, lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    orig_mask = graph.train_mask.clone()
    graph.train_mask = eff_train_mask
    loss_curve = train_with_logging(
        model, graph, optimizer, loss_fn, device,
        epochs=epochs, label=f"distributed/emb{embedding_dim}", val_mask=val_mask,
    )
    graph.train_mask = orig_mask

    test_acc  = evaluate_accuracy(model, graph, graph.test_mask)
    train_acc = evaluate_accuracy(model, graph, eff_train_mask)
    val_acc = evaluate_accuracy(model, graph, val_mask)

    cost = compute_costs(
        system_name="distributed",
        num_nodes=int(graph.num_nodes),
        embedding_dim=embedding_dim,
        model_flops=model_flops,
        is_cloud=False,
        num_messages=communication_edge_count(graph),
        cfg=scenario,
    )

    return DistributedSystemResult(
        test_accuracy=test_acc,
        train_accuracy=train_acc,
        num_parameters=count_trainable_parameters(model),
        embedding_dim=embedding_dim,
        cost=cost,
        loss_curve=loss_curve,
        compression_ratio=input_dim / embedding_dim,
        val_accuracy=val_acc,
    )
