"""System 1: Local-only baseline — no cooperation between nodes.

Each node runs its own tiny classifier on its partial view.
No communication. This is the lower bound.

In CIFAR-10 terms:
  Each "node" sees a subset of images (its local partition).
  It classifies them independently using only its own features.
  No graph, no message passing, no inter-node information.

Cost:
  bandwidth = 0 (no transmission)
  latency   = compute_time only
  energy    = compute FLOPs only
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch_geometric.data import Data

from simulation.cost_model import CostBreakdown, ScenarioConfig, compute_costs
from training.evaluator import evaluate_accuracy
from training.trainer import make_optimizer, train_with_logging
from utils.feature_normalization import normalize_graph_features
from utils.metrics import count_trainable_parameters


@dataclass
class LocalSystemResult:
    test_accuracy: float
    train_accuracy: float
    num_parameters: int
    cost: CostBreakdown
    loss_curve: list[float]
    embedding_dim: int      # = input feature dim (no compression)
    val_accuracy: float = float("nan")
    test_accuracy_std: float = 0.0
    test_accuracy_ci95: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        d = self.cost.as_dict()
        d.update({
            "system": "local",
            "test_accuracy": self.test_accuracy,
            "train_accuracy": self.train_accuracy,
            "num_parameters": self.num_parameters,
            "embedding_dim": self.embedding_dim,
            "val_accuracy": self.val_accuracy,
            "test_accuracy_std": self.test_accuracy_std,
            "test_accuracy_ci95": self.test_accuracy_ci95,
            "loss_curve": self.loss_curve,
        })
        return d


def run_local_system(
    model: nn.Module,
    graph: Data,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    scenario: ScenarioConfig | None = None,
    model_flops: float = 1e7,
    seed: int = 42,
) -> LocalSystemResult:
    """Train and evaluate the local system (no graph, no communication).

    The model receives only node features (graph.x), ignoring edge_index.
    This simulates each camera/sensor running a standalone classifier.
    """
    from utils.seed import set_seed
    set_seed(seed)

    graph = graph.to(device)

    from training.splits import resolve_train_val_masks
    eff_train_mask, val_mask, _ = resolve_train_val_masks(graph, seed=seed)

    # Normalize features using train statistics
    graph = normalize_graph_features(graph, train_mask=eff_train_mask).to(device)

    model = model.to(device)
    optimizer = make_optimizer(model, lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    orig_mask = graph.train_mask.clone()
    graph.train_mask = eff_train_mask
    loss_curve = train_with_logging(
        model, graph, optimizer, loss_fn, device,
        epochs=epochs, label="local", val_mask=val_mask,
    )
    graph.train_mask = orig_mask

    test_acc  = evaluate_accuracy(model, graph, graph.test_mask)
    train_acc = evaluate_accuracy(model, graph, eff_train_mask)
    val_acc = evaluate_accuracy(model, graph, val_mask)

    cost = compute_costs(
        system_name="local",
        num_nodes=int(graph.num_nodes),
        embedding_dim=0,        # no transmission
        model_flops=model_flops,
        is_cloud=False,
        cfg=scenario,
    )

    return LocalSystemResult(
        test_accuracy=test_acc,
        train_accuracy=train_acc,
        num_parameters=count_trainable_parameters(model),
        cost=cost,
        loss_curve=loss_curve,
        embedding_dim=int(graph.num_node_features),
        val_accuracy=val_acc,
    )
