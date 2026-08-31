"""Legacy CIFAR System 3: centralized three-layer GCN.

All data goes to a central server:
  - full feature access (no compression required)
  - larger model capacity
  - global graph over all nodes

Cost constraints (making it realistic, not unfairly strong):
  - bandwidth: num_nodes * full_feature_dim floats transmitted
  - latency: transmission + cloud compute (faster per FLOP, but more data)
  - energy: dominated by transmission cost (alpha * total_bytes)

The cloud system is intentionally NOT given unlimited capacity.
It uses the same feature pipeline and graph construction, just:
  - no compression on transmission
  - larger GNN (LegacyCloudGCN instead of TinyGNN)
  - cloud-speed compute (10 GFLOPs/s vs 100 MFLOPs/s edge)

This module is not used by the paper pipeline. The paper's matched centralized
reference is ``systems.distributed_system.FullFeatureGAT``.
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
class CloudSystemResult:
    test_accuracy: float
    train_accuracy: float
    num_parameters: int
    cost: CostBreakdown
    loss_curve: list[float]
    full_feature_dim: int
    val_accuracy: float = float("nan")
    test_accuracy_std: float = 0.0
    test_accuracy_ci95: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        d = self.cost.as_dict()
        d.update({
            "system": "cloud",
            "test_accuracy": self.test_accuracy,
            "train_accuracy": self.train_accuracy,
            "num_parameters": self.num_parameters,
            "full_feature_dim": self.full_feature_dim,
            "val_accuracy": self.val_accuracy,
            "test_accuracy_std": self.test_accuracy_std,
            "test_accuracy_ci95": self.test_accuracy_ci95,
            "loss_curve": self.loss_curve,
        })
        return d


def run_cloud_system(
    model: nn.Module,
    graph: Data,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    scenario: ScenarioConfig | None = None,
    model_flops: float = 1e8,
    seed: int = 42,
) -> CloudSystemResult:
    """Train and evaluate the cloud system (full features, larger model).

    Bandwidth cost reflects transmitting full features — no compression.
    Compute uses cloud_flops_per_s from the scenario config.
    """
    from utils.seed import set_seed
    set_seed(seed)

    graph = graph.to(device)
    full_feature_dim = int(graph.num_node_features)

    from training.splits import make_inductive_view, resolve_train_val_masks
    eff_train_mask, val_mask, _ = resolve_train_val_masks(graph, seed=seed)

    graph = normalize_graph_features(graph, train_mask=eff_train_mask).to(device)
    graph = make_inductive_view(graph, eff_train_mask).to(device)

    model = model.to(device)
    optimizer = make_optimizer(model, lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    orig_mask = graph.train_mask.clone()
    graph.train_mask = eff_train_mask
    loss_curve = train_with_logging(
        model, graph, optimizer, loss_fn, device,
        epochs=epochs, label="cloud", val_mask=val_mask,
    )
    graph.train_mask = orig_mask

    test_acc  = evaluate_accuracy(model, graph, graph.test_mask)
    train_acc = evaluate_accuracy(model, graph, eff_train_mask)
    val_acc = evaluate_accuracy(model, graph, val_mask)

    cost = compute_costs(
        system_name="cloud",
        num_nodes=int(graph.num_nodes),
        embedding_dim=full_feature_dim,   # no compression: full features transmitted
        model_flops=model_flops,
        is_cloud=True,
        transmit_full_features=True,
        full_feature_dim=full_feature_dim,
        cfg=scenario,
    )

    return CloudSystemResult(
        test_accuracy=test_acc,
        train_accuracy=train_acc,
        num_parameters=count_trainable_parameters(model),
        cost=cost,
        loss_curve=loss_curve,
        full_feature_dim=full_feature_dim,
        val_accuracy=val_acc,
    )
