"""Reproducible, leakage-safe masks for node-classification experiments."""

from __future__ import annotations

import torch
from torch_geometric.data import Data


def resolve_train_val_masks(
    graph: Data,
    *,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Return leakage-safe train/validation masks, preferring official splits.

    PyG benchmark datasets such as Planetoid and Flickr ship canonical
    validation masks.  Those masks are part of the benchmark protocol and
    must not be replaced by a split of the usually much smaller training set.
    Synthetic and legacy graphs without ``val_mask`` retain the existing
    stratified fallback.
    """

    train_mask = graph.train_mask.bool()
    val_mask = getattr(graph, "val_mask", None)
    if val_mask is None:
        train_mask, val_mask = stratified_train_val_masks(
            graph.y,
            train_mask,
            val_fraction=val_fraction,
            seed=seed,
        )
        return train_mask, val_mask, "stratified_fallback"

    val_mask = val_mask.bool()
    if train_mask.ndim != 1 or val_mask.ndim != 1:
        raise ValueError("train_mask and val_mask must be one-dimensional.")
    if train_mask.shape[0] != graph.num_nodes or val_mask.shape[0] != graph.num_nodes:
        raise ValueError("train_mask and val_mask must contain one entry per node.")
    if not train_mask.any() or not val_mask.any():
        raise ValueError("Official train and validation masks must be non-empty.")
    if (train_mask & val_mask).any():
        raise ValueError("Official train and validation masks must be disjoint.")
    test_mask = getattr(graph, "test_mask", None)
    if test_mask is not None and ((train_mask | val_mask) & test_mask.bool()).any():
        raise ValueError("Train/validation/test masks must be disjoint.")
    return train_mask, val_mask, "official"


def stratified_train_val_masks(
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    *,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split labelled training nodes while preserving class coverage when possible."""

    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1.")
    if labels.ndim != 1 or train_mask.ndim != 1:
        raise ValueError("labels and train_mask must be one-dimensional.")
    if labels.shape[0] != train_mask.shape[0]:
        raise ValueError("labels and train_mask must have the same length.")

    device = train_mask.device
    labels_cpu = labels.detach().cpu()
    candidate_idx = train_mask.detach().cpu().bool().nonzero(as_tuple=True)[0]
    if candidate_idx.numel() < 2:
        raise ValueError("At least two training nodes are required for a train/val split.")

    generator = torch.Generator().manual_seed(seed)
    val_parts: list[torch.Tensor] = []

    for class_id in torch.unique(labels_cpu[candidate_idx], sorted=True):
        class_idx = candidate_idx[labels_cpu[candidate_idx] == class_id]
        if class_idx.numel() < 2:
            continue
        shuffled = class_idx[torch.randperm(class_idx.numel(), generator=generator)]
        class_val_size = max(1, round(class_idx.numel() * val_fraction))
        class_val_size = min(class_val_size, class_idx.numel() - 1)
        val_parts.append(shuffled[:class_val_size])

    if val_parts:
        val_idx = torch.cat(val_parts)
    else:
        shuffled = candidate_idx[
            torch.randperm(candidate_idx.numel(), generator=generator)
        ]
        val_idx = shuffled[:1]

    val_mask_cpu = torch.zeros_like(train_mask.detach().cpu(), dtype=torch.bool)
    val_mask_cpu[val_idx] = True
    effective_train_cpu = train_mask.detach().cpu().bool() & ~val_mask_cpu

    if not effective_train_cpu.any() or not val_mask_cpu.any():
        raise RuntimeError("Unable to create non-empty train and validation masks.")

    return effective_train_cpu.to(device), val_mask_cpu.to(device)


def make_inductive_view(graph: Data, source_mask: torch.Tensor) -> Data:
    """Keep only messages originating from currently supervised train nodes.

    PyG represents an edge as ``source -> target``. Filtering on the source
    prevents validation/test features from influencing train-node embeddings,
    while still allowing train nodes to send context to validation and test
    targets.
    """

    if source_mask.ndim != 1 or source_mask.shape[0] != graph.num_nodes:
        raise ValueError("source_mask must contain one boolean per graph node.")

    view = graph.clone()
    source_mask = source_mask.to(graph.edge_index.device).bool()
    keep = source_mask[graph.edge_index[0]]
    view.edge_index = graph.edge_index[:, keep]

    edge_weight = getattr(graph, "edge_weight", None)
    if edge_weight is not None:
        view.edge_weight = edge_weight[keep]

    boundary_edge_mask = getattr(graph, "boundary_edge_mask", None)
    if boundary_edge_mask is not None:
        view.boundary_edge_mask = boundary_edge_mask[keep]
        view.num_boundary_edges = int(view.boundary_edge_mask.sum().item())

    view.message_source_mask = source_mask
    return view
