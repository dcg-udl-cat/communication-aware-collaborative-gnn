"""Load standard natural graph benchmarks through PyTorch Geometric."""

from __future__ import annotations

from pathlib import Path

import torch
from torch_geometric.data import Data


SUPPORTED_DATASETS = ("cora", "citeseer", "pubmed", "flickr", "reddit")


def load_graph_benchmark(name: str, root: str | Path) -> Data:
    """Download/load a benchmark and return one graph with canonical masks."""

    normalized = name.lower()
    root = Path(root).expanduser()

    if normalized in {"cora", "citeseer", "pubmed"}:
        from torch_geometric.datasets import Planetoid

        canonical_name = {
            "cora": "Cora",
            "citeseer": "CiteSeer",
            "pubmed": "PubMed",
        }[normalized]
        graph = Planetoid(root=str(root / normalized), name=canonical_name)[0]
    elif normalized == "flickr":
        from torch_geometric.datasets import Flickr

        graph = Flickr(root=str(root / normalized))[0]
    elif normalized == "reddit":
        from torch_geometric.datasets import Reddit

        graph = Reddit(root=str(root / normalized))[0]
    else:
        raise ValueError(
            f"Unsupported dataset {name!r}. Choose one of {SUPPORTED_DATASETS}."
        )

    graph = graph.clone()
    graph.x = graph.x.float()
    graph.y = graph.y.long()
    graph.train_mask = _flatten_mask(graph.train_mask)
    if getattr(graph, "val_mask", None) is not None:
        graph.val_mask = _flatten_mask(graph.val_mask)
    graph.test_mask = _flatten_mask(graph.test_mask)
    graph.node_image_counts = torch.ones(graph.num_nodes, dtype=torch.long)
    graph.edge_weight = None
    graph.dataset_name = normalized
    return graph


def _flatten_mask(mask: torch.Tensor) -> torch.Tensor:
    """Use the first predefined split when a dataset exposes multiple splits."""

    if mask.ndim == 1:
        return mask.bool()
    if mask.ndim == 2:
        return mask[:, 0].bool()
    raise ValueError(f"Expected a 1-D or 2-D split mask, got shape {tuple(mask.shape)}.")
