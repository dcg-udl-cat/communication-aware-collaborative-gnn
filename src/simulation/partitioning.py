"""Deterministic graph partition metadata and boundary-message accounting."""

from __future__ import annotations

import torch
from torch_geometric.data import Data


def assign_balanced_partitions(
    graph: Data,
    num_partitions: int,
    *,
    seed: int = 42,
) -> Data:
    """Assign nodes evenly to partitions and annotate cross-partition edges.

    This is intentionally a simple reproducible baseline partitioner. It makes
    communication accounting exact for a chosen placement without pretending
    to be a graph-cut optimizer such as METIS.
    """

    if num_partitions <= 0:
        raise ValueError("num_partitions must be positive.")
    if graph.num_nodes == 0:
        raise ValueError("Cannot partition an empty graph.")

    num_partitions = min(num_partitions, int(graph.num_nodes))
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(graph.num_nodes, generator=generator)
    partition_id = torch.empty(graph.num_nodes, dtype=torch.long)
    partition_id[permutation] = torch.arange(graph.num_nodes) % num_partitions

    return _annotate_partitions(graph, partition_id, num_partitions)


def assign_locality_aware_partitions(graph: Data, num_partitions: int) -> Data:
    """Create deterministic balanced partitions from a locality-preserving order.

    Reverse Cuthill--McKee places adjacent nodes near one another in a linear
    order. Splitting that order into balanced contiguous blocks is a small,
    dependency-free-in-practice baseline (SciPy is already required by
    scikit-learn), not a claim of optimal graph partitioning.
    """

    if num_partitions <= 0:
        raise ValueError("num_partitions must be positive.")
    if graph.num_nodes == 0:
        raise ValueError("Cannot partition an empty graph.")

    import numpy as np
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import reverse_cuthill_mckee

    num_partitions = min(num_partitions, int(graph.num_nodes))
    src, dst = graph.edge_index.detach().cpu().numpy()
    values = np.ones(src.shape[0], dtype=np.uint8)
    adjacency = coo_matrix(
        (values, (src, dst)),
        shape=(graph.num_nodes, graph.num_nodes),
    ).tocsr()
    adjacency = (adjacency + adjacency.T).astype(bool).astype(np.uint8)
    order = torch.from_numpy(
        reverse_cuthill_mckee(adjacency, symmetric_mode=True).copy()
    ).long()
    partition_id = torch.empty(graph.num_nodes, dtype=torch.long)
    balanced_ids = torch.arange(graph.num_nodes) * num_partitions // graph.num_nodes
    partition_id[order] = balanced_ids
    return _annotate_partitions(graph, partition_id, num_partitions)


def assign_partitions(
    graph: Data,
    num_partitions: int,
    *,
    strategy: str = "random",
    seed: int = 42,
) -> Data:
    if strategy == "random":
        return assign_balanced_partitions(graph, num_partitions, seed=seed)
    if strategy == "locality":
        return assign_locality_aware_partitions(graph, num_partitions)
    raise ValueError("strategy must be 'random' or 'locality'.")


def _annotate_partitions(
    graph: Data,
    partition_id: torch.Tensor,
    num_partitions: int,
) -> Data:
    src, dst = graph.edge_index.cpu()
    boundary_mask = partition_id[src] != partition_id[dst]
    partitioned = graph.clone()
    partitioned.partition_id = partition_id.to(graph.edge_index.device)
    partitioned.boundary_edge_mask = boundary_mask.to(graph.edge_index.device)
    partitioned.num_partitions = num_partitions
    partitioned.num_boundary_edges = int(boundary_mask.sum().item())
    return partitioned


def communication_edge_count(graph: Data) -> int:
    """Return the number of remote messages required for one GNN layer."""

    boundary_mask = getattr(graph, "boundary_edge_mask", None)
    if boundary_mask is None:
        return int(graph.num_edges)
    return int(boundary_mask.sum().item())
