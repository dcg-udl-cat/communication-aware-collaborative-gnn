"""Exact batched communication traces for partitioned one-layer GNN inference."""

from __future__ import annotations

import struct
from dataclasses import asdict, dataclass

import torch
from torch_geometric.data import Data


@dataclass(frozen=True)
class CommunicationTrace:
    cross_partition_edges: int
    unique_transfers: int
    transfer_batches: int
    embedding_dim: int
    bytes_per_value: int
    raw_payload_bytes: int
    serialized_wire_bytes: int
    conservative_edge_payload_bytes: int
    boundary_edge_ratio: float

    def as_dict(self) -> dict:
        return asdict(self)


def boundary_transfer_index(graph: Data) -> torch.Tensor:
    """Return unique ``(source node, destination partition)`` transfers."""

    partition_id = getattr(graph, "partition_id", None)
    if partition_id is None:
        raise ValueError("graph must have partition_id metadata.")
    src, dst = graph.edge_index.detach().cpu()
    partitions = partition_id.detach().cpu().long()
    keep = partitions[src] != partitions[dst]
    if not keep.any():
        return torch.empty((2, 0), dtype=torch.long)
    pairs = torch.stack((src[keep], partitions[dst[keep]]), dim=1)
    return torch.unique(pairs, dim=0, sorted=True).T.contiguous()


def communication_trace(
    graph: Data,
    embedding_dim: int,
    *,
    bytes_per_value: int = 4,
) -> CommunicationTrace:
    """Account for the exact wire format used by ``serialize_transfer_batches``."""

    if embedding_dim <= 0 or bytes_per_value not in {1, 4}:
        raise ValueError("embedding_dim must be positive and values must be int8 or float32.")
    transfers = boundary_transfer_index(graph)
    partition_id = graph.partition_id.detach().cpu().long()
    if transfers.shape[1]:
        batch_pairs = torch.stack(
            (partition_id[transfers[0]], transfers[1]), dim=1
        )
        batch_count = int(torch.unique(batch_pairs, dim=0).shape[0])
    else:
        batch_count = 0
    unique_count = int(transfers.shape[1])
    source, target = graph.edge_index.detach().cpu()
    cross_edges = int((partition_id[source] != partition_id[target]).sum().item())
    raw_bytes = unique_count * embedding_dim * bytes_per_value
    scale_bytes = 4 * batch_count if bytes_per_value == 1 else 0
    wire_bytes = 8 + batch_count * 12 + unique_count * 8 + raw_bytes + scale_bytes
    return CommunicationTrace(
        cross_partition_edges=cross_edges,
        unique_transfers=unique_count,
        transfer_batches=batch_count,
        embedding_dim=embedding_dim,
        bytes_per_value=bytes_per_value,
        raw_payload_bytes=raw_bytes,
        serialized_wire_bytes=wire_bytes,
        conservative_edge_payload_bytes=cross_edges * embedding_dim * bytes_per_value,
        boundary_edge_ratio=(cross_edges / graph.num_edges if graph.num_edges else 0.0),
    )


def centralized_upload_size(
    num_nodes: int,
    feature_dim: int,
    *,
    bytes_per_value: int = 4,
) -> tuple[int, int]:
    """Return raw-payload and serialized sizes for one centralized upload."""

    if num_nodes < 0 or feature_dim <= 0 or bytes_per_value not in {1, 4}:
        raise ValueError("Invalid centralized upload dimensions or precision.")
    raw_bytes = num_nodes * feature_dim * bytes_per_value
    scale_bytes = 4 if bytes_per_value == 1 else 0
    return raw_bytes, 8 + 12 + num_nodes * 8 + scale_bytes + raw_bytes


def symmetric_int8_quantize(values: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Apply the per-batch symmetric int8 transport quantizer."""

    values = values.detach().cpu().float()
    scale = float(values.abs().amax().clamp_min(1e-12).item() / 127.0)
    return (values / scale).round().clamp(-127, 127).to(torch.int8), scale


def symmetric_int8_round_trip(values: torch.Tensor) -> torch.Tensor:
    quantized, scale = symmetric_int8_quantize(values)
    return quantized.float().mul_(scale).to(values.device)


def serialize_centralized_upload(features: torch.Tensor, *, quantized: bool = False) -> bytes:
    """Serialize all node features as one upload using the trace framing."""

    if features.ndim != 2:
        raise ValueError("features must have shape (num_nodes, feature_dim).")
    node_ids = torch.arange(features.shape[0], dtype=torch.int64)
    output = bytearray(struct.pack("<4sI", b"CGNN", 1))
    output.extend(struct.pack("<III", 0, 0, features.shape[0]))
    output.extend(node_ids.numpy().tobytes())
    values = features.detach().cpu().float().contiguous()
    if quantized:
        values, scale = symmetric_int8_quantize(values)
        output.extend(struct.pack("<f", scale))
    output.extend(values.numpy().tobytes())
    return bytes(output)


def serialize_transfer_batches(
    embeddings: torch.Tensor,
    graph: Data,
    *,
    quantized: bool = False,
) -> bytes:
    """Serialize boundary batches with node ids and an optional int8 scale."""

    if embeddings.ndim != 2 or embeddings.shape[0] != graph.num_nodes:
        raise ValueError("embeddings must have shape (num_nodes, embedding_dim).")
    transfers = boundary_transfer_index(graph)
    partition_id = graph.partition_id.detach().cpu().long()
    grouped: dict[tuple[int, int], list[int]] = {}
    for node, destination in transfers.T.tolist():
        grouped.setdefault((int(partition_id[node]), destination), []).append(node)

    output = bytearray(struct.pack("<4sI", b"CGNN", len(grouped)))
    values = embeddings.detach().cpu().float()
    for (source_partition, destination_partition), node_ids in sorted(grouped.items()):
        output.extend(struct.pack("<III", source_partition, destination_partition, len(node_ids)))
        output.extend(torch.tensor(node_ids, dtype=torch.int64).numpy().tobytes())
        batch = values[node_ids].contiguous()
        if quantized:
            batch, scale = symmetric_int8_quantize(batch)
            output.extend(struct.pack("<f", scale))
        output.extend(batch.numpy().tobytes())
    return bytes(output)


def deserialize_transfer_batches(
    payload: bytes,
    embedding_dim: int,
    *,
    quantized: bool = False,
) -> dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]]:
    """Decode the trace format; primarily used for equivalence tests."""

    import numpy as np

    magic, batch_count = struct.unpack_from("<4sI", payload, 0)
    if magic != b"CGNN":
        raise ValueError("Invalid communication trace magic.")
    offset = 8
    decoded = {}
    for _ in range(batch_count):
        source, destination, count = struct.unpack_from("<III", payload, offset)
        offset += 12
        ids_size = count * 8
        ids = torch.from_numpy(
            np.frombuffer(payload, dtype="<i8", count=count, offset=offset).copy()
        )
        offset += ids_size
        if quantized:
            scale = struct.unpack_from("<f", payload, offset)[0]
            offset += 4
            size = count * embedding_dim
            array = np.frombuffer(payload, dtype=np.int8, count=size, offset=offset).copy()
            values = torch.from_numpy(array.reshape(count, embedding_dim)).float() * scale
            offset += size
        else:
            size = count * embedding_dim * 4
            array = np.frombuffer(payload, dtype="<f4", count=count * embedding_dim, offset=offset).copy()
            values = torch.from_numpy(array.reshape(count, embedding_dim))
            offset += size
        decoded[(source, destination)] = (ids, values)
    if offset != len(payload):
        raise ValueError("Communication trace contains trailing or malformed bytes.")
    return decoded
