#!/usr/bin/env python3
"""Post-review METIS comparison; preserve the submitted experiment records.

Requires the optional analysis dependency ``pymetis==2025.2.2``. This topology
analysis never trains models or uses test outcomes to select dimensions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import statistics
import sys
from pathlib import Path

import numpy as np
import pymetis
import torch
from scipy.sparse import coo_matrix
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import DATA_DIR
from data.benchmark_loader import load_graph_benchmark
from simulation.communication import communication_trace
from simulation.partitioning import assign_partitions

DATASETS = ("cora", "citeseer", "pubmed", "flickr")
PARTITIONS = (2, 4, 8)
SEEDS = (42, 123, 2024)
SOURCE = ROOT / "artifacts/paper/results/release-v1.0.0-final"
OUTPUT = ROOT / "artifacts/paper/revision/analysis"
METRICS = (
    "boundary_edge_ratio", "cross_partition_edges", "unique_transfers",
    "transfer_batches", "serialized_wire_bytes", "max_to_mean_nodes",
)


def selected_widths() -> dict[str, int]:
    """Recheck the frozen gate using validation outcomes, deduplicating repeats."""
    values = {}
    with (SOURCE / "raw/results.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            if (row.get("system") != "learned" or row.get("status") != "ok"
                    or row.get("feature_normalization") != "row-l2"
                    or row.get("message_protocol") != "transductive"):
                continue
            key = row["dataset"], row["embedding_dim"], row["seed"]
            if key in values:
                assert values[key] == row["val_accuracy"]
            values[key] = row["val_accuracy"]
    frozen = json.loads((SOURCE / "scientific_gate.json").read_text())
    widths = {}
    for dataset in DATASETS:
        means = {}
        for dim in (8, 16, 32, 64):
            scores = [v for (d, k, _), v in values.items() if d == dataset and k == dim]
            assert len(scores) == 5, (dataset, dim, len(scores))
            means[dim] = statistics.mean(scores)
        widths[dataset] = min(means, key=lambda k: (-means[k], k))
        assert widths[dataset] == frozen["datasets"][dataset]["selected_embedding_dim"]
    return widths


def metis_adjacency(graph: Data):
    """Build the undirected, simple, loop-free graph required by METIS."""
    src, dst = graph.edge_index.numpy()
    adjacency = coo_matrix(
        (np.ones(len(src), dtype=bool), (src, dst)),
        shape=(graph.num_nodes, graph.num_nodes),
    ).tocsr()
    adjacency = adjacency.maximum(adjacency.T)
    adjacency.setdiag(False)
    adjacency.eliminate_zeros()
    adjacency.sort_indices()
    return adjacency


def metis_partition(graph: Data, adjacency, count: int, seed: int) -> tuple[Data, int]:
    result = pymetis.part_graph(
        count,
        adjacency=pymetis.CSRAdjacency(adjacency.indptr, adjacency.indices),
        recursive=False,
        tpwgts=[1 / count] * count,
        options=pymetis.Options(
            seed=seed, ufactor=30, objtype=pymetis.ObjType.CUT,
            ncuts=1, niter=10, numbering=0, contig=0,
        ),
    )
    partition = torch.tensor(result.vertex_part, dtype=torch.long)
    assert partition.shape == (graph.num_nodes,)
    assert set(partition.tolist()) == set(range(count))
    source, target = adjacency.nonzero()
    actual_cut = np.count_nonzero(partition.numpy()[source] != partition.numpy()[target]) // 2
    assert actual_cut == result.edge_cuts
    return Data(edge_index=graph.edge_index, num_nodes=graph.num_nodes,
                partition_id=partition, num_partitions=count), result.edge_cuts


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    widths = selected_widths()
    rows = []
    graph_hashes = {}
    for dataset in DATASETS:
        loaded = load_graph_benchmark(dataset, DATA_DIR / "benchmarks")
        graph = Data(edge_index=loaded.edge_index.cpu(), num_nodes=loaded.num_nodes)
        input_dim = loaded.num_node_features
        del loaded
        graph_hashes[dataset] = hashlib.sha256(graph.edge_index.numpy().tobytes()).hexdigest()
        adjacency = metis_adjacency(graph)
        for count in PARTITIONS:
            for strategy in ("random", "rcm", "metis"):
                for seed in (None,) if strategy == "rcm" else SEEDS:
                    edge_cut = None
                    if strategy == "metis":
                        placed, edge_cut = metis_partition(graph, adjacency, count, seed)
                    else:
                        placed = assign_partitions(
                            graph, count, strategy="locality" if strategy == "rcm" else "random",
                            seed=seed or 0,
                        )
                    trace = communication_trace(placed, widths[dataset]).as_dict()
                    sizes = torch.bincount(placed.partition_id, minlength=count).tolist()
                    rows.append(dict(
                        dataset=dataset, num_partitions=count, strategy=strategy,
                        placement_seed=seed, num_nodes=graph.num_nodes, num_edges=graph.num_edges,
                        input_dim=input_dim, **trace, min_partition_nodes=min(sizes),
                        max_partition_nodes=max(sizes), max_to_mean_nodes=max(sizes) / (graph.num_nodes / count),
                        partition_sizes=sizes, metis_undirected_edge_cut=edge_cut,
                        partition_sha256=hashlib.sha256(placed.partition_id.numpy().tobytes()).hexdigest(),
                    ))
                print(f"{dataset} P={count} {strategy}: done", flush=True)
    assert len(rows) == 84

    # The new random/RCM accounting must reproduce the submitted topology traces.
    existing = {}
    with (SOURCE / "raw/results.jsonl").open() as handle:
        for line in handle:
            r = json.loads(line)
            if (r.get("system") == "learned" and r.get("status") == "ok"
                    and r.get("embedding_dim") == widths[r["dataset"]]):
                key = r["dataset"], r["num_partitions"], r["partition_strategy"], r["partition_seed"]
                existing[key] = r
    checked = 0
    for row in rows:
        if row["strategy"] == "metis":
            continue
        key = (row["dataset"], row["num_partitions"],
               "locality" if row["strategy"] == "rcm" else "random", row["placement_seed"] or 0)
        for metric in METRICS[:-1]:
            assert row[metric] == existing[key][metric], (key, metric)
        checked += 1
    summary = []
    for dataset in DATASETS:
        for count in PARTITIONS:
            for strategy in ("random", "rcm", "metis"):
                subset = [r for r in rows if (r["dataset"], r["num_partitions"], r["strategy"])
                          == (dataset, count, strategy)]
                item = dict(dataset=dataset, num_partitions=count, strategy=strategy, n=len(subset))
                for metric in METRICS:
                    vals = [r[metric] for r in subset]
                    item.update({f"{metric}_mean": statistics.mean(vals),
                                 f"{metric}_min": min(vals), f"{metric}_max": max(vals),
                                 f"{metric}_sd": statistics.stdev(vals) if len(vals) > 1 else None})
                summary.append(item)
    metadata = dict(
        analysis="Post-review topology accounting; no retraining or retuning",
        selected_widths=widths, partition_counts=PARTITIONS, placement_seeds=SEEDS,
        metis=dict(version=pymetis.version, algorithm="direct k-way", objective="edge cut",
                   ufactor=30, ncuts=1, niter=10, equal_target_weights=True,
                   graph="simple undirected, self loops removed; original directed entries used for wire accounting"),
        uncertainty="Mean, sample SD and observed min/max across 3 placement seeds; not training uncertainty or confidence bounds. RCM is one deterministic reference.",
        reproduced_submitted_traces=checked,
        source_hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [SOURCE / "raw/results.jsonl", SOURCE / "scientific_gate.json"]},
        graph_edge_index_sha256=graph_hashes,
        package_versions={p: importlib.metadata.version(p) for p in ["torch", "torch-geometric", "numpy", "scipy", "pymetis"]},
        python=platform.python_version(), platform=platform.platform(),
        documentation=["https://documen.tician.de/pymetis/functionality.html",
                       "https://github.com/KarypisLab/METIS/blob/master/manual/manual.pdf"],
    )
    (output / "placement_results.json").write_text(json.dumps(dict(metadata=metadata, rows=rows, summary=summary), indent=2) + "\n")
    write_csv(output / "placement_raw.csv", rows)
    write_csv(output / "placement_summary.csv", summary)
    lines = ["# Post-review placement analysis", "", metadata["analysis"], "",
             "METIS 2025.2.2 (PyMETIS version), direct k-way edge-cut minimization; equal node-count targets; ufactor=30 (3% requested imbalance tolerance).",
             "Random and METIS each use seeds 42, 123, 2024. RCM retains the submitted deterministic realization. Frozen validation-only k=64 on every dataset.",
             "No model retraining is needed: float32 byte counts depend on topology and width, not embedding values. No METIS runtime, accuracy, or device results are inferred.",
             "", "Mean [observed min, max] across three placements; RCM is one value. MB denotes 10^6 bytes. These are placement sensitivity ranges, not confidence intervals.", "",
             "| Dataset | P | Random cut | RCM cut | METIS cut | Random MB | RCM MB | METIS MB |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for dataset in DATASETS:
        for count in PARTITIONS:
            cells = [dataset, str(count)]
            for metric, scale in (("boundary_edge_ratio", 1), ("serialized_wire_bytes", 1e6)):
                for strategy in ("random", "rcm", "metis"):
                    item = next(s for s in summary if (s["dataset"], s["num_partitions"], s["strategy"]) == (dataset, count, strategy))
                    value = f"{item[metric + '_mean'] / scale:.4f}"
                    if strategy != "rcm":
                        value += f" [{item[metric + '_min'] / scale:.4f}, {item[metric + '_max'] / scale:.4f}]"
                    cells.append(value)
            lines.append("| " + " | ".join(cells) + " |")
    lines.extend(["", f"Verification: all {checked} random/RCM topology traces exactly reproduce submitted counts and bytes; METIS reported edge cuts equal independently counted undirected cuts.",
                  "Raw CSV/JSON records contain U=unique_transfers, B=transfer_batches, directed boundary-edge counts, complete partition sizes, actual max/mean node balance, bytes and membership hashes.",
                  "", "Sources: [PyMETIS API](https://documen.tician.de/pymetis/functionality.html); [METIS manual](https://github.com/KarypisLab/METIS/blob/master/manual/manual.pdf).",
                  "", "Reproduce: `python -m pip install pymetis==2025.2.2` then `python scripts/analyze_revision_placement.py` from the repository environment."])
    (output / "placement_analysis.md").write_text("\n".join(lines) + "\n")
    print(f"Saved {len(rows)} placements and {len(summary)} summaries to {output}")


if __name__ == "__main__":
    main()
