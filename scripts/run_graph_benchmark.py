#!/usr/bin/env python3
"""Run the three-system comparison on a natural graph benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import (
    CANONICAL_EMBEDDING_DIMS,
    CANONICAL_EPOCHS,
    DATA_DIR,
    DEVICE,
    DROPOUT,
    LEARNING_RATE,
    RANDOM_SEED,
    RESULTS_DIR,
    WEIGHT_DECAY,
)
from data.benchmark_loader import SUPPORTED_DATASETS, load_graph_benchmark
from experiment.three_system_comparison import run_three_system_comparison
from simulation.cost_model import ScenarioConfig
from simulation.partitioning import assign_balanced_partitions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, default="cora")
    parser.add_argument("--epochs", type=int, default=CANONICAL_EPOCHS)
    parser.add_argument("--seeds", type=int, nargs="+", default=[RANDOM_SEED])
    parser.add_argument(
        "--embedding-dims", type=int, nargs="+", default=CANONICAL_EMBEDDING_DIMS
    )
    parser.add_argument("--num-partitions", type=int, default=4)
    parser.add_argument("--bandwidth-mbps", type=float, default=1.0)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    graph = load_graph_benchmark(args.dataset, DATA_DIR / "benchmarks")
    graph = assign_balanced_partitions(
        graph, num_partitions=args.num_partitions, seed=args.seeds[0]
    )
    num_classes = int(graph.y.max().item()) + 1

    print(
        f"{args.dataset}: {graph.num_nodes} nodes, {graph.num_edges} edges, "
        f"{graph.num_node_features} features, {num_classes} classes"
    )
    print(
        f"Partitioning: {graph.num_partitions} partitions, "
        f"{graph.num_boundary_edges} cross-partition edges"
    )

    results = run_three_system_comparison(
        graph=graph,
        input_dim=graph.num_node_features,
        num_classes=num_classes,
        embedding_dims=args.embedding_dims,
        epochs=args.epochs,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        dropout=DROPOUT,
        seeds=args.seeds,
        device_str=DEVICE,
        scenario=ScenarioConfig(bandwidth_bps=args.bandwidth_mbps * 1e6),
    )
    results.print_summary()

    output = args.output or str(
        RESULTS_DIR / f"three_system_{args.dataset}.json"
    )
    results.save(output)


if __name__ == "__main__":
    main()
