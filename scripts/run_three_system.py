#!/usr/bin/env python3
"""Entry point: run the three-system comparison and save results.

Usage:
    python scripts/run_three_system.py
    python scripts/run_three_system.py --epochs 50 --seeds 42 123
    python scripts/run_three_system.py --embedding-dims 8 16 32 64
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import (
    CANONICAL_EMBEDDING_DIMS,
    CANONICAL_EPOCHS,
    CANONICAL_K_NEIGHBORS,
    CONTROLLED_TEST_SUBSET_SIZE,
    CONTROLLED_TRAIN_SUBSET_SIZE,
    DEVICE,
    DROPOUT,
    ENCODER_TYPE,
    IMAGES_PER_NODE,
    KNN_METRIC,
    LEARNING_RATE,
    PCA_COMPONENTS,
    RANDOM_SEED,
    RESULTS_DIR,
    USE_MUTUAL_KNN,
    USE_PCA,
    USE_PRETRAINED_ENCODER,
    WEIGHT_DECAY,
)
from data.cifar_loader import load_cifar10
from data.feature_extractor import build_node_features_from_images
from data.graph_builder import build_similarity_graph
from data.pca_compressor import PCACompressor
from experiment.three_system_comparison import run_three_system_comparison
from simulation.cost_model import ScenarioConfig
from simulation.partitioning import assign_balanced_partitions
from utils.alignment_check import verify_graph_alignment
from utils.seed import set_seed


def build_graph(seed: int, train_subset: int, test_subset: int, k_neighbors: int = CANONICAL_K_NEIGHBORS):
    """Load data, extract features, build graph.

    Feature pipeline (controlled by config flags):
      USE_PRETRAINED_ENCODER=True  → frozen ResNet18/DINO embeddings
                                     (best graph quality, requires torchvision)
      USE_PCA=True                 → flatten + PCA + StandardScaler
                                     (faster, no download)

    Cloud note: all three systems use the SAME feature pipeline.
    Cloud advantage = bigger model + no bandwidth constraint, not richer inputs.
    """
    set_seed(seed)
    cifar = load_cifar10(train_subset_size=train_subset, test_subset_size=test_subset)

    gen = torch.Generator().manual_seed(seed)

    if USE_PRETRAINED_ENCODER:
        from data.encoder import DinoEncoder, ResNet18Encoder
        encoder = DinoEncoder() if ENCODER_TYPE == 'dino' else ResNet18Encoder()
        print(f"Using pretrained encoder: {ENCODER_TYPE} "
              f"(output_dim={encoder.OUTPUT_DIM})")
        # Encode all images; labels and counts come from feature_extractor
        train_enc = encoder.encode(cifar.train.images)
        test_enc  = encoder.encode(cifar.test.images)
        # Wrap as node features: each image becomes one node
        train_f, train_l, train_c = train_enc, cifar.train.labels, torch.ones(len(train_enc), dtype=torch.long)
        test_f,  test_l,  test_c  = test_enc,  cifar.test.labels,  torch.ones(len(test_enc),  dtype=torch.long)
    else:
        train_f, train_l, train_c = build_node_features_from_images(
            cifar.train.images, cifar.train.labels,
            images_per_node=IMAGES_PER_NODE, shuffle=True, generator=gen,
        )
        test_f, test_l, test_c = build_node_features_from_images(
            cifar.test.images, cifar.test.labels,
            images_per_node=IMAGES_PER_NODE, shuffle=True, generator=gen,
        )

        if USE_PCA:
            comp = PCACompressor(n_components=PCA_COMPONENTS)
            train_f = comp.fit_transform(train_f)
            test_f  = comp.transform(test_f)

    graph = build_similarity_graph(
        train_features=train_f, train_labels=train_l,
        test_features=test_f,   test_labels=test_l,
        train_node_image_counts=train_c, test_node_image_counts=test_c,
        k_neighbors=k_neighbors, knn_metric=KNN_METRIC,
        use_mutual_knn=USE_MUTUAL_KNN,
    )
    verify_graph_alignment(graph, label="graph")
    # Tag graph with the feature pipeline used — read by _print_graph_summary
    graph.encoder_name = ENCODER_TYPE if USE_PRETRAINED_ENCODER else None
    return graph, int(train_f.shape[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",         type=int,   default=CANONICAL_EPOCHS,
                        help=f"Training epochs (canonical={CANONICAL_EPOCHS})")
    parser.add_argument("--seeds",          type=int,   nargs="+", default=[RANDOM_SEED])
    parser.add_argument("--embedding-dims", type=int,   nargs="+",
                        default=CANONICAL_EMBEDDING_DIMS,
                        help="Embedding dims for distributed ablation")
    parser.add_argument("--k-neighbors",    type=int,   default=CANONICAL_K_NEIGHBORS)
    parser.add_argument("--train-subset",   type=int,   default=CONTROLLED_TRAIN_SUBSET_SIZE)
    parser.add_argument("--test-subset",    type=int,   default=CONTROLLED_TEST_SUBSET_SIZE)
    parser.add_argument("--bandwidth-mbps", type=float, default=1.0,
                        help="Edge link bandwidth in Mbit/s")
    parser.add_argument(
        "--num-partitions",
        type=int,
        default=0,
        help=(
            "Balanced device partitions. 0 places each graph node on its own "
            "device, so every retained graph edge is a network message."
        ),
    )
    parser.add_argument("--output",         type=str,   default=None)
    args = parser.parse_args()

    # NOTE on multi-seed semantics in this script:
    # The graph is built ONCE (from seeds[0]) and reused across all seeds.
    # The seed sweep therefore measures TRAINING randomness only
    # (weight init, dropout, shuffle of val split), not full pipeline
    # randomness (which would also vary node grouping and graph topology).
    # This is a deliberate choice: keeping the graph fixed isolates the
    # training variance from the graph construction variance, making the
    # accuracy-vs-embedding_dim curve more stable and interpretable.
    # Contrast with main.py, which rebuilds the graph per seed (full pipeline).
    graph, input_dim = build_graph(
        seed=args.seeds[0],
        train_subset=args.train_subset,
        test_subset=args.test_subset,
        k_neighbors=args.k_neighbors,
    )
    num_partitions = args.num_partitions or graph.num_nodes
    graph = assign_balanced_partitions(
        graph, num_partitions=num_partitions, seed=args.seeds[0]
    )
    print(
        f"Partitioning: {graph.num_partitions} partitions, "
        f"{graph.num_boundary_edges}/{graph.num_edges} cross-partition edges"
    )

    scenario = ScenarioConfig(bandwidth_bps=args.bandwidth_mbps * 1e6)

    results = run_three_system_comparison(
        graph=graph,
        input_dim=input_dim,
        embedding_dims=args.embedding_dims,
        epochs=args.epochs,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        dropout=DROPOUT,
        seeds=args.seeds,
        scenario=scenario,
    )

    results.print_summary()

    # Convergence verification — non-negotiable before reporting results
    conv = results.convergence_report
    print("\nConvergence report:")
    for system, status in conv.items():
        icon = "✅" if status["converged"] else "⚠️ "
        print(f"  {icon} {system}: {status['reason']} "
              f"(loss {status.get('first_loss','?')} → {status.get('last_loss','?')})")

    out_path = args.output or str(RESULTS_DIR / "three_system_comparison.json")
    results.save(out_path)


if __name__ == "__main__":
    main()
