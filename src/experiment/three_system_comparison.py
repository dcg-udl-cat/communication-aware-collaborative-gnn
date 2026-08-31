"""Main experiment: Local vs Distributed vs Cloud trade-off comparison.

Produces the core results of the thesis:
  - Accuracy vs Bandwidth curve
  - Accuracy vs Latency curve
  - Accuracy vs Model Size curve

Each point on the distributed curve corresponds to a different embedding_dim,
showing that the distributed GNN can approach cloud accuracy at a fraction
of the communication cost.

Framing note — what "cloud" means here:
  All three systems receive the same PCA-compressed features (default: 128-D).
  The cloud system is therefore "bigger model on the same compressed features,"
  not "bigger model on raw pixels." This is an honest comparison: the cloud
  advantage comes from model capacity and no bandwidth constraint, not from
  a fundamentally richer input representation.

  If you want a raw-pixel cloud baseline, set USE_PCA=False for the cloud
  graph only (not implemented by default to keep comparisons apples-to-apples).
  Documenting this choice in the thesis is important: readers will ask.

Usage:
    from experiment.three_system_comparison import run_three_system_comparison
    results = run_three_system_comparison(graph, input_dim=128, num_classes=10)
    results.print_summary()
    results.save("results/three_system.json")
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Data

from config import (
    CANONICAL_EMBEDDING_DIMS,
    CANONICAL_EPOCHS,
    CANONICAL_K_NEIGHBORS,
    CONTROLLED_TEST_SUBSET_SIZE,
    CONTROLLED_TRAIN_SUBSET_SIZE,
    CONVERGENCE_MIN_LOSS_DROP,
    DEVICE,
    DROPOUT,
    EPOCHS,
    LEARNING_RATE,
    NUM_CLASSES,
    RANDOM_SEED,
    WEIGHT_DECAY,
)
from models.baselines import SimpleMLPBaseline
from models.legacy_cloud_gcn import LegacyCloudGCN
from simulation.cost_model import ScenarioConfig
from systems.cloud_system import CloudSystemResult, run_cloud_system
from systems.distributed_system import DistributedSystemResult, run_distributed_system
from systems.local_system import LocalSystemResult, run_local_system

# Embedding dims to sweep for the distributed system (bandwidth ablation)
DEFAULT_EMBEDDING_DIMS = CANONICAL_EMBEDDING_DIMS


@dataclass
class ThreeSystemResults:
    local: LocalSystemResult
    distributed_variants: list[DistributedSystemResult]   # one per embedding_dim
    cloud: CloudSystemResult
    seeds_used: list[int]
    scenario: ScenarioConfig
    convergence_report: dict = None   # filled by check_convergence()
    majority_baseline_accuracy: float = float("nan")

    # -----------------------------------------------------------------------
    # Derived metrics
    # -----------------------------------------------------------------------

    def efficiency_at(self, embedding_dim: int) -> float | None:
        """Accuracy / total_bytes for a given embedding dim (higher is better)."""
        for r in self.distributed_variants:
            if r.embedding_dim == embedding_dim:
                tb = r.cost.total_bytes
                return r.test_accuracy / tb if tb > 0 else None
        return None

    def check_convergence(
        self,
        min_relative_drop: float = CONVERGENCE_MIN_LOSS_DROP,
    ) -> dict:
        """Verify all systems actually converged during training.

        A system has converged if its final training loss is at least
        `min_relative_drop` lower than its first epoch loss.
        A curve stuck near 2.30 (cross-entropy random baseline for 10 classes)
        has NOT converged and should not be reported as a valid result.

        Returns a report dict with per-system convergence status.
        """
        report = {}

        def _check(name: str, loss_curve: list[float]) -> dict:
            if not loss_curve or len(loss_curve) < 2:
                return {"converged": False, "reason": "too few epochs"}
            first, last = loss_curve[0], loss_curve[-1]
            if first <= 0:
                return {"converged": False, "reason": "invalid first loss"}
            drop = (first - last) / first
            converged = drop >= min_relative_drop
            return {
                "converged": converged,
                "first_loss": round(first, 4),
                "last_loss": round(last, 4),
                "relative_drop": round(drop, 4),
                "reason": "OK" if converged else (
                    f"only {drop:.1%} drop < {min_relative_drop:.0%} threshold"
                ),
            }

        report["local"] = _check("local", self.local.loss_curve)
        for r in self.distributed_variants:
            report[f"distributed_emb{r.embedding_dim}"] = _check(
                f"distributed_{r.embedding_dim}", r.loss_curve
            )
        report["cloud"] = _check("cloud", self.cloud.loss_curve)

        all_converged = all(v["converged"] for v in report.values())
        if not all_converged:
            failed = [k for k, v in report.items() if not v["converged"]]
            print(f"\n⚠️  Convergence warning: {failed} did not converge. "
                  f"Results may be unreliable. Increase CANONICAL_EPOCHS.")
        return report

    def best_distributed(self) -> DistributedSystemResult:
        """Select the embedding dimension using validation accuracy only."""
        return max(
            self.distributed_variants,
            key=lambda r: (
                -math.inf if math.isnan(r.val_accuracy) else r.val_accuracy,
                -r.embedding_dim,
            ),
        )

    def cloud_pct_at(self, embedding_dim: int) -> float | None:
        """Distributed accuracy as % of cloud accuracy for a given embedding_dim."""
        cloud_acc = self.cloud.test_accuracy
        if cloud_acc <= 0:
            return None
        for r in self.distributed_variants:
            if r.embedding_dim == embedding_dim:
                return 100.0 * r.test_accuracy / cloud_acc
        return None

    def bandwidth_reduction_at(self, embedding_dim: int) -> float | None:
        """Bandwidth reduction vs cloud (fraction) for a given embedding_dim."""
        cloud_bytes = self.cloud.cost.total_bytes
        if cloud_bytes <= 0:
            return None
        for r in self.distributed_variants:
            if r.embedding_dim == embedding_dim:
                return 1.0 - r.cost.total_bytes / cloud_bytes
        return None

    # -----------------------------------------------------------------------
    # Output
    # -----------------------------------------------------------------------

    def print_summary(self) -> None:
        c = self.cloud
        l = self.local
        sep = "=" * 72

        print(f"\n{sep}")
        print("THREE-SYSTEM COMPARISON RESULTS")
        print(f"Scenario: bandwidth={self.scenario.bandwidth_bps/1e6:.1f} Mbit/s  "
              f"edge={self.scenario.edge_flops_per_s/1e6:.0f} MFLOPs/s  "
              f"cloud={self.scenario.cloud_flops_per_s/1e9:.0f} GFLOPs/s")
        print(sep)
        print(f"{'System':<24} {'Test acc':>9} {'Bytes/msg':>12} {'Latency ms':>12} "
              f"{'%Cloud':>8} {'%BW saved':>10}")
        print("-" * 72)

        # Local
        print(f"{'Local (no comms)':<24} {l.test_accuracy:>6.3f} "
              f"{'0':>12} {l.cost.total_latency_s*1000:>12.2f} "
              f"{100*l.test_accuracy/c.test_accuracy:>8.1f}%  {'—':>10}")

        # Distributed variants
        for r in sorted(self.distributed_variants, key=lambda x: x.embedding_dim):
            bw_red = self.bandwidth_reduction_at(r.embedding_dim)
            bw_str = f"{bw_red:.0%}" if bw_red is not None else "—"
            cloud_pct = self.cloud_pct_at(r.embedding_dim)
            cloud_str = f"{cloud_pct:.1f}%" if cloud_pct is not None else "—"
            print(f"{'Distributed emb='+str(r.embedding_dim):<24} "
                  f"{r.test_accuracy:>6.3f} "
                  f"{r.cost.bytes_per_node:>12.0f} "
                  f"{r.cost.total_latency_s*1000:>12.2f} "
                  f"{cloud_str:>8}  {bw_str:>10}")

        # Cloud
        print(f"{'Cloud (full access)':<24} {c.test_accuracy:>6.3f} "
              f"{c.cost.bytes_per_node:>12.0f} {c.cost.total_latency_s*1000:>12.2f} "
              f"{'100.0%':>8}  {'0%':>10}")
        print(sep)

        # Validation-selected claim
        best = self.best_distributed()
        bw = self.bandwidth_reduction_at(best.embedding_dim)
        pct = self.cloud_pct_at(best.embedding_dim)
        print(f"\nValidation-selected result: distributed (emb={best.embedding_dim}, "
              f"val={best.val_accuracy:.3f}) achieves "
              f"{pct:.1f}% of cloud accuracy "
              f"using {bw:.0%} less bandwidth than cloud.\n")
        print(f"Majority-class test baseline: {self.majority_baseline_accuracy:.3f}")
        if max(best.test_accuracy, self.cloud.test_accuracy) <= self.majority_baseline_accuracy:
            print("WARNING: learned systems do not beat the majority-class baseline.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "local": self.local.as_dict(),
            "distributed": [r.as_dict() for r in self.distributed_variants],
            "cloud": self.cloud.as_dict(),
            "seeds_used": self.seeds_used,
            "scenario": self.scenario.__dict__,
            "majority_baseline_accuracy": self.majority_baseline_accuracy,
            "selection_metric": "validation_accuracy",
            "selected_embedding_dim": self.best_distributed().embedding_dim,
        }

    def save(self, path: str | Path, plot: bool = True) -> None:
        """Save results JSON and optionally generate all trade-off figures."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.as_dict(), f, indent=2)
        print(f"Results saved to {path}")
        if plot:
            from utils.trade_off_plot import plot_trade_off
            figures_dir = Path(path).parent / "figures"
            plot_trade_off(self, save_dir=figures_dir)


def run_three_system_comparison(
    graph: Data,
    input_dim: int,
    num_classes: int = NUM_CLASSES,
    embedding_dims: list[int] = DEFAULT_EMBEDDING_DIMS,
    hidden_dim: int = 64,
    epochs: int = EPOCHS,
    lr: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    dropout: float = DROPOUT,
    seeds: list[int] | None = None,
    device_str: str = DEVICE,
    scenario: ScenarioConfig | None = None,
) -> ThreeSystemResults:
    """Run all three systems and return the full comparison.

    Multi-seed semantics:
      The same `graph` object is passed to every seed iteration.
      Seeds vary: weight initialisation, dropout masks, val-split shuffle.
      Seeds do NOT vary: node grouping, KNN structure, PCA projection.
      This isolates training variance from graph construction variance.
      See run_three_system.py for the rationale; contrast with main.py
      which rebuilds the full pipeline (including graph) per seed.

    Args:
        graph:          PyG Data with x, edge_index, y, train_mask, test_mask.
        input_dim:      Feature dimension of graph.x (must match graph.num_node_features).
        num_classes:    Number of output classes.
        embedding_dims: List of embedding dims for the distributed system ablation.
        hidden_dim:     Hidden dim for distributed GNN and cloud GNN.
        epochs:         Training epochs for all systems.
        lr:             Learning rate.
        weight_decay:   L2 regularization.
        dropout:        Dropout probability.
        seeds:          Random seeds for multi-seed averaging.
        device_str:     PyTorch device string.
        scenario:       Hardware/network cost scenario config.
    """
    seeds = seeds or [RANDOM_SEED]
    device = torch.device(device_str)
    scenario = scenario or ScenarioConfig()

    # Accumulators for multi-seed averaging
    local_results: list[LocalSystemResult] = []
    dist_results: dict[int, list[DistributedSystemResult]] = {d: [] for d in embedding_dims}
    cloud_results: list[CloudSystemResult] = []

    for seed in seeds:
        print(f"\n{'─'*60}")
        print(f"Seed {seed}")
        print(f"{'─'*60}")

        # --- System 1: Local ---
        print("\n[1/3] Local system")
        local_model = SimpleMLPBaseline(
            input_dim=input_dim, hidden_dim=256, num_classes=num_classes, dropout=dropout
        )
        local_results.append(run_local_system(
            model=local_model, graph=graph.clone(),
            epochs=epochs, lr=lr, weight_decay=weight_decay,
            device=device, scenario=scenario,
            model_flops=_estimate_mlp_flops(
                input_dim, 256, num_classes, num_nodes=graph.num_nodes
            ),
            seed=seed,
        ))

        # --- System 2: Distributed (one per embedding_dim) ---
        print("\n[2/3] Distributed system (embedding dim ablation)")
        for emb_dim in embedding_dims:
            print(f"  embedding_dim={emb_dim}")
            dist_results[emb_dim].append(run_distributed_system(
                input_dim=input_dim, embedding_dim=emb_dim,
                hidden_dim=hidden_dim, num_classes=num_classes,
                graph=graph.clone(),
                epochs=epochs, lr=lr, weight_decay=weight_decay,
                device=device, scenario=scenario,
                model_flops=_estimate_gnn_flops(
                    input_dim,
                    emb_dim,
                    hidden_dim,
                    graph.num_edges,
                    num_nodes=graph.num_nodes,
                    num_classes=num_classes,
                ),
                dropout=dropout, seed=seed,
            ))

        # --- System 3: Cloud ---
        print("\n[3/3] Cloud system")
        # Use the same hidden_dim as the distributed system for a fair FLOPs comparison.
        # FLOPs are derived from the actual model dimensions, not separately-specified args,
        # so cost estimate and model are guaranteed consistent.
        cloud_model = LegacyCloudGCN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
        )
        cloud_flops = _estimate_legacy_cloud_gcn_flops(
            input_dim=input_dim,
            hidden_dim=cloud_model.conv1.out_channels,   # read from model, not args
            num_edges=graph.num_edges,
            num_nodes=graph.num_nodes,
            num_classes=num_classes,
        )
        cloud_results.append(run_cloud_system(
            model=cloud_model, graph=graph.clone(),
            epochs=epochs, lr=lr, weight_decay=weight_decay,
            device=device, scenario=scenario,
            model_flops=cloud_flops,
            seed=seed,
        ))

    # Aggregate
    local_agg = _aggregate_local(local_results)
    dist_agg  = [_aggregate_distributed(dist_results[d]) for d in embedding_dims]
    cloud_agg = _aggregate_cloud(cloud_results)

    result = ThreeSystemResults(
        local=local_agg,
        distributed_variants=dist_agg,
        cloud=cloud_agg,
        seeds_used=seeds,
        scenario=scenario,
        majority_baseline_accuracy=_majority_baseline_accuracy(graph),
    )
    result.convergence_report = result.check_convergence()
    return result


# ---------------------------------------------------------------------------
# FLOP estimators (rough but proportional — sufficient for relative comparison)
# ---------------------------------------------------------------------------

def _estimate_mlp_flops(
    input_dim: int,
    hidden_dim: int,
    num_classes: int,
    *,
    num_nodes: int = 1,
) -> float:
    """2-layer MLP: 2 * (in*hid + hid*hid + hid*out) MACs → FLOPs."""
    per_node = 2.0 * (
        input_dim * hidden_dim
        + hidden_dim * hidden_dim
        + hidden_dim * num_classes
    )
    return num_nodes * per_node


def _estimate_gnn_flops(
    input_dim: int,
    embedding_dim: int,
    hidden_dim: int,
    num_edges: int,
    *,
    num_nodes: int = 1,
    num_classes: int = 10,
) -> float:
    """FLOPs for DistributedTinyGNN: compressor + GATConv + linear out.

    Compressor is Linear(input_dim -> embedding_dim): 2 * input_dim * embedding_dim MACs.
    The previous version incorrectly used embedding_dim**2, which only holds when
    input_dim == embedding_dim (rare in practice; common case is input_dim >> embedding_dim).
    """
    # Linear(input_dim -> embedding_dim): weight matrix multiply
    compressor_flops = num_nodes * 2.0 * input_dim * embedding_dim
    node_projection_flops = num_nodes * 2.0 * embedding_dim * hidden_dim
    # GATConv on embedding_dim -> hidden_dim: per-edge attention + aggregation
    gat_flops = num_edges * 3.0 * hidden_dim
    # Linear(hidden_dim -> num_classes)
    out_flops = num_nodes * 2.0 * hidden_dim * num_classes
    return compressor_flops + node_projection_flops + gat_flops + out_flops


def _estimate_legacy_cloud_gcn_flops(
    input_dim: int,
    hidden_dim: int,
    num_edges: int,
    *,
    num_nodes: int = 1,
    num_classes: int = 10,
) -> float:
    """Approximate work for the legacy CIFAR three-layer GCN."""
    dense = num_nodes * 2.0 * (
        input_dim * hidden_dim
        + hidden_dim * hidden_dim
        + hidden_dim * num_classes
    )
    aggregation = num_edges * (2.0 * hidden_dim + hidden_dim + num_classes)
    return dense + aggregation


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _avg(values: list[float]) -> float:
    return statistics.mean(values) if values else float("nan")


def _uncertainty(values: list[float]) -> tuple[float, float]:
    if len(values) < 2:
        return 0.0, 0.0
    from scipy.stats import t

    std = statistics.stdev(values)
    critical = float(t.ppf(0.975, df=len(values) - 1))
    return std, critical * std / math.sqrt(len(values))


def _majority_baseline_accuracy(graph: Data) -> float:
    train_labels = graph.y[graph.train_mask]
    test_labels = graph.y[graph.test_mask]
    if train_labels.numel() == 0 or test_labels.numel() == 0:
        return float("nan")
    majority = torch.bincount(train_labels).argmax()
    return float((test_labels == majority).float().mean().item())


def _aggregate_local(results: list[LocalSystemResult]) -> LocalSystemResult:
    if len(results) == 1:
        return results[0]
    base = results[0]
    std, ci95 = _uncertainty([r.test_accuracy for r in results])
    return LocalSystemResult(
        test_accuracy=_avg([r.test_accuracy for r in results]),
        train_accuracy=_avg([r.train_accuracy for r in results]),
        num_parameters=base.num_parameters,
        cost=base.cost,
        loss_curve=base.loss_curve,
        embedding_dim=base.embedding_dim,
        val_accuracy=_avg([r.val_accuracy for r in results]),
        test_accuracy_std=std,
        test_accuracy_ci95=ci95,
    )


def _aggregate_distributed(results: list[DistributedSystemResult]) -> DistributedSystemResult:
    if len(results) == 1:
        return results[0]
    base = results[0]
    std, ci95 = _uncertainty([r.test_accuracy for r in results])
    return DistributedSystemResult(
        test_accuracy=_avg([r.test_accuracy for r in results]),
        train_accuracy=_avg([r.train_accuracy for r in results]),
        num_parameters=base.num_parameters,
        embedding_dim=base.embedding_dim,
        cost=base.cost,
        loss_curve=base.loss_curve,
        compression_ratio=base.compression_ratio,
        val_accuracy=_avg([r.val_accuracy for r in results]),
        test_accuracy_std=std,
        test_accuracy_ci95=ci95,
    )


def _aggregate_cloud(results: list[CloudSystemResult]) -> CloudSystemResult:
    if len(results) == 1:
        return results[0]
    base = results[0]
    std, ci95 = _uncertainty([r.test_accuracy for r in results])
    return CloudSystemResult(
        test_accuracy=_avg([r.test_accuracy for r in results]),
        train_accuracy=_avg([r.train_accuracy for r in results]),
        num_parameters=base.num_parameters,
        cost=base.cost,
        loss_curve=base.loss_curve,
        full_feature_dim=base.full_feature_dim,
        val_accuracy=_avg([r.val_accuracy for r in results]),
        test_accuracy_std=std,
        test_accuracy_ci95=ci95,
    )
