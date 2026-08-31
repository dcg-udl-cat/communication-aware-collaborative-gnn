"""Main figure: Accuracy vs Embedding Dimension (the edge/cloud trade-off curve).

This is the central result of the thesis. It shows:
  - Local (MLP): horizontal line — fixed accuracy, zero bandwidth
  - Distributed GNN: a curve of points, one per embedding_dim
  - Cloud GNN: horizontal line — best accuracy, maximum bandwidth

The "sweet spot" is where the distributed curve is close to the cloud line
while using much less bandwidth. That gap is the contribution.

Secondary figures:
  - Accuracy vs Bandwidth (bytes)
  - Accuracy vs Total Latency
  - Loss curves (convergence verification)

Usage:
    from utils.trade_off_plot import plot_trade_off
    plot_trade_off(results, save_dir="results/figures/")
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from experiment.three_system_comparison import ThreeSystemResults


def plot_trade_off(
    results: "ThreeSystemResults",
    save_dir: str | Path = "results/figures",
    show: bool = False,
) -> list[Path]:
    """Generate all trade-off figures and save to disk.

    Returns list of saved file paths.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")          # headless — safe for servers
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mtick
    except ImportError:
        print("matplotlib not available — skipping plots. "
              "Install with: pip install matplotlib")
        return []

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    saved.append(_plot_accuracy_vs_embedding_dim(results, plt, save_dir, show))
    saved.append(_plot_accuracy_vs_bandwidth(results, plt, save_dir, show))
    saved.append(_plot_accuracy_vs_latency(results, plt, save_dir, show))
    saved.append(_plot_loss_curves(results, plt, save_dir, show))

    print(f"\nFigures saved to {save_dir}:")
    for p in saved:
        print(f"  {p.name}")
    return saved


# ---------------------------------------------------------------------------
# Figure 1: Accuracy vs Embedding Dimension — THE MAIN FIGURE
# ---------------------------------------------------------------------------

def _plot_accuracy_vs_embedding_dim(results, plt, save_dir, show):
    """Main thesis figure.

    X-axis: embedding dimension (log scale — 8, 16, 32, 64)
    Y-axis: test accuracy

    Three elements:
      - Shaded horizontal band: cloud accuracy ± 0 (reference ceiling)
      - Solid curve with markers: distributed GNN across embedding dims
      - Dashed horizontal line: local MLP (lower bound)
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))

    dist = sorted(results.distributed_variants, key=lambda r: r.embedding_dim)
    emb_dims   = [r.embedding_dim    for r in dist]
    dist_accs  = [r.test_accuracy    for r in dist]
    dist_bytes = [r.cost.bytes_per_node for r in dist]

    cloud_acc = results.cloud.test_accuracy
    local_acc = results.local.test_accuracy

    # Cloud ceiling (shaded band)
    ax.axhline(cloud_acc, color="#2196F3", linewidth=2.0,
               linestyle="-", label=f"Cloud GNN  ({cloud_acc:.3f})", zorder=3)
    ax.axhspan(cloud_acc - 0.005, cloud_acc + 0.005,
               color="#2196F3", alpha=0.12, zorder=1)

    # Local lower bound
    ax.axhline(local_acc, color="#9E9E9E", linewidth=1.5,
               linestyle="--", label=f"Local MLP  ({local_acc:.3f})", zorder=3)

    # Distributed curve (the contribution)
    ax.plot(emb_dims, dist_accs, color="#F44336", linewidth=2.5,
            marker="o", markersize=8, zorder=4,
            label="Distributed GNN (ours)")

    # Annotate each point with bandwidth cost
    for x, y, b in zip(emb_dims, dist_accs, dist_bytes):
        ax.annotate(f"{int(b)} B/node",
                    xy=(x, y), xytext=(0, 10),
                    textcoords="offset points",
                    ha="center", fontsize=8, color="#F44336")

    # Mark the sweet spot: highest accuracy within 5% of cloud
    threshold = cloud_acc * 0.95
    candidates = [(e, a) for e, a in zip(emb_dims, dist_accs) if a >= threshold]
    if candidates:
        best_e, best_a = min(candidates, key=lambda t: t[0])  # smallest emb_dim
        ax.annotate("sweet spot",
                    xy=(best_e, best_a), xytext=(best_e * 1.3, best_a - 0.02),
                    arrowprops=dict(arrowstyle="->", color="#4CAF50", lw=1.5),
                    color="#4CAF50", fontsize=9, fontweight="bold")

    ax.set_xscale("log", base=2)
    ax.set_xticks(emb_dims)
    ax.set_xticklabels([str(d) for d in emb_dims])
    ax.set_xlabel("Embedding dimension (floats transmitted per node)", fontsize=11)
    ax.set_ylabel("Test accuracy", fontsize=11)
    ax.set_title("Accuracy vs Communication Cost\n"
                 "(main result: distributed GNN sweet spot)", fontsize=12)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(max(0, min(local_acc, min(dist_accs)) - 0.05),
                min(1.0, cloud_acc + 0.08))
    ax.grid(True, alpha=0.3)

    _add_bandwidth_twin_axis(ax, emb_dims, plt)

    fig.tight_layout()
    path = save_dir / "fig1_accuracy_vs_embedding_dim.pdf"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return path


def _add_bandwidth_twin_axis(ax, emb_dims, plt):
    """Add a secondary x-axis showing bytes/node instead of embedding dim."""
    ax2 = ax.twiny()
    ax2.set_xscale("log", base=2)
    ax2.set_xlim(ax.get_xlim())
    byte_labels = [f"{d*4}B" for d in emb_dims]
    ax2.set_xticks(emb_dims)
    ax2.set_xticklabels(byte_labels, fontsize=8, color="#555")
    ax2.set_xlabel("Bytes per node (float32)", fontsize=9, color="#555")


# ---------------------------------------------------------------------------
# Figure 2: Accuracy vs Bandwidth (total bytes)
# ---------------------------------------------------------------------------

def _plot_accuracy_vs_bandwidth(results, plt, save_dir, show):
    fig, ax = plt.subplots(figsize=(6, 4))

    dist = sorted(results.distributed_variants, key=lambda r: r.cost.total_bytes)
    xs = [0] + [r.cost.total_bytes   for r in dist] + [results.cloud.cost.total_bytes]
    ys = ([results.local.test_accuracy]
          + [r.test_accuracy          for r in dist]
          + [results.cloud.test_accuracy])
    labels = (["Local"] +
              [f"emb={r.embedding_dim}" for r in dist] +
              ["Cloud"])
    colors = (["#9E9E9E"] +
              ["#F44336"] * len(dist) +
              ["#2196F3"])
    markers = ["s"] + ["o"] * len(dist) + ["^"]

    for x, y, lbl, c, m in zip(xs, ys, labels, colors, markers):
        ax.scatter(x, y, color=c, marker=m, s=80, zorder=4, label=lbl)

    # Connect distributed points as a curve
    dist_xs = [r.cost.total_bytes for r in dist]
    dist_ys = [r.test_accuracy for r in dist]
    ax.plot(dist_xs, dist_ys, color="#F44336", linewidth=2, zorder=3)

    ax.set_xlabel("Total bytes transmitted (all nodes)", fontsize=11)
    ax.set_ylabel("Test accuracy", fontsize=11)
    ax.set_title("Accuracy vs Bandwidth", fontsize=12)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = save_dir / "fig2_accuracy_vs_bandwidth.pdf"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 3: Accuracy vs Latency
# ---------------------------------------------------------------------------

def _plot_accuracy_vs_latency(results, plt, save_dir, show):
    fig, ax = plt.subplots(figsize=(6, 4))

    dist = sorted(results.distributed_variants, key=lambda r: r.cost.total_latency_s)

    systems = (
        [("Local", results.local.test_accuracy,
          results.local.cost.total_latency_s * 1000, "#9E9E9E", "s")]
        + [(f"emb={r.embedding_dim}", r.test_accuracy,
            r.cost.total_latency_s * 1000, "#F44336", "o") for r in dist]
        + [("Cloud", results.cloud.test_accuracy,
            results.cloud.cost.total_latency_s * 1000, "#2196F3", "^")]
    )

    for label, acc, lat_ms, color, marker in systems:
        ax.scatter(lat_ms, acc, color=color, marker=marker, s=80, zorder=4, label=label)

    dist_lats = [r.cost.total_latency_s * 1000 for r in dist]
    dist_accs = [r.test_accuracy for r in dist]
    ax.plot(dist_lats, dist_accs, color="#F44336", linewidth=2, zorder=3)

    ax.set_xlabel("Total latency (ms)", fontsize=11)
    ax.set_ylabel("Test accuracy", fontsize=11)
    ax.set_title("Accuracy vs Latency", fontsize=12)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = save_dir / "fig3_accuracy_vs_latency.pdf"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 4: Loss curves — convergence verification
# ---------------------------------------------------------------------------

def _plot_loss_curves(results, plt, save_dir, show):
    """Plot training loss curves for all systems.

    If any curve is flat near 2.30, the model did not converge.
    This figure is the convergence sanity check — include it in the appendix.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)

    def _plot_curve(ax, curve, label, color):
        if not curve:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center")
            return
        ax.plot(range(1, len(curve) + 1), curve, color=color, linewidth=2)
        ax.axhline(2.302, color="#BDBDBD", linewidth=1, linestyle=":",
                   label="random baseline (ln 10 ≈ 2.30)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training loss")
        ax.set_title(label)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        # Annotate convergence
        first, last = curve[0], curve[-1]
        drop = (first - last) / first if first > 0 else 0
        converged = drop >= 0.10
        status = f"✅ -{drop:.0%}" if converged else f"⚠️  -{drop:.0%} (not converged)"
        ax.set_title(f"{label}\n{status}", fontsize=10)

    _plot_curve(axes[0], results.local.loss_curve,        "Local MLP",     "#9E9E9E")
    best_dist = results.best_distributed()
    _plot_curve(axes[1], best_dist.loss_curve,
                f"Distributed GNN (emb={best_dist.embedding_dim})", "#F44336")
    _plot_curve(axes[2], results.cloud.loss_curve,        "Cloud GNN",     "#2196F3")

    fig.suptitle("Training Loss Curves (convergence verification)", fontsize=12)
    fig.tight_layout()
    path = save_dir / "fig4_loss_curves.pdf"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return path
