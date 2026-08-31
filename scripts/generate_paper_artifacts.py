#!/usr/bin/env python3
"""Generate every numerical paper figure/table from append-only raw records."""

from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
PAPER_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "paper"
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "gnn-paper-matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from utils.paper_results import read_jsonl, scientific_gate, summarize_jsonl

COLORS = {
    "local": "#7f8c8d",
    "full": "#2980b9",
    "full_int8": "#16a085",
    "learned": "#c0392b",
    "pca": "#8e44ad",
    "learned_int8": "#d35400",
}
LABELS = {
    "local": "Local",
    "full": "Full feature",
    "full_int8": "Full + int8 upload",
    "learned": "Learned",
    "pca": "PCA",
    "learned_int8": "Learned int8",
}


def generate(
    raw_path: str | Path,
    output_dir: str | Path,
    *,
    required_seeds: int = 5,
) -> list[Path]:
    output_dir = Path(output_dir)
    figures = output_dir / "figures"
    tables = output_dir / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    records = [record for record in read_jsonl(raw_path) if record.get("status") == "ok"]
    if not records:
        raise ValueError("No successful raw experiment records were found.")
    frame = pd.DataFrame(records)
    saved = [
        _pareto(frame, figures),
        _network_profiles(frame, figures),
        _partition_sensitivity(frame, figures),
        _resources(frame, figures),
    ]
    summary = summarize_jsonl(raw_path, tables / "supplementary_results.csv")
    _main_table(summary, tables / "main_results.tex")
    _partition_table(frame, tables / "partition_results.tex")
    _result_macros(records, output_dir / "results_macros.tex", required_seeds)
    return saved


def _canonical(frame: pd.DataFrame) -> pd.DataFrame:
    canonical = frame[
        (frame["partition_strategy"] == "random")
        & (frame["num_partitions"] == 4)
        & (frame["network_profile"] == "constrained")
    ].copy()
    if not canonical.empty:
        return canonical
    first = frame.sort_values(
        ["partition_strategy", "num_partitions", "network_profile"]
    ).iloc[0]
    return frame[
        (frame["partition_strategy"] == first["partition_strategy"])
        & (frame["num_partitions"] == first["num_partitions"])
        & (frame["network_profile"] == first["network_profile"])
    ].copy()


def _pareto(frame: pd.DataFrame, figures: Path) -> Path:
    canonical = _canonical(frame)
    datasets = sorted(canonical["dataset"].unique())
    cols = 2
    rows = math.ceil(len(datasets) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(10, 4 * rows), squeeze=False)
    for ax, dataset in zip(axes.flat, datasets):
        subset = canonical[canonical["dataset"] == dataset]
        grouped = subset.groupby(["system", "embedding_dim"], as_index=False).agg(
            accuracy=("test_accuracy", "mean"),
            bytes=("serialized_wire_bytes", "mean"),
        )
        local = grouped[grouped["system"] == "local"]
        for system, points in grouped[grouped["system"] != "local"].groupby("system"):
            points = points.sort_values("bytes")
            ax.plot(
                points["bytes"],
                points["accuracy"],
                marker="o",
                label=LABELS[system],
                color=COLORS.get(system),
            )
            for _, point in points.iterrows():
                if system in {"learned", "pca"}:
                    offset = (3, 3) if system == "learned" else (3, -10)
                    ax.annotate(
                        str(int(point["embedding_dim"])),
                        (max(point["bytes"], 1), point["accuracy"]),
                        xytext=offset,
                        textcoords="offset points",
                        fontsize=7,
                    )
        if not local.empty:
            local_accuracy = local["accuracy"].mean()
            ax.axhline(
                local_accuracy,
                color=COLORS["local"],
                linestyle=":",
                linewidth=1,
                label="Local: 0 B",
            )
            ax.annotate(
                "Local: 0 B",
                (0.01, local_accuracy),
                xycoords=("axes fraction", "data"),
                xytext=(3, 3),
                textcoords="offset points",
                color=COLORS["local"],
                fontsize=7,
            )
        ax.set_xscale("log")
        ax.set_title(dataset.title())
        ax.set_xlabel("Serialized bytes per inference round")
        ax.set_ylabel("Test accuracy")
        ax.grid(alpha=0.25)
    for ax in axes.flat[len(datasets):]:
        ax.axis("off")
    legend = {
        label: handle
        for ax in axes.flat
        for handle, label in zip(*ax.get_legend_handles_labels())
    }
    fig.legend(legend.values(), legend.keys(), loc="upper center", ncol=max(1, len(legend)))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, figures / "accuracy_vs_wire_bytes")


def _network_profiles(frame: pd.DataFrame, figures: Path) -> Path:
    canonical = frame[
        (frame["partition_strategy"] == "random")
        & (frame["num_partitions"] == 4)
    ].copy()
    if canonical.empty:
        first = frame.sort_values(
            ["partition_strategy", "num_partitions"]
        ).iloc[0]
        canonical = frame[
            (frame["partition_strategy"] == first["partition_strategy"])
            & (frame["num_partitions"] == first["num_partitions"])
        ].copy()
    selected = _selected_rows(canonical)
    selected = selected[~selected["system"].isin(["pca", "learned_int8"])]
    grouped = selected.groupby(["dataset", "system", "network_profile"], as_index=False).agg(
        accuracy=("test_accuracy", "mean"),
        latency=("composite_latency_s", "mean"),
    )
    datasets = sorted(grouped["dataset"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), squeeze=False)
    for ax, dataset in zip(axes.flat, datasets):
        subset = grouped[grouped["dataset"] == dataset]
        for system, points in subset.groupby("system"):
            points = points.sort_values("latency")
            ax.plot(
                points["latency"] * 1000,
                points["accuracy"],
                marker="o",
                label=LABELS[system],
                color=COLORS.get(system),
                alpha=0.85,
            )
        ax.set_xscale("log")
        ax.set_title(dataset.title())
        ax.set_xlabel("Composite emulated latency (ms)")
        ax.set_ylabel("Test accuracy")
        ax.grid(alpha=0.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, figures / "network_regimes")


def _partition_sensitivity(frame: pd.DataFrame, figures: Path) -> Path:
    learned = frame[frame["system"] == "learned"].copy()
    learned = _selected_rows(learned)
    deduplicated = learned.drop_duplicates(
        ["dataset", "partition_strategy", "num_partitions", "partition_seed"]
    )
    grouped = deduplicated.groupby(
        ["partition_strategy", "num_partitions"], as_index=False
    )["boundary_edge_ratio"].mean()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for strategy, points in grouped.groupby("partition_strategy"):
        ax.plot(
            points["num_partitions"],
            points["boundary_edge_ratio"],
            marker="o",
            label=strategy,
        )
    ax.set_xlabel("Number of logical partitions")
    ax.set_ylabel("Mean boundary-edge ratio")
    ax.set_title("Partition sensitivity")
    ax.set_xticks(sorted(grouped["num_partitions"].unique()))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return _save(fig, figures / "partition_sensitivity")


def _resources(frame: pd.DataFrame, figures: Path) -> Path:
    canonical = _selected_rows(_canonical(frame))
    canonical = canonical[~canonical["system"].isin(["full_int8", "learned_int8"])]
    deduplicated = canonical.drop_duplicates(
        ["dataset", "seed", "system", "embedding_dim"]
    )
    grouped = deduplicated.groupby(["dataset", "system"], as_index=False).agg(
        latency=("inference_median_s", "mean"),
        memory=("peak_rss_mb", "mean"),
    )
    datasets = sorted(grouped["dataset"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), squeeze=False)
    for ax, dataset in zip(axes.flat, datasets):
        subset = grouped[grouped["dataset"] == dataset]
        for _, point in subset.iterrows():
            ax.scatter(
                point["latency"] * 1000,
                point["memory"],
                s=70,
                color=COLORS.get(point["system"]),
                label=LABELS[point["system"]],
            )
        ax.set_title(dataset.title())
        ax.set_xlabel("Reference-host inference latency (ms)")
        ax.set_ylabel("Isolated worker peak RSS (MiB)")
        ax.grid(alpha=0.25)
    legend = {
        label: handle
        for ax in axes.flat
        for handle, label in zip(*ax.get_legend_handles_labels())
    }
    fig.legend(legend.values(), legend.keys(), loc="upper center", ncol=max(1, len(legend)))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, figures / "resource_measurements")


def _selected_rows(frame: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for dataset, dataset_rows in frame.groupby("dataset"):
        learned = dataset_rows[dataset_rows["system"] == "learned"]
        if not learned.empty:
            selected_dim = learned.groupby("embedding_dim")["val_accuracy"].mean().idxmax()
            keep = ~dataset_rows["system"].isin(["learned", "learned_int8"])
            keep |= dataset_rows["embedding_dim"] == selected_dim
            pieces.append(dataset_rows[keep])
        else:
            pieces.append(dataset_rows)
    return pd.concat(pieces, ignore_index=True) if pieces else frame.iloc[0:0]


def _main_table(summary: pd.DataFrame, path: Path) -> None:
    if summary.empty:
        path.write_text("% No completed results.\n", encoding="utf-8")
        return
    canonical = summary[
        (summary["partition_strategy"] == "random")
        & (summary["num_partitions"] == 4)
        & (summary["network_profile"] == "constrained")
    ]
    rows = []
    for dataset, group in canonical.groupby("dataset", sort=True):
        learned = group[group["system"] == "learned"]
        selected_dim = int(
            learned.sort_values(["val_accuracy_mean", "embedding_dim"], ascending=[False, True])
            .iloc[0]["embedding_dim"]
        )
        keep = group["system"].isin(["local", "full", "full_int8"])
        keep |= group["system"].isin(["learned", "learned_int8"]) & (
            group["embedding_dim"] == selected_dim
        )
        selected = group[keep].copy()
        selected["system_order"] = selected["system"].map(
            {"local": 0, "full": 1, "full_int8": 2, "learned": 3, "learned_int8": 4}
        )
        for _, row in selected.sort_values("system_order").iterrows():
            system = str(row["system"])
            rows.append(
                {
                    "Dataset": dataset.title(),
                    "System": LABELS[system],
                    "$k$": "--" if system in {"local", "full", "full_int8"} else str(selected_dim),
                    "Accuracy (95\\% CI)": (
                        f"${row['test_accuracy_mean']:.4f} \\pm "
                        f"{row['test_accuracy_ci95']:.4f}$"
                    ),
                    "Macro-F1 (95\\% CI)": (
                        f"${row['test_macro_f1_mean']:.4f} \\pm "
                        f"{row['test_macro_f1_ci95']:.4f}$"
                    ),
                    "Wire bytes": f"{row['serialized_wire_bytes_mean']:,.0f}",
                }
            )
    path.write_text(
        pd.DataFrame(rows).to_latex(index=False, escape=False),
        encoding="utf-8",
    )


def _partition_table(frame: pd.DataFrame, path: Path) -> None:
    learned = _selected_rows(frame[frame["system"] == "learned"])
    deduplicated = learned.drop_duplicates(
        ["dataset", "partition_strategy", "num_partitions", "partition_seed"]
    )
    grouped = deduplicated.groupby(
        ["partition_strategy", "num_partitions"], as_index=False
    ).agg(
        boundary_edge_ratio=("boundary_edge_ratio", "mean"),
        unique_transfers=("unique_transfers", "mean"),
        serialized_wire_bytes=("serialized_wire_bytes", "mean"),
    )

    lines = [
        r"\begin{tabular}{r rrr rrr}",
        r"\toprule",
        r"& \multicolumn{3}{c}{Random} & \multicolumn{3}{c}{Locality-aware (RCM)} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"$P$ & $\rho_{\mathrm{cut}}$ & $|T|$ & Serialized (MB) & $\rho_{\mathrm{cut}}$ & $|T|$ & Serialized (MB) \\",
        r"\midrule",
    ]
    for partition_count in (2, 4, 8):
        values = []
        for strategy in ("random", "locality"):
            match = grouped[
                (grouped["partition_strategy"] == strategy)
                & (grouped["num_partitions"] == partition_count)
            ]
            if len(match) != 1:
                raise ValueError(
                    f"Expected one {strategy} aggregate for P={partition_count}."
                )
            row = match.iloc[0]
            values.extend(
                [
                    f"{row['boundary_edge_ratio']:.3f}",
                    f"{row['unique_transfers']:,.0f}",
                    f"{row['serialized_wire_bytes'] / 1_000_000:.2f}",
                ]
            )
        lines.append(f"{partition_count} & " + " & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _result_macros(records, path: Path, required_seeds: int) -> None:
    gate = scientific_gate(records, required_seeds=required_seeds)
    datasets = sorted({record["dataset"] for record in records})
    seeds = sorted({record["seed"] for record in records})
    outcome = "passed" if gate["passed"] else "did not pass"
    passed = ", ".join(gate["passed_datasets"]) or "none"
    path.write_text(
        "%% Generated from raw/results.jsonl; do not edit.\n"
        f"\\newcommand{{\\CompletedDatasetCount}}{{{len(datasets)}}}\n"
        f"\\newcommand{{\\CompletedSeedCount}}{{{len(seeds)}}}\n"
        f"\\newcommand{{\\ScientificGateOutcome}}{{{outcome}}}\n"
        f"\\newcommand{{\\GateDatasets}}{{{passed}}}\n",
        encoding="utf-8",
    )


def _save(fig, stem: Path) -> Path:
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)
    return stem.with_suffix(".pdf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw",
        default=str(PAPER_ARTIFACT_ROOT / "results" / "raw" / "results.jsonl"),
    )
    parser.add_argument("--output-dir", default=str(PAPER_ARTIFACT_ROOT / "generated"))
    parser.add_argument("--required-seeds", type=int, default=5)
    args = parser.parse_args()
    for path in generate(
        args.raw,
        args.output_dir,
        required_seeds=args.required_seeds,
    ):
        print(path)


if __name__ == "__main__":
    main()
