#!/usr/bin/env python3
"""Resumable experiment matrix for the Future Internet submission artifact."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "paper" / "results"

from config import DATA_DIR
from data.benchmark_loader import SUPPORTED_DATASETS, load_graph_benchmark
from experiment.paper_experiment import (
    load_partitioned_int8_model,
    partitioned_int8_logits,
    run_paper_model,
)
from simulation.communication import centralized_upload_size, communication_trace
from simulation.network_emulation import LINK_PROFILES, modeled_transfer_s, replay_profile
from simulation.partitioning import assign_partitions
from training.splits import make_inductive_view, resolve_train_val_masks
from training.evaluator import classification_metrics_from_logits
from utils.paper_results import (
    append_record,
    read_jsonl,
    record_key,
    scientific_gate,
    summarize_jsonl,
)

DEFAULT_DATASETS = ["cora", "citeseer", "pubmed", "flickr"]
DEFAULT_SEEDS = [42, 123, 2024, 2025, 2026]


def main() -> None:
    args = _parser().parse_args()
    if args.worker:
        _worker(args)
        return

    output_dir = Path(args.output_dir)
    if args.smoke and output_dir == DEFAULT_OUTPUT_DIR:
        output_dir /= "smoke"
    # Keep worker scratch files and checkpoints inside the resolved result tree.
    args.output_dir = str(output_dir)
    raw_path = output_dir / "raw" / "results.jsonl"
    summary_path = output_dir / "summary.csv"
    gate_path = output_dir / "scientific_gate.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = min(args.epochs, 5) if args.smoke else args.epochs
    seeds = args.seeds[:1] if args.smoke else args.seeds
    existing_keys = {
        record_key(record)
        for record in read_jsonl(raw_path)
        if record.get("status") == "ok"
    }
    calibrations = _calibrations(args)
    (output_dir / "network_calibration.json").write_text(
        json.dumps(calibrations, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    graph_cache = {}

    tasks = []
    for dataset in args.datasets:
        for seed in seeds:
            tasks.extend([(dataset, seed, "local", None), (dataset, seed, "full", None)])
            if args.include_ablations:
                tasks.append((dataset, seed, "full_int8", None))
            tasks.extend((dataset, seed, "learned", dim) for dim in args.embedding_dims)
            if args.include_ablations and dataset in {"cora", "flickr"}:
                tasks.extend((dataset, seed, "pca", dim) for dim in args.pca_dims)

    for task in tasks:
        _run_and_expand(
            args,
            task,
            epochs,
            raw_path,
            existing_keys,
            calibrations,
            graph_cache,
        )

    if args.include_ablations:
        selected = _validation_selected_dims(
            read_jsonl(raw_path),
            args.datasets,
            args.message_protocol,
            args.feature_normalization,
        )
        for dataset, embedding_dim in selected.items():
            for seed in seeds:
                _run_and_expand(
                    args,
                    (dataset, seed, "learned_int8", embedding_dim),
                    epochs,
                    raw_path,
                    existing_keys,
                    calibrations,
                    graph_cache,
                )

    summarize_jsonl(raw_path, summary_path)
    gate = scientific_gate(
        read_jsonl(raw_path),
        required_seeds=1 if args.smoke else len(seeds),
        expected_embedding_dims=tuple(args.embedding_dims),
        feature_normalization=args.feature_normalization,
    )
    gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "environment.json").write_text(
        json.dumps(
            _environment(args.message_protocol, args.feature_normalization),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Raw records: {raw_path}")
    print(f"Summary:     {summary_path}")
    print(f"Gate:        {'PASS' if gate['passed'] else 'NOT PASSED'} ({gate_path})")


def _run_and_expand(
    args,
    task,
    epochs,
    raw_path,
    existing_keys,
    calibrations,
    graph_cache,
) -> None:
    dataset, seed, system, embedding_dim = task
    if _task_complete(args, existing_keys, dataset, seed, system, embedding_dim):
        print(f"[resume] {dataset}/{system}/emb={embedding_dim}/seed={seed}")
        return

    checkpoint = _checkpoint_path(
        Path(args.output_dir),
        dataset,
        seed,
        system,
        embedding_dim,
        epochs,
        args.message_protocol,
        args.feature_normalization,
    )
    source_checkpoint = None
    if system == "learned_int8":
        source_checkpoint = _checkpoint_path(
            Path(args.output_dir),
            dataset,
            seed,
            "learned",
            embedding_dim,
            epochs,
            args.message_protocol,
            args.feature_normalization,
        )
        if not source_checkpoint.exists() and args.checkpoint_source_dir:
            source_checkpoint = _checkpoint_path(
                Path(args.checkpoint_source_dir),
                dataset,
                seed,
                "learned",
                embedding_dim,
                epochs,
                args.message_protocol,
                args.feature_normalization,
            )
    elif system == "full_int8":
        source_checkpoint = _checkpoint_path(
            Path(args.output_dir),
            dataset,
            seed,
            "full",
            None,
            epochs,
            args.message_protocol,
            args.feature_normalization,
        )
        if not source_checkpoint.exists() and args.checkpoint_source_dir:
            source_checkpoint = _checkpoint_path(
                Path(args.checkpoint_source_dir),
                dataset,
                seed,
                "full",
                None,
                epochs,
                args.message_protocol,
                args.feature_normalization,
            )
    elif args.checkpoint_source_dir and not checkpoint.exists():
        source_checkpoint = _checkpoint_path(
            Path(args.checkpoint_source_dir),
            dataset,
            seed,
            system,
            embedding_dim,
            epochs,
            args.message_protocol,
            args.feature_normalization,
        )

    try:
        model_record = _invoke_worker(
            args,
            dataset=dataset,
            seed=seed,
            system=system,
            embedding_dim=embedding_dim,
            epochs=epochs,
            checkpoint=checkpoint,
            source_checkpoint=source_checkpoint,
        )
        if dataset not in graph_cache:
            graph_cache[dataset] = load_graph_benchmark(dataset, DATA_DIR / "benchmarks")
        for record in _expand_model_record(
            model_record,
            graph_cache[dataset],
            args,
            calibrations,
            source_checkpoint=source_checkpoint,
        ):
            key = record_key(record)
            if key not in existing_keys:
                append_record(raw_path, record)
                existing_keys.add(key)
    except Exception as error:
        failure = {
            "record_type": "experiment",
            "status": "failed",
            "dataset": dataset,
            "seed": seed,
            "system": system,
            "embedding_dim": embedding_dim,
            "message_protocol": args.message_protocol,
            "partition_strategy": None,
            "num_partitions": None,
            "partition_seed": None,
            "network_profile": None,
            "error": f"{type(error).__name__}: {error}",
        }
        append_record(raw_path, failure)
        print(f"[failed] {dataset}/{system}/emb={embedding_dim}/seed={seed}: {error}")


def _invoke_worker(
    args,
    *,
    dataset,
    seed,
    system,
    embedding_dim,
    epochs,
    checkpoint,
    source_checkpoint,
):
    with tempfile.NamedTemporaryFile(
        suffix=".json", dir=Path(args.output_dir), delete=False
    ) as temporary:
        output_path = Path(temporary.name)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--worker-output",
        str(output_path),
        "--worker-system",
        system,
        "--datasets",
        dataset,
        "--seeds",
        str(seed),
        "--epochs",
        str(epochs),
        "--message-protocol",
        args.message_protocol,
        "--feature-normalization",
        args.feature_normalization,
        "--checkpoint",
        str(checkpoint),
    ]
    if embedding_dim is not None:
        command += ["--worker-embedding-dim", str(embedding_dim)]
    if source_checkpoint is not None:
        command += ["--source-checkpoint", str(source_checkpoint)]
    try:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        return json.loads(output_path.read_text(encoding="utf-8"))
    finally:
        output_path.unlink(missing_ok=True)


def _worker(args) -> None:
    dataset = args.datasets[0]
    graph = load_graph_benchmark(dataset, DATA_DIR / "benchmarks")
    record = run_paper_model(
        graph,
        dataset=dataset,
        system=args.worker_system,
        seed=args.seeds[0],
        epochs=args.epochs,
        embedding_dim=args.worker_embedding_dim,
        checkpoint_path=args.checkpoint,
        source_checkpoint_path=args.source_checkpoint,
        message_protocol=args.message_protocol,
        feature_normalization=args.feature_normalization,
    )
    Path(args.worker_output).write_text(
        json.dumps(record, sort_keys=True, allow_nan=False), encoding="utf-8"
    )


def _expand_model_record(
    model_record,
    graph,
    args,
    calibrations,
    *,
    source_checkpoint=None,
):
    train_mask, _, _ = resolve_train_val_masks(graph, seed=int(model_record["seed"]))
    view = (
        make_inductive_view(graph, train_mask)
        if model_record["system"] != "local" and args.message_protocol == "source-only"
        else graph
    )
    int8_context = None
    if model_record["system"] == "learned_int8":
        if source_checkpoint is None or not Path(source_checkpoint).exists():
            raise FileNotFoundError("Protocol-faithful int8 evaluation needs the learned checkpoint.")
        int8_context = load_partitioned_int8_model(
            graph,
            model_record,
            source_checkpoint,
        )
        _, view, train_mask, _ = int8_context
    for strategy in args.partition_strategies:
        for partition_count in args.partition_counts:
            partition_seeds = args.partition_seeds if strategy == "random" else [0]
            for partition_seed in partition_seeds:
                partitioned = assign_partitions(
                    view,
                    partition_count,
                    strategy=strategy,
                    seed=partition_seed,
                )
                int8_metrics = {}
                if int8_context is not None:
                    model, _, int8_train_mask, int8_val_mask = int8_context
                    logits = partitioned_int8_logits(model, partitioned)
                    for split, mask in (
                        ("train", int8_train_mask),
                        ("val", int8_val_mask),
                        ("test", partitioned.test_mask),
                    ):
                        metrics = classification_metrics_from_logits(
                            logits,
                            partitioned.y,
                            mask,
                        )
                        int8_metrics[f"{split}_accuracy"] = metrics.accuracy
                        int8_metrics[f"{split}_macro_f1"] = metrics.macro_f1
                    int8_metrics["accuracy_measurement"] = (
                        "remote_embeddings_int8_per_source_destination_batch; "
                        "local_embeddings_float32"
                    )
                trace = communication_trace(
                    partitioned,
                    int(model_record["embedding_dim"]),
                    bytes_per_value=int(model_record["bytes_per_value"]),
                )
                communication = trace.as_dict()
                if model_record["system"] == "local":
                    communication.update(
                        unique_transfers=0,
                        transfer_batches=0,
                        raw_payload_bytes=0,
                        serialized_wire_bytes=0,
                        conservative_edge_payload_bytes=0,
                    )
                elif model_record["system"] in {"full", "full_int8"}:
                    nodes = int(model_record["num_nodes"])
                    dim = int(model_record["input_dim"])
                    raw, wire = centralized_upload_size(
                        nodes,
                        dim,
                        bytes_per_value=int(model_record["bytes_per_value"]),
                    )
                    communication.update(
                        unique_transfers=nodes,
                        transfer_batches=1,
                        raw_payload_bytes=raw,
                        serialized_wire_bytes=wire,
                        conservative_edge_payload_bytes=raw,
                    )

                for profile_name in args.network_profiles:
                    profile = LINK_PROFILES[profile_name]
                    wire_bytes = int(communication["serialized_wire_bytes"])
                    transfer_s = 0.0 if model_record["system"] == "local" else modeled_transfer_s(wire_bytes, profile)
                    record = deepcopy(model_record)
                    record.update(
                        record_type="experiment",
                        partition_strategy=strategy,
                        num_partitions=partition_count,
                        partition_seed=partition_seed,
                        network_profile=profile_name,
                        bandwidth_bps=profile.bandwidth_bps,
                        rtt_ms=profile.rtt_ms,
                        modeled_transfer_s=transfer_s,
                        composite_latency_s=float(model_record["inference_median_s"]) + transfer_s,
                        representation_reduction=(
                            float(model_record["input_dim"]) / float(model_record["embedding_dim"])
                            if model_record["system"] not in {"local", "full", "full_int8"} else 1.0
                        ),
                        **communication,
                        **calibrations[profile_name],
                        **int8_metrics,
                    )
                    yield record


def _calibrations(args):
    results = {}
    for profile_name in args.network_profiles:
        if args.skip_calibration:
            results[profile_name] = {
                "calibration_method": "not_run",
                "calibration_passed": False,
                "calibration_bandwidth_error_fraction": None,
                "calibration_observed_rtt_s": None,
                "calibration_rtt_error_fraction": None,
                "calibration_rtt_error_q1_fraction": None,
                "calibration_rtt_error_q3_fraction": None,
                "calibration_rtt_within_tolerance": None,
                "calibration_bandwidth_error_q1_fraction": None,
                "calibration_bandwidth_error_q3_fraction": None,
                "calibration_repeats": 0,
            }
            continue
        replay = replay_profile(
            LINK_PROFILES[profile_name],
            wire_bytes=256 * 1024,
            repeats=args.calibration_repeats,
        )
        results[profile_name] = {
            "calibration_method": replay.method,
            "calibration_passed": replay.calibrated,
            "calibration_bandwidth_error_fraction": replay.bandwidth_error_fraction,
            "calibration_bandwidth_error_q1_fraction": replay.bandwidth_error_q1_fraction,
            "calibration_bandwidth_error_q3_fraction": replay.bandwidth_error_q3_fraction,
            "calibration_observed_rtt_s": replay.observed_rtt_median_s,
            "calibration_rtt_error_fraction": replay.rtt_error_fraction,
            "calibration_rtt_error_q1_fraction": replay.rtt_error_q1_fraction,
            "calibration_rtt_error_q3_fraction": replay.rtt_error_q3_fraction,
            "calibration_rtt_within_tolerance": replay.rtt_within_tolerance,
            "calibration_median_s": replay.median_s,
            "calibration_replay_bytes": replay.replay_bytes,
            "calibration_repeats": replay.repeats,
        }
        print(
            f"[calibration] {profile_name}: bandwidth error="
            f"{replay.bandwidth_error_fraction:.1%}, RTT error="
            f"{replay.rtt_error_fraction:.1%} "
            f"(bandwidth {'PASS' if replay.calibrated else 'FAIL'})"
        )
        if not replay.calibrated:
            raise RuntimeError(
                f"Network profile {profile_name!r} missed the 10% bandwidth "
                "calibration tolerance."
            )
    return results


def _validation_selected_dims(
    records, datasets, message_protocol, feature_normalization
):
    selected = {}
    for dataset in datasets:
        candidates = {}
        seen = set()
        for record in records:
            if (
                record.get("status") == "ok"
                and record.get("dataset") == dataset
                and record.get("system") == "learned"
                and record.get("message_protocol") == message_protocol
                and record.get("feature_normalization") == feature_normalization
            ):
                key = (record["seed"], record["embedding_dim"])
                if key not in seen:
                    candidates.setdefault(int(record["embedding_dim"]), []).append(float(record["val_accuracy"]))
                    seen.add(key)
        if candidates:
            selected[dataset] = max(
                candidates,
                key=lambda dim: (sum(candidates[dim]) / len(candidates[dim]), -dim),
            )
    return selected


def _task_complete(args, existing_keys, dataset, seed, system, embedding_dim):
    required_suffixes = set()
    for strategy in args.partition_strategies:
        partition_seeds = args.partition_seeds if strategy == "random" else [0]
        for count in args.partition_counts:
            for partition_seed in partition_seeds:
                for profile in args.network_profiles:
                    required_suffixes.add((strategy, count, partition_seed, profile))
    available_suffixes = {
        key[6:]
        for key in existing_keys
        if key[:3] == (dataset, seed, system)
        and (embedding_dim is None or key[3] == embedding_dim)
        and key[4] == args.message_protocol
        and key[5] == args.feature_normalization
    }
    return required_suffixes.issubset(available_suffixes)


def _checkpoint_path(
    output_dir,
    dataset,
    seed,
    system,
    embedding_dim,
    epochs,
    message_protocol,
    feature_normalization,
):
    dim = "full" if embedding_dim is None else str(embedding_dim)
    protocol = message_protocol.replace("-", "_")
    normalization = feature_normalization.replace("-", "_")
    filename = (
        f"{dataset}_{system}_{protocol}_{normalization}_d{dim}_s{seed}_e{epochs}.pt"
    )
    return output_dir / "checkpoints" / filename


def _environment(message_protocol="transductive", feature_normalization="row-l2"):
    import platform
    import torch

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cpu_threads": os.cpu_count(),
        "framing": "single-workstation partition-aware emulation",
        "message_protocol": message_protocol,
        "feature_normalization": feature_normalization,
        "physical_edge_hardware": False,
        "energy_measured": False,
    }


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=SUPPORTED_DATASETS, default=DEFAULT_DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--embedding-dims", nargs="+", type=int, default=[8, 16, 32, 64])
    parser.add_argument("--pca-dims", nargs="+", type=int, default=[16, 32])
    parser.add_argument("--partition-counts", nargs="+", type=int, default=[2, 4, 8])
    parser.add_argument("--partition-strategies", nargs="+", choices=["random", "locality"], default=["random", "locality"])
    parser.add_argument("--partition-seeds", nargs="+", type=int, default=[42, 123, 2024])
    parser.add_argument("--network-profiles", nargs="+", choices=sorted(LINK_PROFILES), default=["constrained", "moderate", "lan"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--message-protocol",
        choices=["transductive", "source-only"],
        default="transductive",
        help="Use the full observed graph by default; source-only is a diagnostic ablation.",
    )
    parser.add_argument(
        "--feature-normalization",
        choices=["row-l2", "train-standardized-row-l2"],
        default="row-l2",
        help="Row-L2 is the primary sparse-feature protocol; train-standardized reproduces the frozen diagnostic.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--checkpoint-source-dir",
        help="Restore matching checkpoints from another result tree without copying them.",
    )
    parser.add_argument("--include-ablations", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--calibration-repeats", type=int, default=3)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", help=argparse.SUPPRESS)
    parser.add_argument("--worker-system", help=argparse.SUPPRESS)
    parser.add_argument("--worker-embedding-dim", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint", help=argparse.SUPPRESS)
    parser.add_argument("--source-checkpoint", help=argparse.SUPPRESS)
    return parser


if __name__ == "__main__":
    main()
