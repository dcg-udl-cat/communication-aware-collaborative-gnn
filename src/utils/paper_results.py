"""Append-only records and deterministic summaries for the paper artifact."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


KEY_FIELDS = (
    "dataset",
    "seed",
    "system",
    "embedding_dim",
    "message_protocol",
    "feature_normalization",
    "partition_strategy",
    "num_partitions",
    "partition_seed",
    "network_profile",
)


def record_key(record: dict[str, Any]) -> tuple:
    return tuple(record.get(field) for field in KEY_FIELDS)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSONL at line {line_number}: {error}") from error
    return records


def append_record(path: str | Path, record: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        json.dump(record, handle, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()


def summarize_jsonl(raw_path: str | Path, summary_path: str | Path) -> pd.DataFrame:
    records = [r for r in read_jsonl(raw_path) if r.get("status") == "ok"]
    if not records:
        dataframe = pd.DataFrame()
    else:
        frame = pd.DataFrame(records)
        if "message_protocol" not in frame:
            frame["message_protocol"] = "source-only"
        else:
            frame["message_protocol"] = frame["message_protocol"].fillna("source-only")
        if "feature_normalization" not in frame:
            frame["feature_normalization"] = "train-standardized-row-l2"
        else:
            frame["feature_normalization"] = frame["feature_normalization"].fillna(
                "train-standardized-row-l2"
            )
        group_columns = [
            "dataset",
            "system",
            "compression",
            "embedding_dim",
            "message_protocol",
            "feature_normalization",
            "partition_strategy",
            "num_partitions",
            "network_profile",
        ]
        numeric = [
            "test_accuracy",
            "test_macro_f1",
            "val_accuracy",
            "serialized_wire_bytes",
            "boundary_edge_ratio",
            "inference_median_s",
            "peak_rss_mb",
            "modeled_transfer_s",
            "composite_latency_s",
        ]
        rows = []
        for keys, group in frame.groupby(group_columns, dropna=False, sort=True):
            row = dict(zip(group_columns, keys))
            row["runs"] = int(group["seed"].nunique())
            row["partition_runs"] = int(
                group[["seed", "partition_seed"]].drop_duplicates().shape[0]
            )
            for column in numeric:
                # Placement seeds are an ablation nested inside each training seed.
                # Average them first; statistical uncertainty is across model seeds.
                values = [
                    float(v)
                    for v in group.groupby("seed", sort=True)[column]
                    .mean()
                    .dropna()
                    .tolist()
                ]
                if values:
                    row[f"{column}_mean"] = statistics.mean(values)
                    row[f"{column}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
                    row[f"{column}_ci95"] = _student_t_half_width(values)
            rows.append(row)
        dataframe = pd.DataFrame(rows).sort_values(group_columns).reset_index(drop=True)

    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(summary_path, index=False)
    return dataframe


def scientific_gate(
    records: Iterable[dict[str, Any]],
    *,
    required_seeds: int = 5,
    expected_embedding_dims: tuple[int, ...] = (8, 16, 32, 64),
    feature_normalization: str = "row-l2",
) -> dict[str, Any]:
    """Evaluate the pre-registered accuracy and single-round communication gate."""

    frame = pd.DataFrame([r for r in records if r.get("status") == "ok"])
    if frame.empty:
        return {
            "passed": False,
            "calibration_passed": False,
            "required_seeds": required_seeds,
            "expected_embedding_dims": list(expected_embedding_dims),
            "feature_normalization": feature_normalization,
            "passed_datasets": [],
            "datasets": {},
        }
    if "message_protocol" not in frame:
        frame["message_protocol"] = "source-only"
    else:
        frame["message_protocol"] = frame["message_protocol"].fillna("source-only")
    if "feature_normalization" not in frame:
        frame["feature_normalization"] = "train-standardized-row-l2"
    else:
        frame["feature_normalization"] = frame["feature_normalization"].fillna(
            "train-standardized-row-l2"
        )
    subset = frame[
        (frame["partition_strategy"] == "random")
        & (frame["num_partitions"] == 4)
        & (frame["network_profile"] == "constrained")
        & (frame["message_protocol"] == "transductive")
        & (frame["feature_normalization"] == feature_normalization)
    ]
    calibration_passed = (
        not subset.empty
        and "calibration_passed" in subset
        and bool(subset["calibration_passed"].fillna(False).all())
    )
    outcomes = {}
    for dataset, group in subset.groupby("dataset"):
        full = (
            group[group["system"] == "full"]
            .groupby("seed")[["test_accuracy", "serialized_wire_bytes"]]
            .mean()
        )
        learned = group[
            (group["system"] == "learned")
            & (group["embedding_dim"].isin(expected_embedding_dims))
        ]
        if full.empty or learned.empty:
            continue
        dim_seed_counts = {
            int(dim): int(dim_group["seed"].nunique())
            for dim, dim_group in learned.groupby("embedding_dim")
        }
        missing_dims = sorted(set(expected_embedding_dims) - set(dim_seed_counts))
        incomplete_dims = {
            dim: dim_seed_counts.get(dim, 0)
            for dim in expected_embedding_dims
            if dim_seed_counts.get(dim, 0) < required_seeds
        }
        validation = learned.groupby("embedding_dim")["val_accuracy"].mean()
        selected_dim = int(validation.idxmax())
        selected = (
            learned[learned["embedding_dim"] == selected_dim]
            .groupby("seed")[["test_accuracy", "serialized_wire_bytes"]]
            .mean()
        )
        common = full.index.intersection(selected.index)
        if common.empty:
            continue
        accuracy_ratio = float(selected.loc[common, "test_accuracy"].mean() / full.loc[common, "test_accuracy"].mean())
        communication_reduction = float(
            full.loc[common, "serialized_wire_bytes"].mean()
            / selected.loc[common, "serialized_wire_bytes"].mean()
        )
        complete = (
            len(common) >= required_seeds
            and len(full) >= required_seeds
            and not incomplete_dims
        )
        outcomes[dataset] = {
            "selected_embedding_dim": selected_dim,
            "successful_seeds": int(len(common)),
            "full_feature_seeds": int(len(full)),
            "candidate_seed_counts": dim_seed_counts,
            "missing_embedding_dims": missing_dims,
            "complete": complete,
            "accuracy_ratio": accuracy_ratio,
            "single_round_communication_reduction_factor": communication_reduction,
            "passed": (
                complete
                and accuracy_ratio >= 0.95
                and communication_reduction >= 4.0
            ),
        }
    passed_datasets = sorted(name for name, outcome in outcomes.items() if outcome["passed"])
    return {
        "passed": calibration_passed and len(passed_datasets) >= 3,
        "calibration_passed": calibration_passed,
        "required_seeds": required_seeds,
        "expected_embedding_dims": list(expected_embedding_dims),
        "feature_normalization": feature_normalization,
        "passed_datasets": passed_datasets,
        "datasets": outcomes,
    }


def _student_t_half_width(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    from scipy.stats import t

    std = statistics.stdev(values)
    return float(t.ppf(0.975, len(values) - 1)) * std / math.sqrt(len(values))
