#!/usr/bin/env python3
"""Post-review, exploratory paired analyses of the frozen submission evidence.

Run with .venv/bin/python scripts/analyze_revision_statistics.py. No models are
trained, no frozen evidence is modified, and placements are never replicates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import permutation_test, t, ttest_rel

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 123, 2024, 2025, 2026)
DATASETS = ("cora", "citeseer", "pubmed", "flickr")


def paired_summary(left, right):
    differences = [a - b for a, b in zip(left, right, strict=True)]
    n = len(differences)
    mean = statistics.mean(differences)
    sd = statistics.stdev(differences)
    halfwidth = float(t.ppf(0.975, n - 1)) * sd / math.sqrt(n)
    # The null is invariant to independently reversing paired difference signs.
    # Exhaustive enumeration is appropriate here (2**5 = 32 sign patterns).
    observed = abs(sum(differences))
    extreme = sum(
        abs(sum(sign * value for sign, value in zip(signs, differences)))
        >= observed - 1e-12
        for signs in itertools.product((-1, 1), repeat=n)
    )
    p = extreme / 2**n
    reference = permutation_test(
        (np.array(differences),), np.mean, permutation_type="samples",
        n_resamples=np.inf, alternative="two-sided",
    )
    assert math.isclose(p, float(reference.pvalue), abs_tol=1e-12)
    ci = ttest_rel(left, right).confidence_interval()
    assert math.isclose(mean - halfwidth, float(ci.low), abs_tol=1e-12)
    assert math.isclose(mean + halfwidth, float(ci.high), abs_tol=1e-12)
    return {
        "n_training_seed_pairs": n,
        "left_mean_accuracy": statistics.mean(left),
        "right_mean_accuracy": statistics.mean(right),
        "mean_accuracy_difference": mean,
        "difference_sample_sd": sd,
        "paired_t_ci95_lower": mean - halfwidth,
        "paired_t_ci95_upper": mean + halfwidth,
        "positive_pairs": sum(value > 0 for value in differences),
        "zero_pairs": sum(value == 0 for value in differences),
        "negative_pairs": sum(value < 0 for value in differences),
        "exact_two_sided_sign_flip_p": p,
    }


def holm(pvalues):
    adjusted = [0.0] * len(pvalues)
    running = 0.0
    for rank, index in enumerate(sorted(range(len(pvalues)), key=pvalues.__getitem__)):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[index]))
        adjusted[index] = running
    return adjusted


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "artifacts/paper/results/release-v1.0.0-final/raw/results.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/paper/revision/analysis")
    args = parser.parse_args()
    raw_bytes = args.raw.read_bytes()
    records = [json.loads(line) for line in raw_bytes.splitlines() if line.strip()]
    assert len(records) == 6480 and all(row["status"] == "ok" for row in records)
    assert {row["message_protocol"] for row in records} == {"transductive"}
    assert {row["feature_normalization"] for row in records} == {"row-l2"}
    canonical = [row for row in records if row["partition_strategy"] == "random"
                 and row["num_partitions"] == 4 and row["network_profile"] == "constrained"]
    grouped = defaultdict(list)
    for row in canonical:
        grouped[row["dataset"], row["system"], row["embedding_dim"], row["seed"]].append(row)
    models = {}
    for key, rows in grouped.items():
        assert {row["partition_seed"] for row in rows} == {42, 123, 2024}
        assert len(rows) == 3
        if key[1] == "learned_int8":
            continue  # Int8 accuracy is legitimately placement-dependent.
        for field in ("test_accuracy", "val_accuracy", "train_accuracy", "loss_curve", "checkpoint_sha256"):
            assert all(row[field] == rows[0][field] for row in rows)
        models[key] = rows[0]  # One trained model, repeated in three placement rows.
    selected = {
        dataset: min((8, 16, 32, 64), key=lambda k: (
            -statistics.mean(models[dataset, "learned", k, seed]["val_accuracy"] for seed in SEEDS), k))
        for dataset in DATASETS
    }
    frozen_gate = json.loads((args.raw.parent.parent / "scientific_gate.json").read_text())
    assert all(k == frozen_gate["datasets"][dataset]["selected_embedding_dim"] for dataset, k in selected.items())

    comparisons, pairs = [], []
    specs = [(dataset, "learned_vs_full", selected[dataset], "full") for dataset in DATASETS]
    specs += [(dataset, "learned_vs_pca", k, "pca") for dataset in ("cora", "flickr") for k in (16, 32)]
    for dataset, name, k, comparator in specs:
        right_k = models[dataset, "learned", k, SEEDS[0]]["input_dim"] if comparator == "full" else k
        left = [models[dataset, "learned", k, seed]["test_accuracy"] for seed in SEEDS]
        right = [models[dataset, comparator, right_k, seed]["test_accuracy"] for seed in SEEDS]
        comparisons.append({"dataset": dataset, "comparison": name, "embedding_dim": k, **paired_summary(left, right)})
        for seed, a, b in zip(SEEDS, left, right, strict=True):
            pairs.append({"dataset": dataset, "comparison": name, "embedding_dim": k,
                          "seed": seed, "learned_accuracy": a, "comparator_accuracy": b,
                          "accuracy_difference": a - b})
    for row, adjusted in zip(comparisons, holm([r["exact_two_sided_sign_flip_p"] for r in comparisons]), strict=True):
        row["holm_p_eight_exploratory_comparisons"] = adjusted

    diagnostics = []
    import torch
    for (dataset, system, k, seed), row in sorted(models.items()):
        if dataset not in ("cora", "flickr") or system not in ("learned", "pca") or k not in (16, 32):
            continue
        curve = row["loss_curve"]
        checkpoint = ROOT / "artifacts/paper/results/confirmatory-row-l2/checkpoints" / f"{dataset}_{system}_transductive_row_l2_d{k}_s{seed}_e100.pt"
        checkpoint_matches = checkpoint.is_file() and hashlib.sha256(checkpoint.read_bytes()).hexdigest() == row["checkpoint_sha256"]
        assert checkpoint_matches, f"Frozen checkpoint is absent or differs: {checkpoint}"
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        assert saved["loss_curve"] == curve
        assert saved["metrics"]["test_accuracy"] == row["test_accuracy"]
        diagnostics.append({
            "dataset": dataset, "system": system, "embedding_dim": k, "seed": seed,
            "train_nodes": row["train_nodes"], "num_parameters": row["num_parameters"],
            "train_accuracy": row["train_accuracy"], "val_accuracy": row["val_accuracy"],
            "test_accuracy": row["test_accuracy"],
            "train_test_accuracy_gap": row["train_accuracy"] - row["test_accuracy"],
            "epochs_run": row["epochs_run"], "epochs_requested": row["epochs_requested"],
            "reached_epoch_cap": row["epochs_run"] == row["epochs_requested"],
            "first_training_loss": curve[0], "last_training_loss": curve[-1],
            "relative_training_loss_drop": row["relative_loss_drop"],
            "loss_mean_epochs_81_90": statistics.mean(curve[80:90]),
            "loss_mean_epochs_91_100": statistics.mean(curve[90:100]),
            "checkpoint_sha256": row["checkpoint_sha256"],
            "checkpoint_path": str(checkpoint.relative_to(ROOT)),
            "checkpoint_hash_and_history_verified": checkpoint_matches,
        })
    diagnostic_groups = defaultdict(list)
    for row in diagnostics:
        diagnostic_groups[row["dataset"], row["system"], row["embedding_dim"]].append(row)
    summaries = []
    for (dataset, system, k), rows in sorted(diagnostic_groups.items()):
        summary = {"dataset": dataset, "system": system, "embedding_dim": k, "n": len(rows),
                   "train_nodes": rows[0]["train_nodes"], "num_parameters": rows[0]["num_parameters"],
                   "runs_reaching_epoch_cap": sum(row["reached_epoch_cap"] for row in rows)}
        for field in ("train_accuracy", "val_accuracy", "test_accuracy", "train_test_accuracy_gap", "last_training_loss", "loss_mean_epochs_81_90", "loss_mean_epochs_91_100"):
            summary[field + "_mean"] = statistics.mean(row[field] for row in rows)
            summary[field + "_sd"] = statistics.stdev(row[field] for row in rows)
        summaries.append(summary)

    matched_files = []
    for relative in ("tables/main_results.tex", "tables/partition_results.tex", "results_macros.tex"):
        submitted = ROOT / "artifacts/paper/submission/Manuscript_Source/generated" / relative
        generated = ROOT / "artifacts/paper/generated/release-v1.0.0-final" / relative
        assert submitted.read_bytes() == generated.read_bytes()
        matched_files.append(relative)
    provenance = {
        "analysis_status": "post-review exploratory reanalysis; not preregistered",
        "source": str(args.raw.resolve()), "source_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "source_records": len(records), "canonical_placement_rows": len(canonical),
        "submission_files_byte_matched_to_final_generated_artifact": matched_files,
        "seed_pairs_per_comparison": 5, "seeds": SEEDS, "selected_dimensions": selected,
        "unit_of_replication": "training seed on a fixed official dataset split",
        "placement_handling": "three repeated placement rows verified identical for analyzed float32 model metrics and deduplicated",
        "test": "two-sided mean paired-difference exact sign-flip test; all 32 patterns; null requires independent symmetric paired differences",
        "multiplicity": "Holm adjustment across all eight reported exploratory comparisons",
        "confidence_intervals": "unadjusted paired Student-t, df=4; model-based, not distribution-free or simultaneous",
        "minimum_attainable_two_sided_p_at_five_nonzero_pairs": 2 / 2**5,
        "pca_checkpoint_hashes_and_histories_verified": len(diagnostics),
        "validation_loss_histories_or_best_epoch_saved": False,
        "reference_docs": ["https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html", "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html"],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "statistics_paired_summary.csv", comparisons)
    write_csv(args.output_dir / "statistics_seed_pairs.csv", pairs)
    write_csv(args.output_dir / "pca_seed_diagnostics.csv", diagnostics)
    write_csv(args.output_dir / "pca_diagnostic_summary.csv", summaries)
    (args.output_dir / "statistics_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    assert hashlib.sha256(args.raw.read_bytes()).hexdigest() == provenance["source_sha256"]
    assert paired_summary([1, 2, 3, 4, 5], [0, 0, 0, 0, 0])["exact_two_sided_sign_flip_p"] == 0.0625
    assert holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]
    for row in comparisons:
        print(f"{row['dataset']:8s} {row['comparison']:16s} k={row['embedding_dim']:2d}: "
              f"delta={row['mean_accuracy_difference']:+.6f}, "
              f"CI=[{row['paired_t_ci95_lower']:+.6f},{row['paired_t_ci95_upper']:+.6f}], "
              f"p={row['exact_two_sided_sign_flip_p']:.4f}, "
              f"Holm={row['holm_p_eight_exploratory_comparisons']:.4f}")
    print(f"Verified {len(diagnostics)} checkpoint hashes/histories; wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
