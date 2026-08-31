"""Correctness checks for the submission experiment artifact."""

from __future__ import annotations

import json

import pytest
import torch
from torch_geometric.data import Data


def _graph() -> Data:
    x = torch.randn(8, 6)
    y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    edges = torch.tensor(
        [[0, 0, 1, 2, 3, 4, 5, 6], [2, 3, 2, 4, 5, 6, 7, 7]],
        dtype=torch.long,
    )
    train = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.bool)
    val = torch.tensor([0, 0, 0, 0, 1, 1, 0, 0], dtype=torch.bool)
    test = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1], dtype=torch.bool)
    return Data(x=x, y=y, edge_index=edges, train_mask=train, val_mask=val, test_mask=test)


def test_official_validation_mask_is_preserved():
    from src.training.splits import resolve_train_val_masks

    graph = _graph()
    train, validation, source = resolve_train_val_masks(graph, seed=999)
    assert source == "official"
    assert torch.equal(train, graph.train_mask)
    assert torch.equal(validation, graph.val_mask)


def test_fallback_split_never_uses_test_nodes():
    from src.training.splits import resolve_train_val_masks

    graph = _graph()
    del graph.val_mask
    train, validation, source = resolve_train_val_masks(graph, seed=42)
    assert source == "stratified_fallback"
    assert not ((train | validation) & graph.test_mask).any()


def test_paper_message_protocol_defaults_to_full_graph():
    from src.experiment.paper_experiment import run_paper_model

    graph = _graph()
    transductive = run_paper_model(
        graph, dataset="toy", system="full", seed=42, epochs=1
    )
    source_only = run_paper_model(
        graph,
        dataset="toy",
        system="full",
        seed=42,
        epochs=1,
        message_protocol="source-only",
    )
    assert transductive["message_protocol"] == "transductive"
    assert transductive["feature_normalization"] == "row-l2"
    assert transductive["num_edges"] == graph.num_edges
    assert source_only["num_edges"] < graph.num_edges


def test_paper_record_reports_the_actual_local_mlp_configuration(tmp_path):
    import hashlib

    from src.config import MLP_HIDDEN_DIM
    from src.experiment.paper_experiment import run_paper_model

    checkpoint = tmp_path / "local.pt"
    record = run_paper_model(
        _graph(),
        dataset="toy",
        system="local",
        seed=42,
        epochs=1,
        checkpoint_path=checkpoint,
    )
    assert record["hidden_dim"] == MLP_HIDDEN_DIM
    assert record["model_config"]["hidden_dim"] == MLP_HIDDEN_DIM
    assert record["model_config"]["batch_normalization"] is True
    assert record["checkpoint_sha256"] == hashlib.sha256(
        checkpoint.read_bytes()
    ).hexdigest()


def test_centralized_int8_upload_reuses_full_checkpoint(tmp_path):
    from src.experiment.paper_experiment import run_paper_model

    checkpoint = tmp_path / "full.pt"
    baseline = run_paper_model(
        _graph(),
        dataset="toy",
        system="full",
        seed=42,
        epochs=1,
        checkpoint_path=checkpoint,
    )
    compressed = run_paper_model(
        _graph(),
        dataset="toy",
        system="full_int8",
        seed=42,
        epochs=1,
        source_checkpoint_path=checkpoint,
    )

    assert compressed["restored_checkpoint"]
    assert compressed["compression"] == "full_feature_int8_upload"
    assert compressed["bytes_per_value"] == 1
    assert compressed["source_checkpoint_val_accuracy"] == baseline["val_accuracy"]


def test_locality_partitions_are_deterministic_and_balanced():
    from src.simulation.partitioning import assign_locality_aware_partitions

    first = assign_locality_aware_partitions(_graph(), 3)
    second = assign_locality_aware_partitions(_graph(), 3)
    assert torch.equal(first.partition_id, second.partition_id)
    sizes = torch.bincount(first.partition_id)
    assert int(sizes.max() - sizes.min()) <= 1


def test_network_calibration_uses_observed_rtt(monkeypatch):
    from src.simulation import network_emulation

    profile = network_emulation.LinkProfile("test", bandwidth_bps=800.0, rtt_ms=100.0)
    monkeypatch.setattr(
        network_emulation,
        "_replay_once",
        lambda _profile, _byte_count: (0.305, 0.105),
    )
    replay = network_emulation.replay_profile(
        profile,
        wire_bytes=20,
        repeats=3,
        minimum_transfer_s=0.0,
    )

    assert replay.achieved_bandwidth_bps == pytest.approx(800.0)
    assert replay.observed_rtt_median_s == pytest.approx(0.105)
    assert replay.rtt_error_fraction == pytest.approx(0.05)
    assert replay.rtt_error_q1_fraction == pytest.approx(0.05)
    assert replay.rtt_error_q3_fraction == pytest.approx(0.05)
    assert replay.bandwidth_error_q1_fraction == pytest.approx(0.0)
    assert replay.bandwidth_error_q3_fraction == pytest.approx(0.0)
    assert replay.rtt_within_tolerance
    assert replay.calibrated


def test_batched_trace_deduplicates_edges_and_matches_serialization():
    from src.simulation.communication import communication_trace, serialize_transfer_batches

    graph = Data(
        x=torch.randn(4, 3),
        edge_index=torch.tensor([[0, 0, 1], [2, 3, 2]]),
    )
    graph.partition_id = torch.tensor([0, 0, 1, 1])
    graph.num_partitions = 2
    graph.num_boundary_edges = 3
    trace = communication_trace(graph, 3)
    payload = serialize_transfer_batches(graph.x, graph)
    assert trace.cross_partition_edges == 3
    assert trace.unique_transfers == 2
    assert trace.transfer_batches == 1
    assert len(payload) == trace.serialized_wire_bytes


def test_centralized_int8_upload_matches_wire_formula():
    from src.simulation.communication import (
        centralized_upload_size,
        deserialize_transfer_batches,
        serialize_centralized_upload,
    )

    features = torch.randn(8, 6)
    payload = serialize_centralized_upload(features, quantized=True)
    raw_bytes, wire_bytes = centralized_upload_size(8, 6, bytes_per_value=1)
    node_ids, decoded = deserialize_transfer_batches(payload, 6, quantized=True)[(0, 0)]

    assert raw_bytes == 48
    assert len(payload) == wire_bytes
    assert torch.equal(node_ids, torch.arange(8))
    assert torch.allclose(decoded, features, atol=float(features.abs().max() / 127) + 1e-6)


def test_transfer_round_trip_preserves_float_logits():
    from src.simulation.communication import deserialize_transfer_batches, serialize_transfer_batches
    from src.systems.distributed_system import FullFeatureGAT

    graph = Data(
        x=torch.randn(4, 3),
        edge_index=torch.tensor([[0, 1, 2, 3], [2, 2, 3, 0]]),
    )
    graph.partition_id = torch.tensor([0, 0, 1, 1])
    graph.num_partitions = 2
    graph.num_boundary_edges = 3
    payload = serialize_transfer_batches(graph.x, graph)
    decoded = deserialize_transfer_batches(payload, 3)
    model = FullFeatureGAT(3, 4, 2, dropout=0.0)
    model.eval()
    with torch.no_grad():
        expected = model(graph.x, graph.edge_index)
        for destination_partition in range(graph.num_partitions):
            local = graph.partition_id == destination_partition
            available = torch.zeros_like(graph.x)
            available[local] = graph.x[local]
            for (_, destination), (node_ids, values) in decoded.items():
                if destination == destination_partition:
                    available[node_ids] = values
            incoming = local[graph.edge_index[1]]
            actual = model(available, graph.edge_index[:, incoming])
            assert torch.allclose(actual[local], expected[local], atol=1e-7)


def test_int8_wire_round_trip_is_bounded():
    from src.simulation.communication import deserialize_transfer_batches, serialize_transfer_batches

    graph = Data(x=torch.randn(4, 5), edge_index=torch.tensor([[0, 1], [2, 3]]))
    graph.partition_id = torch.tensor([0, 0, 1, 1])
    graph.num_partitions = 2
    graph.num_boundary_edges = 2
    decoded = deserialize_transfer_batches(
        serialize_transfer_batches(graph.x, graph, quantized=True),
        5,
        quantized=True,
    )
    ids, values = next(iter(decoded.values()))
    assert torch.allclose(values, graph.x[ids], atol=float(graph.x.abs().max() / 127) + 1e-6)


def test_partitioned_int8_logits_use_wire_batch_scales():
    from src.experiment.paper_experiment import partitioned_int8_logits
    from src.simulation.communication import (
        deserialize_transfer_batches,
        serialize_transfer_batches,
    )
    from src.systems.distributed_system import DistributedTinyGNN

    graph = _graph()
    graph.partition_id = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    graph.num_partitions = 2
    model = DistributedTinyGNN(6, 3, 4, 2, dropout=0.0)
    model.eval()

    with torch.no_grad():
        embeddings = model.compressor(graph.x)
        decoded = deserialize_transfer_batches(
            serialize_transfer_batches(embeddings, graph, quantized=True),
            3,
            quantized=True,
        )
        expected = torch.empty((graph.num_nodes, 2))
        for destination in range(graph.num_partitions):
            local = graph.partition_id == destination
            available = torch.zeros_like(embeddings)
            available[local] = embeddings[local]
            for (_, batch_destination), (node_ids, values) in decoded.items():
                if batch_destination == destination:
                    available[node_ids] = values
            incoming = local[graph.edge_index[1]]
            partial = model.decode_embeddings(available, graph.edge_index[:, incoming])
            expected[local] = partial[local]

    assert torch.allclose(partitioned_int8_logits(model, graph), expected, atol=1e-7)


def test_append_only_records_summarize_and_gate(tmp_path):
    from src.utils.paper_results import append_record, read_jsonl, scientific_gate, summarize_jsonl

    raw = tmp_path / "raw.jsonl"
    for dataset in ["cora", "citeseer", "pubmed"]:
        for seed in [42, 123]:
            common = {
                "status": "ok",
                "dataset": dataset,
                "seed": seed,
                "compression": "none",
                "partition_strategy": "random",
                "num_partitions": 4,
                "partition_seed": 42,
                "network_profile": "constrained",
                "message_protocol": "transductive",
                "feature_normalization": "row-l2",
                "calibration_passed": True,
                "test_macro_f1": 0.8,
                "boundary_edge_ratio": 0.5,
                "inference_median_s": 0.01,
                "peak_rss_mb": 20.0,
                "modeled_transfer_s": 0.1,
                "composite_latency_s": 0.11,
            }
            append_record(raw, {**common, "system": "full", "embedding_dim": 64, "val_accuracy": 0.8, "test_accuracy": 0.8, "serialized_wire_bytes": 4000})
            append_record(raw, {**common, "system": "learned", "compression": "learned_linear", "embedding_dim": 16, "val_accuracy": 0.79, "test_accuracy": 0.77, "serialized_wire_bytes": 900})
    summary = summarize_jsonl(raw, tmp_path / "summary.csv")
    assert not summary.empty
    gate = scientific_gate(
        read_jsonl(raw),
        required_seeds=2,
        expected_embedding_dims=(16,),
    )
    assert gate["passed"]
    assert gate["datasets"]["cora"][
        "single_round_communication_reduction_factor"
    ] == pytest.approx(4000 / 900)


def test_gate_rejects_an_incomplete_candidate_matrix(tmp_path):
    from src.utils.paper_results import append_record, read_jsonl, scientific_gate

    raw = tmp_path / "raw.jsonl"
    common = {
        "status": "ok",
        "dataset": "cora",
        "seed": 42,
        "partition_strategy": "random",
        "num_partitions": 4,
        "partition_seed": 42,
        "network_profile": "constrained",
        "message_protocol": "transductive",
        "feature_normalization": "row-l2",
        "calibration_passed": True,
        "val_accuracy": 0.8,
        "test_accuracy": 0.8,
        "serialized_wire_bytes": 4000,
    }
    append_record(raw, {**common, "system": "full", "embedding_dim": 64})
    append_record(
        raw,
        {
            **common,
            "system": "learned",
            "embedding_dim": 16,
            "serialized_wire_bytes": 900,
        },
    )
    gate = scientific_gate(
        read_jsonl(raw),
        required_seeds=1,
        expected_embedding_dims=(8, 16),
    )
    assert not gate["passed"]
    assert gate["datasets"]["cora"]["missing_embedding_dims"] == [8]


def test_canonical_figures_average_all_random_placement_seeds():
    import pandas as pd
    from scripts.generate_paper_artifacts import _canonical

    frame = pd.DataFrame(
        [
            {
                "partition_strategy": "random",
                "num_partitions": 4,
                "partition_seed": seed,
                "network_profile": "constrained",
            }
            for seed in (42, 123, 2024)
        ]
    )
    assert set(_canonical(frame)["partition_seed"]) == {42, 123, 2024}


def test_partition_table_uses_unique_placement_realizations(tmp_path):
    import pandas as pd
    from scripts.generate_paper_artifacts import _partition_table

    rows = []
    for partition_count in (2, 4, 8):
        for strategy, seeds in (("random", (42, 123, 2024)), ("locality", (0,))):
            for partition_seed in seeds:
                for training_seed in (1, 2):
                    rows.append(
                        {
                            "dataset": "toy",
                            "system": "learned",
                            "embedding_dim": 64,
                            "val_accuracy": 0.8,
                            "partition_strategy": strategy,
                            "num_partitions": partition_count,
                            "partition_seed": partition_seed,
                            "seed": training_seed,
                            "boundary_edge_ratio": 0.6 if strategy == "random" else 0.3,
                            "unique_transfers": 150 if strategy == "random" else 100,
                            "serialized_wire_bytes": 1_500_000 if strategy == "random" else 1_000_000,
                        }
                    )

    path = tmp_path / "partition_results.tex"
    _partition_table(pd.DataFrame(rows), path)
    output = path.read_text(encoding="utf-8")

    assert output.count("0.600 & 150 & 1.50 & 0.300 & 100 & 1.00") == 3


def test_wire_plot_keeps_zero_byte_local_reference_off_log_axis(monkeypatch, tmp_path):
    import pandas as pd
    from scripts import generate_paper_artifacts as artifacts

    frame = pd.DataFrame(
        [
            {
                "dataset": "toy",
                "partition_strategy": "random",
                "num_partitions": 4,
                "network_profile": "constrained",
                "system": system,
                "embedding_dim": dimension,
                "test_accuracy": accuracy,
                "serialized_wire_bytes": byte_count,
            }
            for system, dimension, accuracy, byte_count in (
                ("local", 0, 0.4, 0),
                ("learned", 8, 0.7, 1024),
            )
        ]
    )
    captured = {}

    def capture(fig, stem):
        captured["fig"] = fig
        return stem.with_suffix(".pdf")

    monkeypatch.setattr(artifacts, "_save", capture)
    artifacts._pareto(frame, tmp_path)
    ax = captured["fig"].axes[0]

    assert ax.get_xscale() == "log"
    assert any(line.get_label() == "Local: 0 B" for line in ax.lines)
    assert any(text.get_text() == "Local: 0 B" for text in ax.texts)
    assert all(
        min(line.get_xdata()) > 0
        for line in ax.lines
        if line.get_label() != "Local: 0 B"
    )


def test_record_reader_reports_corrupt_line(tmp_path):
    from src.utils.paper_results import read_jsonl

    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"ok": True}) + "\nnot json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        read_jsonl(path)
