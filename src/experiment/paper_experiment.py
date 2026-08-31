"""One isolated training/evaluation unit for the Future Internet artifact."""

from __future__ import annotations

import hashlib
import platform
import resource
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch_geometric.data import Data

from config import (
    CONVERGENCE_MIN_LOSS_DROP,
    EARLY_STOPPING_PATIENCE,
    MLP_HIDDEN_DIM,
)
from models.baselines import SimpleMLPBaseline
from systems.distributed_system import (
    DistributedTinyGNN,
    FixedPCACompressor,
    FullFeatureGAT,
)
from simulation.communication import (
    deserialize_transfer_batches,
    serialize_transfer_batches,
    symmetric_int8_round_trip,
)
from training.evaluator import evaluate_classification, measure_inference_stats
from training.splits import make_inductive_view, resolve_train_val_masks
from training.trainer import make_optimizer, train_with_logging
from utils.feature_normalization import normalize_graph_features
from utils.metrics import count_trainable_parameters
from utils.seed import set_seed


def run_paper_model(
    graph: Data,
    *,
    dataset: str,
    system: str,
    seed: int,
    epochs: int,
    embedding_dim: int | None = None,
    hidden_dim: int = 64,
    dropout: float = 0.1,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    checkpoint_path: str | Path | None = None,
    source_checkpoint_path: str | Path | None = None,
    message_protocol: str = "transductive",
    feature_normalization: str = "row-l2",
) -> dict[str, Any]:
    """Train or restore one model and return a JSON-serializable record."""

    if system not in {"local", "full", "full_int8", "learned", "pca", "learned_int8"}:
        raise ValueError(f"Unsupported paper system: {system}")
    if system in {"learned", "pca", "learned_int8"} and not embedding_dim:
        raise ValueError(f"{system} requires a positive embedding_dim.")
    if message_protocol not in {"transductive", "source-only"}:
        raise ValueError(f"Unsupported message protocol: {message_protocol}")
    if feature_normalization not in {"row-l2", "train-standardized-row-l2"}:
        raise ValueError(f"Unsupported feature normalization: {feature_normalization}")

    set_seed(seed)
    device = torch.device("cpu")
    graph = graph.to(device)
    train_mask, val_mask, split_source = resolve_train_val_masks(graph, seed=seed)
    graph = normalize_graph_features(
        graph,
        train_mask=train_mask,
        standardize_columns=feature_normalization == "train-standardized-row-l2",
    ).to(device)
    if system != "local" and message_protocol == "source-only":
        graph = make_inductive_view(graph, train_mask).to(device)
    graph.train_mask = train_mask
    if system == "full_int8":
        graph.x = symmetric_int8_round_trip(graph.x)

    input_dim = int(graph.num_node_features)
    num_classes = int(graph.y.max().item()) + 1
    model = _make_model(
        system,
        graph,
        train_mask,
        input_dim=input_dim,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dropout=dropout,
        seed=seed,
    ).to(device)

    checkpoint = Path(checkpoint_path) if checkpoint_path else None
    source_checkpoint = Path(source_checkpoint_path) if source_checkpoint_path else None
    loss_curve: list[float]
    restored = False
    source_metrics: dict[str, Any] = {}

    load_from = (
        source_checkpoint
        if source_checkpoint is not None and source_checkpoint.exists()
        else checkpoint
    )
    if load_from and load_from.exists():
        saved = torch.load(load_from, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model_state"])
        loss_curve = list(saved.get("loss_curve", []))
        source_metrics = dict(saved.get("metrics", {}))
        restored = True
    elif system in {"learned_int8", "full_int8"}:
        raise FileNotFoundError(f"{system} requires its float32 source checkpoint.")
    else:
        optimizer = make_optimizer(model, lr=lr, weight_decay=weight_decay)
        loss_curve = train_with_logging(
            model,
            graph,
            optimizer,
            nn.CrossEntropyLoss(),
            device,
            epochs=epochs,
            label=f"paper/{dataset}/{system}/{seed}",
            val_mask=val_mask,
        )

    train_metrics = evaluate_classification(model, graph, train_mask)
    val_metrics = evaluate_classification(model, graph, val_mask)
    test_metrics = evaluate_classification(model, graph, graph.test_mask)

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    timing = measure_inference_stats(model, graph, warmup_runs=20, measure_runs=50)
    torch.set_num_threads(previous_threads)

    relative_drop = _relative_loss_drop(loss_curve)
    provenance = _git_provenance()
    record = {
        "record_type": "model",
        "status": "ok",
        "created_at": datetime.now(timezone.utc).isoformat(),
        **provenance,
        "dataset": dataset,
        "dataset_split": split_source,
        "message_protocol": message_protocol,
        "feature_normalization": feature_normalization,
        "seed": seed,
        "system": system,
        "compression": {
            "local": "none",
            "full": "none",
            "full_int8": "full_feature_int8_upload",
            "learned": "learned_linear",
            "pca": "train_fitted_pca",
            "learned_int8": "learned_linear_int8",
        }[system],
        "embedding_dim": input_dim if system in {"local", "full", "full_int8"} else embedding_dim,
        "bytes_per_value": 1 if system in {"full_int8", "learned_int8"} else 4,
        "input_dim": input_dim,
        "hidden_dim": MLP_HIDDEN_DIM if system == "local" else hidden_dim,
        "num_classes": num_classes,
        "num_nodes": int(graph.num_nodes),
        "num_edges": int(graph.num_edges),
        "train_nodes": int(train_mask.sum().item()),
        "val_nodes": int(val_mask.sum().item()),
        "test_nodes": int(graph.test_mask.sum().item()),
        "epochs_requested": epochs,
        "epochs_run": len(loss_curve),
        "restored_checkpoint": restored,
        "converged": relative_drop >= CONVERGENCE_MIN_LOSS_DROP,
        "relative_loss_drop": relative_drop,
        "loss_curve": loss_curve,
        "train_accuracy": train_metrics.accuracy,
        "train_macro_f1": train_metrics.macro_f1,
        "val_accuracy": val_metrics.accuracy,
        "val_macro_f1": val_metrics.macro_f1,
        "test_accuracy": test_metrics.accuracy,
        "test_macro_f1": test_metrics.macro_f1,
        "num_parameters": count_trainable_parameters(model),
        "inference_median_s": timing.median_s,
        "inference_q1_s": timing.q1_s,
        "inference_q3_s": timing.q3_s,
        "peak_rss_mb": _peak_rss_mb(),
        "rss_measurement": "isolated_worker_high_water",
        "inference_measurement": (
            "float_checkpoint_reference_host; transport evaluated after partitioning"
            if system == "learned_int8"
            else "dequantized_features_reference_host; upload quantization not timed"
            if system == "full_int8"
            else "end_to_end_model_reference_host"
        ),
        "model_config": {
            "architecture": {
                "local": "two_layer_mlp",
                "full": "full_feature_gat",
                "full_int8": "full_feature_gat",
                "learned": "linear_compressor_gat",
                "pca": "train_fitted_pca_gat",
                "learned_int8": "linear_compressor_gat",
            }[system],
            "hidden_dim": MLP_HIDDEN_DIM if system == "local" else hidden_dim,
            "embedding_dim": input_dim if system in {"local", "full", "full_int8"} else embedding_dim,
            "dropout": dropout,
            "gat_heads": 4 if system != "local" else None,
            "batch_normalization": system == "local",
            "pca": (
                {
                    "fit_split": "training_nodes",
                    "solver": "randomized",
                    "random_state": seed,
                }
                if system == "pca"
                else None
            ),
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "optimizer": "Adam",
            "loss": "cross_entropy",
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "early_stopping_monitor": "validation_cross_entropy",
            "early_stopping_min_delta": 1e-4,
            "timing_warmup_runs": 20,
            "timing_measure_runs": 50,
            "timing_threads": 1,
        },
        "checkpoint_sha256": _sha256_file(load_from) if restored and load_from else None,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "platform": platform.platform(),
    }
    if system in {"learned_int8", "full_int8"}:
        record["source_checkpoint_val_accuracy"] = source_metrics.get("val_accuracy")

    if checkpoint and system not in {"learned_int8", "full_int8"} and not restored:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": model.state_dict(),
                "loss_curve": loss_curve,
                "metrics": {
                    "val_accuracy": val_metrics.accuracy,
                    "test_accuracy": test_metrics.accuracy,
                },
            },
            checkpoint,
        )
        record["checkpoint_sha256"] = _sha256_file(checkpoint)
    return record


def partitioned_int8_logits(model: DistributedTinyGNN, graph: Data) -> torch.Tensor:
    """Evaluate int8 transport exactly as source/destination wire batches.

    Local embeddings remain float32. Each remote source-partition to
    destination-partition batch is independently quantized, serialized, and
    dequantized before the destination performs its incoming-edge aggregation.
    """

    if getattr(graph, "partition_id", None) is None:
        raise ValueError("graph must have partition_id metadata.")
    if not hasattr(model, "decode_embeddings") or not hasattr(model, "compressor"):
        raise TypeError("partitioned int8 evaluation requires a compressed GNN decoder.")

    model.eval()
    with torch.no_grad():
        embeddings = model.compressor(graph.x)
        payload = serialize_transfer_batches(embeddings, graph, quantized=True)
        decoded = deserialize_transfer_batches(
            payload,
            model.embedding_dim,
            quantized=True,
        )
        logits = None
        partition_ids = graph.partition_id
        for destination in range(int(graph.num_partitions)):
            local = partition_ids == destination
            available = torch.zeros_like(embeddings)
            available[local] = embeddings[local]
            for (_, batch_destination), (node_ids, values) in decoded.items():
                if batch_destination == destination:
                    available[node_ids.to(available.device)] = values.to(available.device)

            incoming = local[graph.edge_index[1]]
            edge_weight = getattr(graph, "edge_weight", None)
            if edge_weight is not None:
                edge_weight = edge_weight[incoming]
            partial = model.decode_embeddings(
                available,
                graph.edge_index[:, incoming],
                edge_weight,
            )
            if logits is None:
                logits = torch.empty_like(partial)
            logits[local] = partial[local]

    if logits is None:
        raise ValueError("graph must contain at least one partition.")
    return logits


def load_partitioned_int8_model(
    graph: Data,
    model_record: dict[str, Any],
    checkpoint_path: str | Path,
) -> tuple[DistributedTinyGNN, Data, torch.Tensor, torch.Tensor]:
    """Restore the learned checkpoint and recreate its evaluation graph."""

    seed = int(model_record["seed"])
    set_seed(seed)
    graph = graph.to("cpu")
    train_mask, val_mask, _ = resolve_train_val_masks(graph, seed=seed)
    graph = normalize_graph_features(
        graph,
        train_mask=train_mask,
        standardize_columns=(
            model_record["feature_normalization"] == "train-standardized-row-l2"
        ),
    ).to("cpu")
    if model_record["message_protocol"] == "source-only":
        graph = make_inductive_view(graph, train_mask).to("cpu")
    graph.train_mask = train_mask

    config = model_record["model_config"]
    model = _make_model(
        "learned_int8",
        graph,
        train_mask,
        input_dim=int(model_record["input_dim"]),
        embedding_dim=int(model_record["embedding_dim"]),
        hidden_dim=int(model_record["hidden_dim"]),
        num_classes=int(model_record["num_classes"]),
        dropout=float(config["dropout"]),
        seed=seed,
    ).to("cpu")
    saved = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    model.load_state_dict(saved["model_state"])
    model.eval()
    return model, graph, train_mask, val_mask


def _make_model(
    system: str,
    graph: Data,
    train_mask: torch.Tensor,
    *,
    input_dim: int,
    embedding_dim: int | None,
    hidden_dim: int,
    num_classes: int,
    dropout: float,
    seed: int,
) -> nn.Module:
    if system == "local":
        return SimpleMLPBaseline(input_dim, MLP_HIDDEN_DIM, num_classes, dropout)
    if system in {"full", "full_int8"}:
        return FullFeatureGAT(input_dim, hidden_dim, num_classes, dropout)

    compressor = None
    if system == "pca":
        from sklearn.decomposition import PCA

        pca = PCA(
            n_components=embedding_dim,
            svd_solver="randomized",
            random_state=seed,
        ).fit(graph.x[train_mask].detach().cpu().numpy())
        compressor = FixedPCACompressor(
            torch.from_numpy(pca.mean_),
            torch.from_numpy(pca.components_),
        )
    return DistributedTinyGNN(
        input_dim,
        int(embedding_dim),
        hidden_dim,
        num_classes,
        dropout,
        compressor=compressor,
    )


def _relative_loss_drop(loss_curve: list[float]) -> float:
    if len(loss_curve) < 2 or loss_curve[0] <= 0:
        return 0.0
    return (loss_curve[0] - loss_curve[-1]) / loss_curve[0]


def _peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024 * 1024) if sys.platform == "darwin" else value / 1024


def _git_provenance() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        listed = subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=root,
            stderr=subprocess.DEVNULL,
        )
        digest = hashlib.sha256()
        for encoded in filter(None, listed.split(b"\0")):
            relative = encoded.decode("utf-8", errors="surrogateescape")
            path = root / relative
            if not path.is_file():
                continue
            digest.update(encoded)
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
        return {
            "git_commit": commit,
            "git_dirty": bool(status.strip()),
            "source_tree_sha256": digest.hexdigest(),
        }
    except (OSError, subprocess.CalledProcessError):
        return {
            "git_commit": "unknown",
            "git_dirty": None,
            "source_tree_sha256": None,
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
