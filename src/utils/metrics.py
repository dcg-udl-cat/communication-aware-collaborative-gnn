"""Helpers para crear y preparar metricas del experimento."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from torch import nn

from config import AppConfig


def build_initial_metrics(config: AppConfig, class_names: Iterable[str]) -> dict[str, Any]:
    """Crea el fichero de metricas inicial de la fase base."""

    return {
        "project": "tfg_gnn",
        "status": "initialized",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": config.data.dataset_name,
            "raw_dir": str(config.data.raw_dir),
            "classes": list(class_names),
        },
        "graph": {
            "k_neighbors": config.graph.k_neighbors,
            "feature_dim": config.graph.feature_dim,
            "status": "pending",
        },
        "training": {
            "epochs": config.training.epochs,
            "learning_rate": config.training.learning_rate,
            "device": config.training.device,
            "status": "pending",
        },
        "models": {
            "full_gnn": {"status": "pending"},
            "tiny_gnn": {"status": "pending"},
        },
        "runs": [],
    }


def count_trainable_parameters(model: nn.Module) -> int:
    """Cuenta los parametros entrenables de un modelo."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def build_metrics_dataframe(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Convierte las metricas de modelos en un DataFrame de pandas."""

    rows = [{"model": model_name, **metrics} for model_name, metrics in results.items()]
    dataframe = pd.DataFrame(rows)

    preferred_columns = [
        "model",
        "run_label",
        "run_mode",
        "model_type",
        "k_neighbors",
        "epochs",
        "images_per_node",
        "train_subset_size",
        "test_subset_size",
        "dataset_mode",
        "train_loss",
        "train_accuracy",
        "test_accuracy",
        "accuracy_gap",
        "inference_time",
        "num_parameters",
        "num_nodes",
        "num_edges",
        "feature_dim",
        "train_nodes",
        "test_nodes",
        "total_images",
        "mean_images_per_node",
    ]
    existing_columns = [column for column in preferred_columns if column in dataframe.columns]
    extra_columns = [column for column in dataframe.columns if column not in existing_columns]

    return dataframe[existing_columns + extra_columns]


def save_metrics(path: Path, metrics: dict[str, Any]) -> None:
    """Guarda metricas en formato JSON legible."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)
        file.write("\n")


def save_metrics_dataframe(path: Path, dataframe: pd.DataFrame) -> None:
    """Guarda un DataFrame de metricas en CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)
