"""Utilidades sencillas para guardar graficos de metricas."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import pandas as pd
from matplotlib import pyplot as plt


def save_metrics_bar_charts(dataframe: pd.DataFrame, output_dir: Path) -> None:
    """Guarda graficos de barras para las metricas principales."""

    output_dir.mkdir(parents=True, exist_ok=True)

    metric_columns = [
        "test_accuracy",
        "accuracy_gap",
        "inference_time",
        "num_parameters",
    ]

    for metric in metric_columns:
        if metric not in dataframe.columns:
            continue

        _save_bar_chart(
            dataframe=dataframe,
            metric=metric,
            output_path=output_dir / f"{metric}.png",
        )

    _save_bar_chart(dataframe, "test_accuracy", output_dir / "comparison_accuracy.png")
    _save_bar_chart(
        dataframe,
        "inference_time",
        output_dir / "comparison_inference_time.png",
    )
    _save_bar_chart(
        dataframe,
        "num_parameters",
        output_dir / "comparison_parameters.png",
    )
    _save_tradeoff_chart(
        dataframe=dataframe,
        output_path=output_dir / "tradeoff_accuracy_vs_inference.png",
    )


def _save_bar_chart(dataframe: pd.DataFrame, metric: str, output_path: Path) -> None:
    """Guarda un grafico de barras de una metrica concreta."""

    if metric not in dataframe.columns:
        return

    figure, axis = plt.subplots(figsize=(7, 4))

    axis.bar(dataframe["model"], dataframe[metric])
    axis.set_title(metric.replace("_", " ").title())
    axis.set_xlabel("Modelo")
    axis.set_ylabel(metric)
    axis.grid(axis="y", alpha=0.3)

    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def _save_tradeoff_chart(dataframe: pd.DataFrame, output_path: Path) -> None:
    """Guarda un scatter de accuracy frente a tiempo de inferencia."""

    required_columns = {"model", "inference_time", "test_accuracy", "num_parameters"}
    if not required_columns.issubset(dataframe.columns):
        return

    figure, axis = plt.subplots(figsize=(7, 5))

    max_parameters = max(float(dataframe["num_parameters"].max()), 1.0)
    point_sizes = 80 + 420 * (dataframe["num_parameters"] / max_parameters)

    axis.scatter(
        dataframe["inference_time"],
        dataframe["test_accuracy"],
        s=point_sizes,
        alpha=0.7,
    )

    for _, row in dataframe.iterrows():
        axis.annotate(
            row["model"],
            (row["inference_time"], row["test_accuracy"]),
            textcoords="offset points",
            xytext=(6, 6),
        )

    axis.set_title("Accuracy vs Inference Time")
    axis.set_xlabel("Inference time (s)")
    axis.set_ylabel("Test accuracy")
    axis.grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)
