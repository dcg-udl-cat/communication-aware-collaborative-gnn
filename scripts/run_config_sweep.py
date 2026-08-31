"""Ejecuta automaticamente varias configuraciones del experimento."""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from config import (
    SWEEP_EPOCHS,
    SWEEP_IMAGES_PER_NODE,
    SWEEP_K_NEIGHBORS,
    SWEEP_RESULTS_DIR,
    SWEEP_RUN_MODE,
)
from main import run
from utils.metrics import save_metrics, save_metrics_dataframe


def main() -> None:
    """Lanza todos los runs definidos en la configuracion del sweep."""

    args = _parse_args()
    sweep_results_dir = Path(args.output_dir) if args.output_dir else SWEEP_RESULTS_DIR

    rows: list[dict[str, Any]] = []
    combinations = list(
        itertools.product(
            args.k_neighbors,
            args.epochs,
            args.images_per_node,
        )
    )

    print(f"Ejecutando {len(combinations)} configuraciones automaticas")
    print(f"RUN_MODE del sweep: {args.run_mode}")

    for run_index, (k_neighbors, epochs, images_per_node) in enumerate(combinations, start=1):
        run_label = f"run_{run_index:03d}_k{k_neighbors}_e{epochs}_ipn{images_per_node}"
        run_dir = sweep_results_dir / run_label

        print(f"\n[{run_index}/{len(combinations)}] {run_label}")

        results = run(
            run_mode=args.run_mode,
            k_neighbors=k_neighbors,
            epochs=epochs,
            images_per_node=images_per_node,
            results_dir=run_dir,
            plots_dir=run_dir / "plots",
            run_label=run_label,
        )

        for model_name, metrics in results.items():
            rows.append(
                {
                    "run_label": run_label,
                    "model": model_name,
                    **metrics,
                }
            )

    summary_dataframe = pd.DataFrame(rows)
    summary_json = sweep_results_dir / "sweep_metrics.json"
    summary_csv = sweep_results_dir / "sweep_metrics.csv"

    save_metrics(summary_json, _rows_to_nested_dict(rows))
    save_metrics_dataframe(summary_csv, summary_dataframe)

    print("\nBarrido terminado")
    print(f"Resumen JSON: {summary_json}")
    print(f"Resumen CSV: {summary_csv}")


def _parse_args() -> argparse.Namespace:
    """Lee argumentos opcionales de consola."""

    parser = argparse.ArgumentParser(
        description="Ejecuta un barrido de KNN, epochs e imagenes por nodo.",
    )
    parser.add_argument(
        "--run-mode",
        default=SWEEP_RUN_MODE,
        choices=["resource_efficiency", "controlled_subset"],
        help="Modo de experimento usado en todos los runs.",
    )
    parser.add_argument(
        "--k-neighbors",
        type=int,
        nargs="+",
        default=SWEEP_K_NEIGHBORS,
        help="Valores de K para construir el grafo.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        nargs="+",
        default=SWEEP_EPOCHS,
        help="Numero de epocas a probar.",
    )
    parser.add_argument(
        "--images-per-node",
        type=int,
        nargs="+",
        default=SWEEP_IMAGES_PER_NODE,
        help="Numero de imagenes que forman cada nodo.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Carpeta donde guardar resultados del sweep.",
    )
    return parser.parse_args()


def _rows_to_nested_dict(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Convierte filas del sweep a un diccionario JSON legible."""

    nested: dict[str, dict[str, Any]] = {}
    for row in rows:
        run_label = str(row["run_label"])
        model_name = str(row["model"])

        nested.setdefault(run_label, {})
        nested[run_label][model_name] = {
            key: value for key, value in row.items() if key not in {"run_label", "model"}
        }

    return nested


if __name__ == "__main__":
    main()
