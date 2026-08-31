"""Sweep de optimizacion de aristas: tradeoff accuracy/latencia en el borde.

Que mide este sweep:
  Para cada combinacion de (k_neighbors, connectivity_mode):
    - Numero de aristas (train-train, test-train, total)
    - Tiempo mediano de inferencia (warmup + mediana de N repeticiones)
    - Test accuracy del modelo entrenado
    - Reduccion de aristas respecto a la configuracion de referencia

Esto permite elegir la configuracion minima de conectividad que mantiene
una accuracy aceptable, que es exactamente lo que importa en el borde:
no el mejor modelo posible, sino el mas rapido que todavia funciona.

Uso:
    from training.edge_sweep import run_edge_sweep
    results = run_edge_sweep(model_class=TinyGNN)
"""

from __future__ import annotations

import time
import statistics
from dataclasses import dataclass, field
from typing import Any, Type

import torch
from torch import nn
from torch_geometric.data import Data

from config import (
    CONNECTIVITY_MODE,
    CONTROLLED_TEST_SUBSET_SIZE,
    CONTROLLED_TRAIN_SUBSET_SIZE,
    DEVICE,
    EPOCHS,
    IMAGES_PER_NODE,
    K_NEIGHBORS,
    LEARNING_RATE,
    PCA_COMPONENTS,
    RANDOM_SEED,
    SWEEP_CONNECTIVITY_MODES,
    SWEEP_K_NEIGHBORS_EDGE,
    USE_PCA,
)
from data.cifar_loader import load_cifar10
from data.feature_extractor import build_node_features_from_images
from data.graph_builder import build_similarity_graph
from data.pca_compressor import PCACompressor
from training.evaluator import evaluate_accuracy
from training.trainer import train_one_epoch
from utils.seed import set_seed

_TIMING_WARMUP = 5
_TIMING_RUNS = 30   # mas repeticiones que el default porque latencia es el punto


@dataclass
class EdgeSweepResult:
    """Resultado de una configuracion (k, connectivity_mode) del sweep."""

    k_neighbors: int
    connectivity_mode: str
    num_train_train_edges: int
    num_train_test_edges: int
    num_total_edges: int
    test_accuracy: float
    inference_time_median_s: float
    inference_time_p25_s: float
    inference_time_p75_s: float
    edge_reduction_vs_reference: float   # fraccion: 1 - (total / reference_total)
    latency_reduction_vs_reference: float
    seed: int
    pca_enabled: bool
    pca_components: int | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "k_neighbors": self.k_neighbors,
            "connectivity_mode": self.connectivity_mode,
            "num_train_train_edges": self.num_train_train_edges,
            "num_train_test_edges": self.num_train_test_edges,
            "num_total_edges": self.num_total_edges,
            "test_accuracy": self.test_accuracy,
            "inference_time_median_s": self.inference_time_median_s,
            "inference_time_p25_s": self.inference_time_p25_s,
            "inference_time_p75_s": self.inference_time_p75_s,
            "edge_reduction_vs_reference": self.edge_reduction_vs_reference,
            "latency_reduction_vs_reference": self.latency_reduction_vs_reference,
            "seed": self.seed,
            "pca_enabled": self.pca_enabled,
            "pca_components": self.pca_components,
        }


def run_edge_sweep(
    model_class: Type[nn.Module],
    model_kwargs: dict[str, Any] | None = None,
    k_values: list[int] = SWEEP_K_NEIGHBORS_EDGE,
    connectivity_modes: list[str] = SWEEP_CONNECTIVITY_MODES,
    reference_k: int = K_NEIGHBORS,
    reference_mode: str = CONNECTIVITY_MODE,
    epochs: int = EPOCHS,
    seed: int = RANDOM_SEED,
    device_str: str = DEVICE,
) -> list[EdgeSweepResult]:
    """Ejecuta el sweep completo sobre k y connectivity_mode.

    Para cada configuracion:
      1. Construye el grafo con esa (k, mode).
      2. Entrena el modelo desde cero (misma semilla -> comparable).
      3. Mide accuracy y latencia de inferencia.
      4. Calcula reducciones respecto a la configuracion de referencia.

    Args:
        model_class:        Clase del modelo a entrenar (e.g. TinyGNN).
        model_kwargs:       Kwargs extra para el constructor del modelo.
                            INPUT_DIM se inyecta automaticamente.
        k_values:           Lista de k a explorar.
        connectivity_modes: Lista de modos a explorar.
        reference_k:        k de referencia para calcular reducciones.
        reference_mode:     Modo de referencia para calcular reducciones.
        epochs:             Epocas de entrenamiento por configuracion.
        seed:               Semilla para reproducibilidad.
        device_str:         Device de torch ("cpu", "cuda", ...).

    Returns:
        Lista de EdgeSweepResult, uno por combinacion (k, mode).
    """

    set_seed(seed)
    device = torch.device(device_str)

    # Carga de datos una sola vez (el grafo se reconstruye por configuracion).
    cifar_data = load_cifar10(
        train_subset_size=CONTROLLED_TRAIN_SUBSET_SIZE,
        test_subset_size=CONTROLLED_TEST_SUBSET_SIZE,
    )

    # Features de imagen (pre-PCA): se computan una vez y se reutilizan.
    gen = torch.Generator().manual_seed(seed)
    train_features_raw, train_labels, train_counts = build_node_features_from_images(
        images=cifar_data.train.images,
        labels=cifar_data.train.labels,
        images_per_node=IMAGES_PER_NODE,
        shuffle=True,
        generator=gen,
    )
    test_features_raw, test_labels, test_counts = build_node_features_from_images(
        images=cifar_data.test.images,
        labels=cifar_data.test.labels,
        images_per_node=IMAGES_PER_NODE,
        shuffle=True,
        generator=gen,
    )

    # PCA: ajustado una sola vez sobre train (no por configuracion de k).
    compressor: PCACompressor | None = None
    if USE_PCA:
        compressor = PCACompressor(n_components=PCA_COMPONENTS)
        train_features = compressor.fit_transform(train_features_raw)
        test_features = compressor.transform(test_features_raw)
    else:
        train_features = train_features_raw
        test_features = test_features_raw

    feat_dim = train_features.shape[1]
    effective_model_kwargs = dict(model_kwargs or {})
    effective_model_kwargs.setdefault("input_dim", feat_dim)

    # Configuracion de referencia para calcular reducciones relativas.
    reference_graph = _build_graph(
        train_features, train_labels, train_counts,
        test_features, test_labels, test_counts,
        k_neighbors=reference_k,
        connectivity_mode=reference_mode,
    )
    reference_edges = int(reference_graph.edge_stats["total"])
    reference_latency = _measure_latency(
        model_class=model_class,
        model_kwargs=effective_model_kwargs,
        graph=reference_graph,
        device=device,
        epochs=epochs,
        seed=seed,
    )

    results: list[EdgeSweepResult] = []

    for k in k_values:
        for mode in connectivity_modes:
            print(f"  k={k:2d} | mode={mode:<12s}", end=" ", flush=True)

            graph = _build_graph(
                train_features, train_labels, train_counts,
                test_features, test_labels, test_counts,
                k_neighbors=k,
                connectivity_mode=mode,
            )
            edge_stats = graph.edge_stats

            accuracy, latency_s, p25, p75 = _train_and_measure(
                model_class=model_class,
                model_kwargs=effective_model_kwargs,
                graph=graph,
                device=device,
                epochs=epochs,
                seed=seed,
            )

            total_edges = int(edge_stats["total"])
            edge_reduction = 1.0 - (total_edges / reference_edges) if reference_edges > 0 else 0.0
            latency_reduction = 1.0 - (latency_s / reference_latency) if reference_latency > 0 else 0.0

            print(
                f"edges={total_edges:5d} "
                f"acc={accuracy:.3f} "
                f"latency={latency_s*1000:.2f}ms "
                f"(-{edge_reduction:.0%} edges, -{latency_reduction:.0%} latency)"
            )

            results.append(EdgeSweepResult(
                k_neighbors=k,
                connectivity_mode=mode,
                num_train_train_edges=int(edge_stats["train_train"]),
                num_train_test_edges=int(edge_stats["train_test"]),
                num_total_edges=total_edges,
                test_accuracy=accuracy,
                inference_time_median_s=latency_s,
                inference_time_p25_s=p25,
                inference_time_p75_s=p75,
                edge_reduction_vs_reference=edge_reduction,
                latency_reduction_vs_reference=latency_reduction,
                seed=seed,
                pca_enabled=compressor is not None,
                pca_components=compressor.output_dim if compressor is not None else None,
            ))

    return results


def summarize_edge_sweep(results: list[EdgeSweepResult]) -> None:
    """Imprime un resumen ordenado por latencia con el tradeoff accuracy/speed."""

    print("\n" + "=" * 72)
    print("SWEEP DE CONECTIVIDAD — tradeoff accuracy / latencia")
    print(f"{'k':>4}  {'mode':<12}  {'edges':>6}  {'acc':>6}  {'ms':>7}  {'Δedges':>8}  {'Δms':>8}")
    print("-" * 72)
    for r in sorted(results, key=lambda x: x.inference_time_median_s):
        print(
            f"{r.k_neighbors:>4}  {r.connectivity_mode:<12}  "
            f"{r.num_total_edges:>6}  "
            f"{r.test_accuracy:>6.3f}  "
            f"{r.inference_time_median_s * 1000:>7.2f}  "
            f"{-r.edge_reduction_vs_reference:>+8.1%}  "
            f"{-r.latency_reduction_vs_reference:>+8.1%}"
        )
    print("=" * 72)

    # Configuracion optima: menor latencia con accuracy >= 90% de la mejor.
    best_acc = max(r.test_accuracy for r in results)
    threshold = 0.90 * best_acc
    candidates = [r for r in results if r.test_accuracy >= threshold]
    if candidates:
        optimal = min(candidates, key=lambda x: x.inference_time_median_s)
        print(
            f"\nOptimo (acc >= {threshold:.3f}, latencia minima): "
            f"k={optimal.k_neighbors}, mode={optimal.connectivity_mode}, "
            f"{optimal.num_total_edges} aristas, "
            f"{optimal.inference_time_median_s * 1000:.2f}ms"
        )


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _build_graph(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    train_counts: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    test_counts: torch.Tensor,
    k_neighbors: int,
    connectivity_mode: str,
) -> Data:
    return build_similarity_graph(
        train_features=train_features,
        train_labels=train_labels,
        test_features=test_features,
        test_labels=test_labels,
        train_node_image_counts=train_counts,
        test_node_image_counts=test_counts,
        k_neighbors=k_neighbors,
        connectivity_mode=connectivity_mode,
    )


def _train_model(
    model: nn.Module,
    graph: Data,
    device: torch.device,
    epochs: int,
) -> None:
    """Entrena el modelo in-place."""

    graph = graph.to(device)
    model = model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        return
    optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        train_one_epoch(model=model, data=graph, optimizer=optimizer,
                        loss_fn=loss_fn, device=device)


def _measure_latency(
    model_class: type,
    model_kwargs: dict[str, Any],
    graph: Data,
    device: torch.device,
    epochs: int,
    seed: int,
) -> float:
    """Entrena y devuelve solo la latencia mediana (para la referencia)."""

    _, latency, _, _ = _train_and_measure(
        model_class=model_class,
        model_kwargs=model_kwargs,
        graph=graph,
        device=device,
        epochs=epochs,
        seed=seed,
    )
    return latency


def _train_and_measure(
    model_class: type,
    model_kwargs: dict[str, Any],
    graph: Data,
    device: torch.device,
    epochs: int,
    seed: int,
) -> tuple[float, float, float, float]:
    """Entrena el modelo y devuelve (accuracy, median_s, p25_s, p75_s)."""

    from training.trainer import model_forward

    set_seed(seed)
    model = model_class(**model_kwargs)
    _train_model(model, graph, device, epochs)

    graph_dev = graph.to(device)
    model = model.to(device)
    model.eval()

    # Accuracy
    accuracy = evaluate_accuracy(model, graph_dev, graph_dev.test_mask)

    # Latencia: warmup + medicion con percentiles
    with torch.no_grad():
        for _ in range(_TIMING_WARMUP):
            model_forward(model, graph_dev)

        times: list[float] = []
        for _ in range(_TIMING_RUNS):
            t0 = time.perf_counter()
            model_forward(model, graph_dev)
            times.append(time.perf_counter() - t0)

    times_sorted = sorted(times)
    n = len(times_sorted)
    median = statistics.median(times_sorted)
    p25 = times_sorted[n // 4]
    p75 = times_sorted[(3 * n) // 4]

    return accuracy, median, p25, p75
