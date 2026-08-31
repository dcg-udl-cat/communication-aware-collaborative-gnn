"""Funciones de evaluacion con timing robusto."""

import statistics
import time
from dataclasses import dataclass

import torch
from torch import nn
from torch_geometric.data import Data

from training.trainer import model_forward

# Numero de repeticiones para la mediana del tiempo de inferencia.
_TIMING_WARMUP_RUNS = 5
_TIMING_MEASURE_RUNS = 20


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    macro_f1: float


@dataclass(frozen=True)
class TimingStats:
    median_s: float
    q1_s: float
    q3_s: float


def evaluate_accuracy(model: nn.Module, data: Data, mask: torch.Tensor) -> float:
    """Calcula accuracy sobre los nodos indicados por una mascara."""

    model.eval()

    with torch.no_grad():
        logits = model_forward(model, data)
        predictions = logits.argmax(dim=1)
        selected = mask.bool()
        total = int(selected.sum().item())
        if total == 0:
            return 0.0

        correct = int((predictions[selected] == data.y[selected]).sum().item())
        return correct / total


def evaluate_classification(
    model: nn.Module,
    data: Data,
    mask: torch.Tensor,
) -> ClassificationMetrics:
    """Return accuracy and macro-F1 for one split."""

    model.eval()
    with torch.no_grad():
        logits = model_forward(model, data)
    return classification_metrics_from_logits(logits, data.y, mask)


def classification_metrics_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> ClassificationMetrics:
    """Return accuracy and macro-F1 for precomputed logits."""

    from sklearn.metrics import f1_score

    predictions = logits.argmax(dim=1)
    selected = mask.bool()
    if not selected.any():
        return ClassificationMetrics(accuracy=0.0, macro_f1=0.0)
    truth = labels[selected].detach().cpu().numpy()
    predicted = predictions[selected].detach().cpu().numpy()
    return ClassificationMetrics(
        accuracy=float((predicted == truth).mean()),
        macro_f1=float(f1_score(truth, predicted, average="macro", zero_division=0)),
    )


def measure_inference_stats(
    model: nn.Module,
    data: Data,
    warmup_runs: int = 20,
    measure_runs: int = 50,
) -> TimingStats:
    """Measure median and interquartile CPU inference time."""

    if warmup_runs < 0 or measure_runs < 1:
        raise ValueError("warmup_runs must be non-negative and measure_runs positive.")
    model.eval()
    with torch.no_grad():
        for _ in range(warmup_runs):
            model_forward(model, data)
        elapsed = []
        for _ in range(measure_runs):
            start = time.perf_counter()
            model_forward(model, data)
            elapsed.append(time.perf_counter() - start)
    elapsed.sort()
    return TimingStats(
        median_s=statistics.median(elapsed),
        q1_s=_percentile(elapsed, 0.25),
        q3_s=_percentile(elapsed, 0.75),
    )


def _percentile(sorted_values: list[float], fraction: float) -> float:
    position = fraction * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def measure_inference_time(
    model: nn.Module,
    data: Data,
    warmup_runs: int = _TIMING_WARMUP_RUNS,
    measure_runs: int = _TIMING_MEASURE_RUNS,
) -> float:
    """Mide el tiempo mediano de un forward completo en segundos.

    Metodologia:
      1. Se realizan `warmup_runs` pasadas de calentamiento (descartadas)
         para estabilizar caches de CPU/GPU y JIT.
      2. Se realizan `measure_runs` pasadas cronometradas.
      3. Se devuelve la MEDIANA para reducir el impacto de interrupciones
         del sistema operativo o picos de carga esporadicos.

    Para benchmarks en GPU seria necesario anadir torch.cuda.synchronize()
    antes y despues de cada medicion. Esta implementacion es para CPU, que
    es el objetivo de despliegue en el borde.
    """

    return measure_inference_stats(
        model,
        data,
        warmup_runs=warmup_runs,
        measure_runs=measure_runs,
    ).median_s
