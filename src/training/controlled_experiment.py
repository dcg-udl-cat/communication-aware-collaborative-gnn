"""Experimento controlado: A vs B vs C.

Pregunta que responde:
  ¿Aportan realmente las aristas del grafo informacion util al tiny GNN?

Los tres setups comparten TODOS los hiperparametros (datos, features, epocas,
semillas) y solo difieren en la estructura del grafo:

  Setup A — MLP sin grafo
    Linea base pura. Solo features de nodo, ninguna informacion de vecinos.
    Si B no supera A, el grafo no aporta nada.

  Setup B — Tiny GNN + aristas KNN reales
    Aristas construidas por similitud de features (KNN coseno sobre PCA-64).
    Si B > A, las aristas transmiten informacion util.

  Setup C — Tiny GNN + aristas aleatorias
    Misma arquitectura y numero de aristas que B, pero los vecinos son
    aleatorios (sin relacion con similitud de features).
    Si B ≈ C, el grafo no importa (la arquitectura GNN tampoco).
    Si B > C, la estructura KNN es la que aporta valor, no el mecanismo GNN per se.

Interpretacion:
  B > A  y  B > C  →  aristas KNN utiles, GNN justificado    ✅
  B > A  y  B ≈ C  →  GNN ayuda pero la topologia no importa  🤷
  B ≈ A            →  aristas no aportan nada                 ❌
  B < A            →  grafo introduce ruido, perjudicial       ⚠️

Condiciones que maximizan la probabilidad de ver B > A:
  - images_per_node = 1  (nodos limpios, sin ruido de promedio)
  - PCA 64D + normalizacion L2 (features estables)
  - KNN coseno k=5..8 (vecindades semanticamente coherentes)
  - Tiny GNN: GCNConv(64→32) ReLU GCNConv(32→10) (sin sobreajuste)
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any, Type

import torch
import torch.nn.functional as functional
from torch import nn
from torch_geometric.data import Data

from config import (
    CONTROLLED_TEST_SUBSET_SIZE,
    CONTROLLED_TRAIN_SUBSET_SIZE,
    DEVICE,
    EPOCHS,
    K_NEIGHBORS,
    KNN_METRIC,
    LEARNING_RATE,
    PCA_COMPONENTS,
    RANDOM_SEED,
    USE_PCA,
)
from data.feature_extractor import build_node_features_from_images
from data.graph_builder import build_similarity_graph
from data.pca_compressor import PCACompressor
from training.evaluator import evaluate_accuracy, measure_inference_time
from training.trainer import train_one_epoch
from utils.seed import set_seed

# images_per_node=1 is non-negotiable for the controlled experiment.
# A tiny GNN has no capacity to recover from noisy averaged features.
_IMAGES_PER_NODE = 1

_TIMING_WARMUP = 5
_TIMING_RUNS = 20


@dataclass
class SetupResult:
    """Resultado de un setup del experimento controlado."""

    setup: str                    # "A_mlp", "B_knn_gnn", "C_random_gnn"
    test_accuracy: float
    val_accuracy: float
    train_accuracy: float
    inference_time_median_s: float
    num_parameters: int
    num_edges: int                # 0 para setup A
    edge_type: str                # "none", "knn", "random"
    seed: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "setup": self.setup,
            "test_accuracy": self.test_accuracy,
            "val_accuracy": self.val_accuracy,
            "train_accuracy": self.train_accuracy,
            "inference_time_median_s": self.inference_time_median_s,
            "num_parameters": self.num_parameters,
            "num_edges": self.num_edges,
            "edge_type": self.edge_type,
            "seed": self.seed,
        }


@dataclass
class ControlledExperimentResult:
    """Resultados agregados de los tres setups, con interpretacion automatica."""

    setup_a: SetupResult
    setup_b: SetupResult
    setup_c: SetupResult
    seeds_used: list[int]

    @property
    def b_over_a(self) -> float:
        return self.setup_b.test_accuracy - self.setup_a.test_accuracy

    @property
    def b_over_c(self) -> float:
        return self.setup_b.test_accuracy - self.setup_c.test_accuracy

    @property
    def interpretation(self) -> str:
        threshold = 0.01   # 1 punto porcentual de margen minimo
        b_beats_a = self.b_over_a > threshold
        b_beats_c = self.b_over_c > threshold
        b_equals_a = abs(self.b_over_a) <= threshold
        b_equals_c = abs(self.b_over_c) <= threshold
        b_worse_a = self.b_over_a < -threshold

        if b_worse_a:
            return (
                "⚠️  B < A: El grafo perjudica al modelo. "
                "Las aristas introducen ruido en lugar de señal. "
                "Revisa la construccion del grafo o aumenta k."
            )
        if b_beats_a and b_beats_c:
            return (
                "✅  B > A y B > C: Las aristas KNN aportan informacion util. "
                "El grafo esta justificado para este pipeline."
            )
        if b_beats_a and b_equals_c:
            return (
                "🤷  B > A pero B ≈ C: El mecanismo GNN ayuda, pero la "
                "topologia KNN no importa mas que aristas aleatorias. "
                "Considera simplificar la construccion del grafo."
            )
        if b_equals_a:
            return (
                "❌  B ≈ A: Las aristas no aportan nada. "
                "El grafo no añade valor sobre el MLP. "
                "Posibles causas: k demasiado pequeño, features poco informativas, "
                "o el problema no tiene estructura de grafo util."
            )
        return (
            f"ℹ️  Resultado mixto. B-A={self.b_over_a:+.3f}, B-C={self.b_over_c:+.3f}. "
            "Revisa los resultados individuales."
        )

    def summary(self) -> str:
        lines = [
            "",
            "=" * 65,
            "EXPERIMENTO CONTROLADO A / B / C",
            f"  images_per_node = {_IMAGES_PER_NODE}  |  PCA = {PCA_COMPONENTS}D  |  "
            f"k = {K_NEIGHBORS}  |  metric = {KNN_METRIC}",
            "-" * 65,
            f"  A (MLP, sin grafo)      acc={self.setup_a.test_accuracy:.3f}  "
            f"edges=0         params={self.setup_a.num_parameters}",
            f"  B (GNN, aristas KNN)   acc={self.setup_b.test_accuracy:.3f}  "
            f"edges={self.setup_b.num_edges:<8d}  params={self.setup_b.num_parameters}",
            f"  C (GNN, aristas rand.) acc={self.setup_c.test_accuracy:.3f}  "
            f"edges={self.setup_c.num_edges:<8d}  params={self.setup_c.num_parameters}",
            "-" * 65,
            f"  B - A = {self.b_over_a:+.3f}   B - C = {self.b_over_c:+.3f}",
            "-" * 65,
            self.interpretation,
            "=" * 65,
        ]
        return "\n".join(lines)


def run_controlled_experiment(
    mlp_class: Type[nn.Module],
    gnn_class: Type[nn.Module],
    mlp_kwargs: dict[str, Any] | None = None,
    gnn_kwargs: dict[str, Any] | None = None,
    k_neighbors: int = K_NEIGHBORS,
    knn_metric: str = KNN_METRIC,
    epochs: int = EPOCHS,
    seeds: list[int] | None = None,
    device_str: str = DEVICE,
) -> ControlledExperimentResult:
    """Ejecuta los tres setups y devuelve el resultado con interpretacion.

    Los tres setups comparten datos, features y semillas.
    Solo difieren en la estructura del grafo pasada al modelo.

    Args:
        mlp_class:   Clase del MLP para setup A (e.g. SimpleMLPBaseline).
        gnn_class:   Clase del GNN para setups B y C (e.g. TinyGNN).
        mlp_kwargs:  Kwargs del constructor del MLP. input_dim se inyecta.
        gnn_kwargs:  Kwargs del constructor del GNN. input_dim se inyecta.
        k_neighbors: K para el grafo KNN real (setup B).
        knn_metric:  Metrica KNN: "cosine" o "euclidean".
        epochs:      Epocas de entrenamiento por setup.
        seeds:       Lista de semillas para multi-seed. None → [RANDOM_SEED].
        device_str:  Device torch.
    """

    seeds = seeds or [RANDOM_SEED]
    device = torch.device(device_str)

    # Acumuladores multi-seed por setup
    results_a: list[SetupResult] = []
    results_b: list[SetupResult] = []
    results_c: list[SetupResult] = []

    for seed in seeds:
        set_seed(seed)

        # --- Datos y features (compartidos por los tres setups) ---
        import config as runtime_config
        from data import cifar_loader

        cifar = cifar_loader.load_cifar10(
            train_subset_size=runtime_config.CONTROLLED_TRAIN_SUBSET_SIZE,
            test_subset_size=runtime_config.CONTROLLED_TEST_SUBSET_SIZE,
        )

        gen = torch.Generator().manual_seed(seed)
        train_features_raw, train_labels, train_counts = build_node_features_from_images(
            images=cifar.train.images,
            labels=cifar.train.labels,
            images_per_node=_IMAGES_PER_NODE,
            shuffle=True,
            generator=gen,
        )
        test_features_raw, test_labels, test_counts = build_node_features_from_images(
            images=cifar.test.images,
            labels=cifar.test.labels,
            images_per_node=_IMAGES_PER_NODE,
            shuffle=True,
            generator=gen,
        )

        # PCA ajustado SOLO sobre train
        if runtime_config.USE_PCA:
            compressor = PCACompressor(n_components=PCA_COMPONENTS)
            train_features = compressor.fit_transform(train_features_raw)
            test_features = compressor.transform(test_features_raw)
        else:
            train_features = train_features_raw
            test_features = test_features_raw

        # L2-normalize antes del KNN coseno (sklearn cosine ya lo hace internamente,
        # pero la normalizacion explicita garantiza que las features en el grafo
        # tambien esten normalizadas, mejorando la estabilidad del GNN).
        train_features = functional.normalize(train_features, p=2, dim=1)
        test_features = functional.normalize(test_features, p=2, dim=1)

        feat_dim = train_features.shape[1]
        eff_mlp_kwargs = dict(mlp_kwargs or {})
        eff_mlp_kwargs.setdefault("input_dim", feat_dim)
        eff_gnn_kwargs = dict(gnn_kwargs or {})
        eff_gnn_kwargs.setdefault("input_dim", feat_dim)

        # --- Setup B: grafo KNN real ---
        knn_graph = build_similarity_graph(
            train_features=train_features,
            train_labels=train_labels,
            test_features=test_features,
            test_labels=test_labels,
            train_node_image_counts=train_counts,
            test_node_image_counts=test_counts,
            k_neighbors=k_neighbors,
            knn_metric=knn_metric,
        ).to(device)

        # --- Setup C: mismo grafo pero aristas aleatorias ---
        # Mismo numero de aristas que B, distribuidas aleatoriamente.
        # Esto aisla el efecto de la topologia KNN del mecanismo de mensaje.
        random_graph = _replace_edges_with_random(knn_graph, seed=seed).to(device)

        # --- Setup A: grafo sin aristas (MLP usa solo features de nodo) ---
        # Pasamos el mismo grafo KNN pero el MLP ignora edge_index.
        results_a.append(_run_setup(
            setup_name="A_mlp",
            edge_type="none",
            model=mlp_class(**eff_mlp_kwargs),
            graph=knn_graph,
            device=device,
            epochs=epochs,
            seed=seed,
        ))
        results_b.append(_run_setup(
            setup_name="B_knn_gnn",
            edge_type="knn",
            model=gnn_class(**eff_gnn_kwargs),
            graph=knn_graph,
            device=device,
            epochs=epochs,
            seed=seed,
        ))
        results_c.append(_run_setup(
            setup_name="C_random_gnn",
            edge_type="random",
            model=gnn_class(**eff_gnn_kwargs),
            graph=random_graph,
            device=device,
            epochs=epochs,
            seed=seed,
        ))

    return ControlledExperimentResult(
        setup_a=_aggregate(results_a),
        setup_b=_aggregate(results_b),
        setup_c=_aggregate(results_c),
        seeds_used=seeds,
    )


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _replace_edges_with_random(graph: Data, seed: int) -> Data:
    """Devuelve una copia del grafo con edge_index aleatorio.

    Conserva exactamente el mismo numero de aristas que el grafo original
    y el mismo conjunto de nodos, para que la comparacion B vs C sea justa:
    la unica variable es si los vecinos son semanticamente relevantes (KNN)
    o completamente aleatorios.

    Las aristas aleatorias respetan la direccion inductiva de PyG:
    train→test permite que test reciba mensajes; test→train introduciria
    features de test en embeddings de entrenamiento.
    """

    num_nodes = graph.num_nodes
    num_train = int(graph.train_mask.sum().item())
    src, dst = graph.edge_index

    gen = torch.Generator().manual_seed(seed + 999)

    new_src_list = []
    new_dst_list = []

    for s, d in zip(src.tolist(), dst.tolist()):
        s_is_test = s >= num_train
        d_is_test = d >= num_train

        if not s_is_test and d_is_test:
            # Keep the source so leakage-safe filtering removes exactly the
            # same number of messages; randomize which test target receives it.
            new_src_list.append(s)
            new_dst_list.append(
                int(torch.randint(num_train, num_nodes, (1,), generator=gen).item())
            )
        elif not s_is_test and not d_is_test:
            # train → train: aleatorizar destino dentro de train
            new_src_list.append(s)
            new_dst_list.append(int(torch.randint(0, num_train, (1,), generator=gen).item()))
        else:
            # Otros tipos (train→test en undirected): mantener proporcional
            new_src_list.append(int(torch.randint(0, num_nodes, (1,), generator=gen).item()))
            new_dst_list.append(int(torch.randint(0, num_nodes, (1,), generator=gen).item()))

    random_edge_index = torch.tensor(
        [new_src_list, new_dst_list], dtype=torch.long
    )

    # Copia superficial del Data con edge_index sustituido
    new_graph = graph.clone()
    new_graph.edge_index = random_edge_index
    if hasattr(new_graph, "edge_stats"):
        new_graph.edge_stats = {**graph.edge_stats, "randomized": True}
    return new_graph


def _run_setup(
    setup_name: str,
    edge_type: str,
    model: nn.Module,
    graph: Data,
    device: torch.device,
    epochs: int,
    seed: int,
) -> SetupResult:
    """Entrena y evalua un setup, con split de validacion 80/20."""

    from utils.metrics import count_trainable_parameters
    from training.trainer import model_forward

    set_seed(seed)
    model = model.to(device)
    graph = graph.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=LEARNING_RATE) if params else None
    loss_fn = nn.CrossEntropyLoss()

    from training.splits import make_inductive_view, stratified_train_val_masks
    eff_train_mask, val_mask = stratified_train_val_masks(
        graph.y, graph.train_mask, seed=seed
    )
    graph = make_inductive_view(graph, eff_train_mask).to(device)

    orig_train_mask = graph.train_mask.clone()
    graph.train_mask = eff_train_mask

    for _ in range(epochs):
        train_one_epoch(model=model, data=graph, optimizer=optimizer,
                        loss_fn=loss_fn, device=device)

    graph.train_mask = orig_train_mask

    train_acc = evaluate_accuracy(model, graph, eff_train_mask)
    val_acc = evaluate_accuracy(model, graph, val_mask)
    test_acc = evaluate_accuracy(model, graph, graph.test_mask)
    latency = measure_inference_time(model, graph,
                                     warmup_runs=_TIMING_WARMUP,
                                     measure_runs=_TIMING_RUNS)

    return SetupResult(
        setup=setup_name,
        test_accuracy=test_acc,
        val_accuracy=val_acc,
        train_accuracy=train_acc,
        inference_time_median_s=latency,
        num_parameters=count_trainable_parameters(model),
        num_edges=int(graph.num_edges),
        edge_type=edge_type,
        seed=seed,
    )


def _aggregate(results: list[SetupResult]) -> SetupResult:
    """Promedia metricas de multiples semillas. Devuelve un SetupResult agregado."""

    if len(results) == 1:
        return results[0]

    base = results[0]
    keys = ["test_accuracy", "val_accuracy", "train_accuracy", "inference_time_median_s"]
    averaged = {k: statistics.mean(getattr(r, k) for r in results) for k in keys}

    return SetupResult(
        setup=base.setup,
        test_accuracy=averaged["test_accuracy"],
        val_accuracy=averaged["val_accuracy"],
        train_accuracy=averaged["train_accuracy"],
        inference_time_median_s=averaged["inference_time_median_s"],
        num_parameters=base.num_parameters,
        num_edges=base.num_edges,
        edge_type=base.edge_type,
        seed=-1,   # -1 indica resultado agregado
    )
