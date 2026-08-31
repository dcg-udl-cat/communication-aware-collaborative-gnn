"""Construccion de un grafo de similitud con k-nearest neighbors.

Modos de conectividad (connectivity_mode):

  Tres modos forman una progresion clara de mayor a menor coste:

  "undirected"    — train↔train  +  train↔test
    Todas las aristas son bidireccionales. Los nodos de train reciben
    mensajes de nodos de test, rompiendo el protocolo inductivo. Util
    solo como linea base para cuantificar el coste de la direccionalidad
    correcta en accuracy.
    Coste relativo de message passing: ALTO (~2x train_test vs directed)

  "directed"      — train↔train  +  train→test      [DEFAULT]
    Las aristas train-train son bidireccionales (flujo libre entre nodos
    conocidos). PyG interpreta edge_index como source→target, de modo que
    train→test permite que test agregue embeddings de train sin que train
    reciba mensajes de test. Protocolo inductivo correcto.
    Coste relativo de mensaje-passing: MEDIO

  "half_directed" — train→train  +  train→test
    Todas las aristas son unidireccionales. Las aristas train-train se
    añaden una sola vez (sin la inversa), reduciendo el numero de aristas
    ~2x respecto a "directed". El paso de mensajes en train es asimetrico:
    cada nodo solo agrega vecinos en una direccion. Menor latencia a costa
    de menor flujo de informacion entre nodos de train.
    Coste relativo de mensaje-passing: BAJO (~0.5x vs directed)

    Nota: "sparse" es un alias de "half_directed" mantenido por
    compatibilidad. Se normalizara automaticamente con un aviso.

Resumen del tradeoff (misma k):
  undirected > directed > half_directed
  (aristas)    (coste)    (latencia)

Combinado con k variable (2, 4, 8), estos tres modos permiten medir el
tradeoff accuracy/latencia que importa en despliegue en el borde.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data

from config import CONNECTIVITY_MODE, K_NEIGHBORS, KNN_METRIC, USE_MUTUAL_KNN, EDGE_WEIGHT_SIGMA

# Modos validos de conectividad.
# "sparse" is a deprecated alias for "half_directed" — normalised in build_similarity_graph.
_VALID_MODES = {"directed", "undirected", "half_directed", "sparse"}


def build_similarity_graph(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    train_node_image_counts: torch.Tensor | None = None,
    test_node_image_counts: torch.Tensor | None = None,
    k_neighbors: int = K_NEIGHBORS,
    connectivity_mode: str = CONNECTIVITY_MODE,
    knn_metric: str = KNN_METRIC,
    use_mutual_knn: bool = USE_MUTUAL_KNN,
    edge_weight_sigma: float | None = EDGE_WEIGHT_SIGMA,
) -> Data:
    """Construye un grafo de similitud KNN con el modo de conectividad indicado.

    Args:
        train_features:          (N_train, D) features de nodos de train.
        train_labels:            (N_train,)   etiquetas de nodos de train.
        test_features:           (N_test, D)  features de nodos de test.
        test_labels:             (N_test,)    etiquetas de nodos de test.
        train_node_image_counts: (N_train,)   imagenes por nodo de train.
        test_node_image_counts:  (N_test,)    imagenes por nodo de test.
        k_neighbors:             Numero de vecinos mas cercanos.
        connectivity_mode:       "directed" | "undirected" | "half_directed" (alias: "sparse").
        knn_metric:              Metrica de distancia para KNN.
                                 "cosine" recomendado con features L2-normalizadas (PCA).
                                 "euclidean" util para ablacion.

    Returns:
        Data de PyG con x, edge_index, y, train_mask, test_mask,
        node_image_counts y edge_stats (dict con conteos por tipo de arista).
    """

    # Normalise deprecated alias so all downstream logic uses the canonical name.
    if connectivity_mode == "sparse":
        import warnings
        warnings.warn(
            "'sparse' ha sido renombrado a 'half_directed' para reflejar mejor "
            "su semantica (train→train unidireccional + train→test). "
            "Por favor actualiza tu codigo. 'sparse' se eliminara en una version futura.",
            DeprecationWarning,
            stacklevel=2,
        )
        connectivity_mode = "half_directed"

    if connectivity_mode not in _VALID_MODES:
        raise ValueError(
            f"connectivity_mode debe ser uno de {_VALID_MODES}, "
            f"recibido: {repr(connectivity_mode)}"
        )

    _validate_inputs(train_features, train_labels, test_features, test_labels)

    x = torch.cat([train_features, test_features], dim=0).float()
    y = torch.cat([train_labels, test_labels], dim=0).long()

    node_image_counts = _build_node_image_counts(
        train_features=train_features,
        test_features=test_features,
        train_node_image_counts=train_node_image_counts,
        test_node_image_counts=test_node_image_counts,
    )

    num_train = train_features.shape[0]
    num_test = test_features.shape[0]
    num_nodes = x.shape[0]

    edge_index, edge_weight, edge_stats = _build_edge_index(
        train_features=train_features,
        test_features=test_features,
        k_neighbors=k_neighbors,
        connectivity_mode=connectivity_mode,
        knn_metric=knn_metric,
        use_mutual_knn=use_mutual_knn,
        edge_weight_sigma=edge_weight_sigma,
    )

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[:num_train] = True
    test_mask[num_train : num_train + num_test] = True

    return Data(
        x=x,
        edge_index=edge_index,
        edge_weight=edge_weight,        # (E,) float or None
        y=y,
        train_mask=train_mask,
        test_mask=test_mask,
        node_image_counts=node_image_counts,
        edge_stats=edge_stats,
        connectivity_mode=connectivity_mode,
        k_neighbors=k_neighbors,
        knn_metric=knn_metric,
        use_mutual_knn=use_mutual_knn,
        edge_weight_sigma=edge_weight_sigma,
    )


# ---------------------------------------------------------------------------
# Edge construction
# ---------------------------------------------------------------------------

def _build_edge_index(
    train_features: torch.Tensor,
    test_features: torch.Tensor,
    k_neighbors: int,
    connectivity_mode: str,
    knn_metric: str = "cosine",
    use_mutual_knn: bool = True,
    edge_weight_sigma: float | None = 0.5,
) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, int]]:
    """Construye edge_index (y opcionalmente edge_weight) segun el modo.

    Mejoras respecto a KNN basico:

    Mutual KNN (use_mutual_knn=True):
      La arista i↔j solo existe si j ∈ KNN(i) Y i ∈ KNN(j).
      Elimina aristas asimetricas ruidosas: si solo una direccion ve al
      otro como vecino, la similitud es probablemente espuria. Los GNNs
      son muy sensibles a aristas ruidosas en grafos dispersos.
      Efecto tipico: reduce aristas ~40-60%, mejora SNR del grafo.

    Pesos Gaussianos (edge_weight_sigma != None):
      w_ij = exp(-d_ij² / sigma²)
      Aristas entre nodos muy similares tienen peso ~1; aristas entre
      nodos lejanos tienen peso ~0. Permite al GNN ponderar la agregacion
      segun calidad de la similitud, no solo presencia/ausencia.
      sigma = 0.5 es un buen punto de partida para features L2-normalizadas.

    Implementacion matricial: O(N²) memoria, sin loops Python.

    Returns:
        edge_index:  (2, E) long tensor.
        edge_weight: (E,) float tensor o None si edge_weight_sigma is None.
        edge_stats:  conteos por tipo de arista.
    """

    if k_neighbors <= 0:
        raise ValueError(f"k_neighbors debe ser > 0, recibido: {k_neighbors}")

    num_train = train_features.shape[0]
    num_test = test_features.shape[0]

    if num_train < 2:
        raise ValueError("Se necesitan al menos 2 nodos de entrenamiento.")

    train_np = train_features.detach().cpu().numpy().astype(np.float32)
    effective_k_train = min(k_neighbors, num_train - 1)

    # KNN con distancias para calcular pesos de arista.
    knn = NearestNeighbors(n_neighbors=effective_k_train + 1, metric=knn_metric)
    knn.fit(train_np)
    train_dists, train_neighbors = knn.kneighbors(train_np, return_distance=True)
    # Columna 0 es el nodo propio (dist=0). Excluir.
    train_dists    = train_dists[:, 1:]        # (N_train, k)
    train_neighbors = train_neighbors[:, 1:]   # (N_train, k)

    # --- Matriz de adyacencia train-train ---
    row_indices = np.arange(num_train)
    adj = np.zeros((num_train, num_train), dtype=np.bool_)
    adj[row_indices[:, None], train_neighbors] = True

    # Mutual KNN: conservar solo aristas donde la similitud es reciproca.
    # i↔j existe solo si j∈KNN(i) AND i∈KNN(j).
    # Elimina vecindades asimetricas ruidosas que degradan la señal del GNN.
    if use_mutual_knn:
        adj = adj & adj.T

    if connectivity_mode == "half_directed":
        # Preserve every neighbour pair found in either direction, but emit
        # exactly one canonical edge per unordered pair.
        adj = np.triu(adj | adj.T, k=1)
    else:
        # directed / undirected: bidireccional en train-train
        adj = adj | adj.T

    tt_src, tt_dst = np.nonzero(adj)
    num_train_train_edges = tt_src.shape[0]

    # Pesos de arista Gaussianos: w = exp(-d² / sigma²).
    # Construimos una matriz de distancias para los pares que quedaron en adj.
    if edge_weight_sigma is not None:
        dist_matrix = np.full((num_train, num_train), np.inf, dtype=np.float32)
        dist_matrix[row_indices[:, None], train_neighbors] = train_dists
        # Simetrizar: usar la distancia minima en cada par.
        dist_matrix = np.minimum(dist_matrix, dist_matrix.T)
        tt_dists = dist_matrix[tt_src, tt_dst]
        tt_weights = np.exp(-(tt_dists ** 2) / (edge_weight_sigma ** 2)).astype(np.float32)
    else:
        tt_weights = None

    # --- Aristas train-test ---
    # PyG message passing uses source→target. Test nodes must therefore be
    # targets of their nearest train neighbours.
    if num_test > 0:
        test_np = test_features.detach().cpu().numpy().astype(np.float32)
        effective_k_test = min(k_neighbors, num_train)

        knn_test = NearestNeighbors(n_neighbors=effective_k_test, metric=knn_metric)
        knn_test.fit(train_np)
        test_dists_raw, test_neighbors = knn_test.kneighbors(test_np, return_distance=True)

        test_global_ids = np.arange(num_test) + num_train
        forward_src = test_neighbors.ravel()
        forward_dst = np.repeat(
            test_global_ids[:, None], effective_k_test, axis=1
        ).ravel()
        te_src = forward_src
        te_dst = forward_dst
        te_dists = test_dists_raw.ravel()

        if connectivity_mode == "undirected":
            te_src = np.concatenate([forward_src, forward_dst])
            te_dst = np.concatenate([forward_dst, forward_src])
            te_dists = np.concatenate([te_dists, te_dists])

        if edge_weight_sigma is not None:
            te_weights = np.exp(-(te_dists ** 2) / (edge_weight_sigma ** 2)).astype(np.float32)
        else:
            te_weights = None
    else:
        te_src = np.empty(0, dtype=np.int64)
        te_dst = np.empty(0, dtype=np.int64)
        te_weights = None

    num_train_test_edges = te_src.shape[0]

    all_src = np.concatenate([tt_src, te_src]).astype(np.int64)
    all_dst = np.concatenate([tt_dst, te_dst]).astype(np.int64)

    edge_index = torch.from_numpy(np.stack([all_src, all_dst])).contiguous()

    if edge_weight_sigma is not None and tt_weights is not None:
        all_weights = np.concatenate([
            tt_weights,
            te_weights if te_weights is not None else np.ones(len(te_src), dtype=np.float32),
        ])
        edge_weight = torch.from_numpy(all_weights)
    else:
        edge_weight = None

    edge_stats = {
        "train_train": num_train_train_edges,
        "train_test": num_train_test_edges,
        "total": int(all_src.shape[0]),
        "mutual_knn": use_mutual_knn,
        "weighted": edge_weight is not None,
    }
    return edge_index, edge_weight, edge_stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_node_image_counts(
    train_features: torch.Tensor,
    test_features: torch.Tensor,
    train_node_image_counts: torch.Tensor | None,
    test_node_image_counts: torch.Tensor | None,
) -> torch.Tensor:
    if train_node_image_counts is None:
        train_node_image_counts = torch.ones(train_features.shape[0], dtype=torch.long)
    if test_node_image_counts is None:
        test_node_image_counts = torch.ones(test_features.shape[0], dtype=torch.long)
    if train_node_image_counts.shape[0] != train_features.shape[0]:
        raise ValueError("train_node_image_counts no coincide con train_features.")
    if test_node_image_counts.shape[0] != test_features.shape[0]:
        raise ValueError("test_node_image_counts no coincide con test_features.")
    return torch.cat([train_node_image_counts, test_node_image_counts], dim=0).long()


def _validate_inputs(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
) -> None:
    if train_features.ndim != 2 or test_features.ndim != 2:
        raise ValueError("Las features deben tener forma (num_samples, feature_dim).")
    if train_features.shape[1] != test_features.shape[1]:
        raise ValueError("train_features y test_features deben tener la misma dimension.")
    if train_features.shape[0] != train_labels.shape[0]:
        raise ValueError("train_features y train_labels tienen tamanos distintos.")
    if test_features.shape[0] != test_labels.shape[0]:
        raise ValueError("test_features y test_labels tienen tamanos distintos.")
    if train_labels.ndim != 1 or test_labels.ndim != 1:
        raise ValueError("Las etiquetas deben tener forma (num_samples,).")
