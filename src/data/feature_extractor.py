"""Extraccion de features para imagenes y construccion de nodos de grafo.

Responsabilidades separadas en tres funciones:
  - extract_features:     imagenes  -> vectores de features (flatten + L2)
  - build_node_labels:    etiquetas -> etiqueta de nodo por mayoria de votos
  - build_node_features_from_images: orquesta ambas y produce nodos completos

Esta separacion facilita sustituir cualquier paso de forma independiente:
por ejemplo, reemplazar flatten por PCA o una embedding compacta sin tocar
la logica de agrupacion ni la de etiquetado.
"""

import torch
import torch.nn.functional as functional

from config import IMAGES_PER_NODE, NORMALIZE_FEATURES


# ---------------------------------------------------------------------------
# 1. Extraccion de features (un vector por imagen)
# ---------------------------------------------------------------------------

def extract_features(
    images: torch.Tensor,
    normalize: bool = NORMALIZE_FEATURES,
) -> torch.Tensor:
    """Convierte imagenes (N, C, H, W) en vectores (N, C*H*W).

    Usa view en lugar de reshape: fuerza un error explicito si el tensor
    no es contiguo, en vez de crear una copia silenciosa.

    Args:
        images:    Tensor de forma (N, C, H, W). Debe ser contiguo.
        normalize: Si True, aplica normalizacion L2 por fila.

    Returns:
        Tensor de forma (N, C*H*W), dtype float32.

    NOTE (siguiente iteracion):
        Sustituir el flatten por PCA o una embedding compacta de 64 dims
        reduciria la dimension de 3072 a ~64, aceleraria el KNN y bajaria
        el footprint del modelo, acercandose mas al escenario edge real.
    """

    if images.ndim != 4:
        raise ValueError(
            f"images debe tener forma (N, C, H, W), recibido: {tuple(images.shape)}"
        )
    if images.shape[0] == 0:
        raise ValueError("images no puede estar vacio.")

    features = images.float().contiguous().view(images.shape[0], -1)

    # L2-normalize BEFORE returning (and before any downstream PCA).
    # Raw pixel vectors span wildly different magnitudes across images
    # (dark vs bright, small object vs full-frame). Without normalization,
    # PCA principal components are biased toward high-variance brightness
    # dimensions rather than class-discriminative structure.
    # Normalizing first ensures PCA captures directional (shape/color pattern)
    # variation rather than intensity variation.
    features = functional.normalize(features, p=2, dim=1)

    # Secondary normalization flag kept for API compatibility; with the
    # pre-normalization above, this is a no-op for already-unit-norm vectors.
    if normalize:
        features = functional.normalize(features, p=2, dim=1)

    return features


# ---------------------------------------------------------------------------
# 2. Etiquetado de nodos por mayoria de votos (determinista)
# ---------------------------------------------------------------------------

def build_node_labels(group_labels: torch.Tensor) -> int:
    """Devuelve la etiqueta mayoritaria de un grupo de imagenes.

    En caso de empate entre varias clases, elige la de menor indice.
    Esto es determinista dado el mismo grupo, sin depender del estado
    interno de torch.mode (que puede variar entre versiones de PyTorch
    en situaciones de empate).

    Args:
        group_labels: Tensor 1-D de etiquetas enteras.

    Returns:
        Entero con la etiqueta mayoritaria.
    """

    if group_labels.numel() == 0:
        raise ValueError("group_labels no puede estar vacio.")

    values, counts = torch.unique(group_labels, return_counts=True)
    return int(values[counts.argmax()].item())


# ---------------------------------------------------------------------------
# 3. Construccion de nodos: agrupa imagenes y produce features + etiquetas
# ---------------------------------------------------------------------------

def build_node_features_from_images(
    images: torch.Tensor,
    labels: torch.Tensor,
    images_per_node: int = IMAGES_PER_NODE,
    normalize: bool = NORMALIZE_FEATURES,
    shuffle: bool = True,
    generator: torch.Generator | None = None,
    use_mean_std: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Agrupa imagenes de forma aleatoria (sin supervision) y crea nodos de grafo.

    Cada nodo agrega `images_per_node` imagenes. Las imagenes se barajan
    internamente antes de agrupar para evitar que el orden del dataset
    (que en CIFAR suele ser aproximadamente por clase) introduzca sesgo
    silencioso en la composicion de los nodos.

    Las etiquetas NO se usan para decidir que imagenes se agrupan juntas.
    Esto refleja el escenario real de borde, donde las etiquetas no estan
    disponibles en tiempo de construccion del grafo.

    La etiqueta de cada nodo se asigna por mayoria de votos entre las
    imagenes del grupo (determinista en caso de empate: gana la clase
    de menor indice).

    Args:
        images:          Tensor (N, C, H, W).
        labels:          Tensor (N,) de etiquetas enteras.
        images_per_node: Numero de imagenes a promediar por nodo.
        normalize:       Si True, aplica L2 a las features de nodo finales.
        shuffle:         Si True (defecto), baraja las imagenes antes de
                         agrupar. Desactivar solo en tests que necesiten
                         orden determinista sin semilla.
        generator:       Generador de torch para reproducibilidad del shuffle.
        use_mean_std:    Si True y images_per_node > 1, concatena mean y std
                         de las features del grupo en lugar de solo la media.
                         Dobla la dimension de salida pero preserva variabilidad
                         intra-nodo. Util con ipn=5..10.
                         Si None, usa el estado global de torch.

    Returns:
        features:          Tensor (num_nodes, feature_dim) float32.
        node_labels:       Tensor (num_nodes,) long.
        node_image_counts: Tensor (num_nodes,) long con el numero real de
                           imagenes que componen cada nodo (el ultimo puede
                           ser menor que images_per_node).
    """

    _validate_grouping_inputs(images, labels, images_per_node)

    num_images = images.shape[0]

    if shuffle:
        perm = torch.randperm(num_images, generator=generator)
        images = images[perm]
        labels = labels[perm]

    image_features = extract_features(images, normalize=False)

    node_features: list[torch.Tensor] = []
    node_labels_list: list[int] = []
    node_image_counts: list[int] = []

    for start in range(0, num_images, images_per_node):
        end = min(start + images_per_node, num_images)

        group_features = image_features[start:end]
        group_labels = labels[start:end]

        if use_mean_std and group_features.shape[0] > 1:
            # mean + std: preserva mas informacion que solo la media.
            # El std captura la variabilidad intra-nodo (util con ipn > 1).
            mean_feat = group_features.mean(dim=0)
            std_feat  = group_features.std(dim=0)
            node_features.append(functional.normalize(
                torch.cat([mean_feat, std_feat], dim=0), p=2, dim=0
            ))
        else:
            node_features.append(group_features.mean(dim=0))

        node_labels_list.append(build_node_labels(group_labels))
        node_image_counts.append(end - start)

    features = torch.stack(node_features).float()
    if normalize:
        features = functional.normalize(features, p=2, dim=1)

    return (
        features,
        torch.tensor(node_labels_list, dtype=torch.long),
        torch.tensor(node_image_counts, dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# Validacion de entradas
# ---------------------------------------------------------------------------

def _validate_grouping_inputs(
    images: torch.Tensor,
    labels: torch.Tensor,
    images_per_node: int,
) -> None:
    if images_per_node <= 0:
        raise ValueError(f"images_per_node debe ser > 0, recibido: {images_per_node}")
    if labels.ndim != 1:
        raise ValueError(
            f"labels debe tener forma (N,), recibido: {tuple(labels.shape)}"
        )
    if images.shape[0] != labels.shape[0]:
        raise ValueError(
            f"images ({images.shape[0]}) y labels ({labels.shape[0]}) "
            "deben tener el mismo numero de muestras."
        )
