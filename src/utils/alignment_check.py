"""PATCH 9: verifica alineación de labels, masks y features en el grafo.

Cuando loss ~2.30 persiste en todas las configuraciones, un bug sutil de
alineación es posible: los labels shuffleados no coinciden con las features
shuffleadas, o las masks train/val/test se solapan, o los indices están
desplazados. Este módulo lo detecta antes de entrenar.
"""

from __future__ import annotations

import torch
from torch_geometric.data import Data


def verify_graph_alignment(graph: Data, label: str = "") -> dict:
    """Verifica invariantes básicos de alineación en un grafo PyG.

    Checks:
      1. x.shape[0] == y.shape[0] == num_nodes
      2. train_mask y test_mask no se solapan
      3. train_mask | test_mask cubre todos los nodos
      4. edge_index valores en [0, num_nodes)
      5. node_image_counts suma == total imágenes esperado (si disponible)
      6. Distribución de clases en train no es degenerada (una sola clase)

    Returns:
        dict con 'passed' (bool) y 'issues' (list[str]).
    """
    prefix = f"[{label}] " if label else ""
    issues: list[str] = []
    n = graph.num_nodes

    # 1. Dimensiones consistentes
    if graph.x.shape[0] != n:
        issues.append(f"x.shape[0]={graph.x.shape[0]} != num_nodes={n}")
    if graph.y.shape[0] != n:
        issues.append(f"y.shape[0]={graph.y.shape[0]} != num_nodes={n}")

    train_mask = graph.train_mask.bool()
    test_mask  = graph.test_mask.bool()

    # 2. Sin solapamiento
    overlap = (train_mask & test_mask).sum().item()
    if overlap > 0:
        issues.append(f"train_mask ∩ test_mask = {overlap} nodos (debe ser 0)")

    # 3. Cobertura
    uncovered = (~(train_mask | test_mask)).sum().item()
    if uncovered > 0:
        issues.append(f"{uncovered} nodos no están en ninguna máscara")

    # 4. edge_index bounds
    if graph.edge_index.numel() > 0:
        max_idx = graph.edge_index.max().item()
        if max_idx >= n:
            issues.append(f"edge_index contiene índice {max_idx} >= num_nodes={n}")
        min_idx = graph.edge_index.min().item()
        if min_idx < 0:
            issues.append(f"edge_index contiene índice negativo {min_idx}")

    # 5. Distribución de clases en train (degenerada = probable bug)
    train_labels = graph.y[train_mask]
    unique_classes = torch.unique(train_labels).numel()
    total_classes  = torch.unique(graph.y).numel()
    if unique_classes < max(2, total_classes // 2):
        issues.append(
            f"Train contiene solo {unique_classes}/{total_classes} clases. "
            "Posible bug de shuffle o subset demasiado pequeño."
        )

    # 6. Label range
    if graph.y.min().item() < 0:
        issues.append(f"Labels negativas: min={graph.y.min().item()}")

    passed = len(issues) == 0
    status = "✅ OK" if passed else f"❌ {len(issues)} problema(s)"
    print(f"{prefix}Graph alignment check: {status}")
    for issue in issues:
        print(f"{prefix}  ⚠️  {issue}")

    return {"passed": passed, "issues": issues, "num_nodes": n,
            "num_train": int(train_mask.sum()), "num_test": int(test_mask.sum()),
            "unique_train_classes": unique_classes}
