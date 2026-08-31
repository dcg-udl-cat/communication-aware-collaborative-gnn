"""KNN classifier como sanity baseline sobre features puras.

Por que este baseline importa:
  Si el KNN classifier (sin ningun grafo) supera al GNN tiny, significa que
  las features ya contienen toda la informacion necesaria y el grafo no añade
  valor. En ese caso el problema no es el GNN — son las features o la tarea.

  Si el GNN supera al KNN: el paso de mensajes esta agregando informacion
  util de los vecinos, lo que justifica la arquitectura de grafo.

  Si el GNN es igual al KNN: los vecinos del grafo son los mismos que los
  k vecinos del KNN → el GNN esta rehaciendo lo que el KNN ya hace.

Uso:
  baseline = KNNClassifierBaseline(k=5)
  baseline.fit(train_features, train_labels)
  acc = baseline.score(test_features, test_labels)
  result = baseline.evaluate(train_features, train_labels, test_features, test_labels)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from sklearn.neighbors import KNeighborsClassifier


@dataclass
class KNNBaselineResult:
    k: int
    metric: str
    train_accuracy: float
    test_accuracy: float
    num_train_nodes: int
    num_test_nodes: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": "knn_baseline",
            "k": self.k,
            "metric": self.metric,
            "train_accuracy": self.train_accuracy,
            "test_accuracy": self.test_accuracy,
            "num_train_nodes": self.num_train_nodes,
            "num_test_nodes": self.num_test_nodes,
        }

    def summary(self) -> str:
        return (
            f"KNN(k={self.k}, metric={self.metric}): "
            f"train={self.train_accuracy:.3f}  test={self.test_accuracy:.3f}  "
            f"[{self.num_train_nodes} train nodes, {self.num_test_nodes} test nodes]"
        )


class KNNClassifierBaseline:
    """Sklearn KNN classifier wrapeado para integrarse con el pipeline."""

    uses_graph = False    # para compatibilidad con model_forward checks

    def __init__(self, k: int = 5, metric: str = "cosine") -> None:
        if k <= 0:
            raise ValueError(f"k debe ser > 0, recibido: {k}")
        self.k = k
        self.metric = metric
        self._clf: KNeighborsClassifier | None = None

    # ------------------------------------------------------------------
    # nn.Module compatibility shims
    # These allow KNNClassifierBaseline to be passed anywhere an nn.Module
    # is expected without crashing on .to(), .train(), .eval(), .parameters().
    # ------------------------------------------------------------------

    def to(self, *args, **kwargs) -> "KNNClassifierBaseline":
        """No-op: sklearn models are CPU-only and device-agnostic."""
        return self

    def train(self, mode: bool = True) -> "KNNClassifierBaseline":
        """No-op: sklearn models have no train/eval distinction."""
        return self

    def eval(self) -> "KNNClassifierBaseline":
        """No-op."""
        return self

    def parameters(self):
        """No trainable parameters — returns empty iterator."""
        return iter([])

    def fit(self, features: torch.Tensor, labels: torch.Tensor) -> "KNNClassifierBaseline":
        """Ajusta el clasificador KNN sobre features de entrenamiento."""
        self._clf = KNeighborsClassifier(n_neighbors=self.k, metric=self.metric)
        self._clf.fit(
            features.detach().cpu().numpy(),
            labels.detach().cpu().numpy(),
        )
        return self

    def score(self, features: torch.Tensor, labels: torch.Tensor) -> float:
        """Accuracy sobre el conjunto dado. Requiere llamar fit() primero."""
        if self._clf is None:
            raise RuntimeError("Llama a fit() antes de score().")
        return float(self._clf.score(
            features.detach().cpu().numpy(),
            labels.detach().cpu().numpy(),
        ))

    def evaluate(
        self,
        train_features: torch.Tensor,
        train_labels: torch.Tensor,
        test_features: torch.Tensor,
        test_labels: torch.Tensor,
    ) -> KNNBaselineResult:
        """Fit sobre train y evalua sobre train y test. Devuelve KNNBaselineResult."""
        self.fit(train_features, train_labels)
        return KNNBaselineResult(
            k=self.k,
            metric=self.metric,
            train_accuracy=self.score(train_features, train_labels),
            test_accuracy=self.score(test_features, test_labels),
            num_train_nodes=int(train_features.shape[0]),
            num_test_nodes=int(test_features.shape[0]),
        )
