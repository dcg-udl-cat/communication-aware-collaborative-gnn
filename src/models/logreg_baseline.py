"""Logistic Regression baseline sobre features de nodo.

Por que este baseline es necesario (patch 7):
  KNN dice "hay estructura en el espacio de features".
  Logistic Regression dice "esa estructura es LINEALMENTE accesible".

  Si LR > GNN: las features son buenas pero el GNN no está aprendiendo.
  Si LR ≈ KNN: la estructura es local (vecindades), no lineal global.
  Si GNN > LR: el GNN está captando estructura no-lineal que LR no ve,
               lo que justifica la arquitectura de grafo.

  Con las features actuales (PCA+L2), LR debería dar 25-35% en CIFAR-10
  si el pipeline es correcto. Si da ~10% (random), hay un bug de alineación.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any

import torch
from sklearn.linear_model import LogisticRegression


@dataclass
class LogRegResult:
    test_accuracy: float
    train_accuracy: float
    num_train_nodes: int
    num_test_nodes: int
    max_iter: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": "logreg_baseline",
            "test_accuracy": self.test_accuracy,
            "train_accuracy": self.train_accuracy,
            "num_train_nodes": self.num_train_nodes,
            "num_test_nodes": self.num_test_nodes,
            "max_iter": self.max_iter,
        }

    def summary(self) -> str:
        return (
            f"LogReg: train={self.train_accuracy:.3f}  test={self.test_accuracy:.3f}"
            f"  [{self.num_train_nodes} train, {self.num_test_nodes} test]"
        )


class LogRegClassifierBaseline:
    """Sklearn LogisticRegression con la misma interfaz que KNNClassifierBaseline."""

    uses_graph = False

    def __init__(self, max_iter: int = 1000, C: float = 1.0) -> None:
        self.max_iter = max_iter
        self.C = C
        self._clf: LogisticRegression | None = None

    # nn.Module compatibility shims
    def to(self, *args, **kwargs) -> "LogRegClassifierBaseline": return self
    def train(self, mode: bool = True) -> "LogRegClassifierBaseline": return self
    def eval(self) -> "LogRegClassifierBaseline": return self
    def parameters(self): return iter([])

    def fit(self, features: torch.Tensor, labels: torch.Tensor) -> "LogRegClassifierBaseline":
        self._clf = LogisticRegression(
            max_iter=self.max_iter, C=self.C,
            solver="lbfgs",
            # multi_class removed in sklearn 1.5; lbfgs handles
            # multiclass natively via the multinomial objective.
        )
        self._clf.fit(
            features.detach().cpu().numpy(),
            labels.detach().cpu().numpy(),
        )
        return self

    def score(self, features: torch.Tensor, labels: torch.Tensor) -> float:
        if self._clf is None:
            raise RuntimeError("Llama a fit() antes de score().")
        return float(self._clf.score(
            features.detach().cpu().numpy(),
            labels.detach().cpu().numpy(),
        ))

    def evaluate(
        self,
        train_features: torch.Tensor, train_labels: torch.Tensor,
        test_features: torch.Tensor,  test_labels: torch.Tensor,
    ) -> LogRegResult:
        self.fit(train_features, train_labels)
        return LogRegResult(
            test_accuracy=self.score(test_features, test_labels),
            train_accuracy=self.score(train_features, train_labels),
            num_train_nodes=int(train_features.shape[0]),
            num_test_nodes=int(test_features.shape[0]),
            max_iter=self.max_iter,
        )
