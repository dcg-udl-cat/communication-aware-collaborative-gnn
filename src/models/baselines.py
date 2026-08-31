"""Modelos baseline para comparar contra las GNN."""

import torch
from torch import nn
import torch.nn.functional as functional

from config import DROPOUT, INPUT_DIM, MLP_HIDDEN_DIM, NUM_CLASSES, TINY_GNN_HIDDEN_DIM


class RandomBaseline(nn.Module):
    """Baseline de prediccion constante (logits uniformes cero).

    Note: despite the name, this is NOT a stochastic random classifier.
    It outputs constant zero logits for every node on every call, which
    produces a uniform class distribution after softmax. This is a
    deterministic sanity floor: any trained model must beat it.

    If you want a truly stochastic baseline (different random prediction
    each call), use sklearn.dummy.DummyClassifier with strategy="uniform".
    That is more honest but less stable as a reference line in plots.
    """

    uses_graph = False

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.register_buffer("fixed_bias", torch.zeros(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fixed_bias.unsqueeze(0).expand(x.shape[0], -1)


class SimpleMLPBaseline(nn.Module):
    """MLP de dos capas ocultas con BatchNorm para clasificar features de nodo.

    Arquitectura (patch 6):
      Linear(input -> hidden)  -> BN -> ReLU -> Dropout
      Linear(hidden -> hidden) -> BN -> ReLU -> Dropout
      Linear(hidden -> classes)

    Por que dos capas y BatchNorm:
      Una capa oculta de 64-D tiene ~8k parámetros, insuficiente para
      aprender representaciones útiles de features PCA-128. Con dos capas
      de 256-D el MLP tiene ~100k parámetros y puede aprender transformaciones
      no-lineales complejas.

      BatchNorm estabiliza la distribución de activaciones capa a capa,
      lo que acelera la convergencia y reduce la sensibilidad al LR.
      Es especialmente útil aquí porque las features de entrada ya están
      L2-normalizadas pero la distribución interna cambia durante el entreno.

    Target: >40% test accuracy en CIFAR-10 con PCA-128 features.
    Si no lo alcanza, el pipeline de features está roto (patch 4).
    """

    uses_graph = False

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dim: int = MLP_HIDDEN_DIM,
        num_classes: int = NUM_CLASSES,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.dropout = dropout

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = functional.dropout(functional.relu(self.bn1(self.fc1(x))),
                               p=self.dropout, training=self.training)
        x = functional.dropout(functional.relu(self.bn2(self.fc2(x))),
                               p=self.dropout, training=self.training)
        return self.fc3(x)
