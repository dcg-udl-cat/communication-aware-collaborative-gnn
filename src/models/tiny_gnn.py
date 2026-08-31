"""Tiny GNN para clasificacion de nodos en el borde.

Arquitectura: Linear -> GATConv (con residual) -> Linear + Dropout.

Por que GAT en lugar de SAGEConv/GCNConv:
  El grafo KNN construido sobre features visuales es RUIDOSO incluso con
  mutual KNN. Algunos vecinos son semanticamente irrelevantes. GAT aprende
  a ignorarlos: los coeficientes de atencion pesan cada vecino de forma
  adaptativa en lugar de promediar uniformemente. Esto es critico cuando
  la SNR del grafo es baja.

Por que residual:
  Sin residual, en grafos pequenos y dispersos las capas GNN tienden a
  over-smoothing: todos los nodos convergen a representaciones similares
  independientemente de su clase. La conexion residual (x = x + conv(x))
  preserva la informacion original del nodo mientras añade la señal del grafo.

Arquitectura detallada:
  lin_in:   Linear(input_dim -> hidden_dim)     proyeccion inicial
  conv:     GATConv(hidden_dim -> hidden_dim, heads=4, concat=False)
  residual: x = x + conv(x, edge_index)         preserva info original
  lin_out:  Linear(hidden_dim -> num_classes)   clasificador

Footprint con input=512, hidden=128, classes=10, heads=4:
  lin_in    = 512*128 + 128       =  65 664
  GATConv   ~ 128*128*4 + 128*4  =  66 048  (aprox, sin bias att)
  lin_out   = 128*10 + 10        =   1 290
  Total     ≈ 133 000 parametros  (sigue siendo "tiny" vs ResNet18 ~11M)

Con encoder PCA (no ResNet):
  input=128 → Total ≈ 35 000 parametros
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from config import DROPOUT, INPUT_DIM, NUM_CLASSES, TINY_GNN_HIDDEN_DIM


class TinyGNN(nn.Module):
    """Linear -> GAT (con residual) -> Linear + Dropout.

    Acepta edge_weight opcional: si el grafo tiene pesos de arista,
    el GATConv los usa como bias de atencion (atencion ponderada).
    """

    uses_graph = True

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dim: int = TINY_GNN_HIDDEN_DIM,
        num_classes: int = NUM_CLASSES,
        dropout: float = DROPOUT,
        gat_heads: int = 4,
    ) -> None:
        super().__init__()
        self.dropout_p = dropout

        # Proyeccion inicial: lleva el input a hidden_dim antes del grafo.
        self.lin_in = nn.Linear(input_dim, hidden_dim)

        # GAT con multi-head attention, concat=False → salida = hidden_dim.
        # edge_dim=1 permite pasar edge_weight como feature de arista.
        self.conv = GATConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=gat_heads,
            concat=False,
            dropout=dropout,
            edge_dim=1,             # para edge_weight como feature
        )

        # Clasificador final.
        self.lin_out = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Devuelve logits por nodo.

        Args:
            x:           (N, input_dim) features de nodo.
            edge_index:  (2, E) aristas.
            edge_weight: (E,) pesos opcionales. Si se proporcionan,
                         se pasan como edge_attr al GATConv.
        """
        x = F.relu(self.lin_in(x))
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        # Preparar edge_attr para GATConv si hay pesos.
        edge_attr = edge_weight.unsqueeze(1) if edge_weight is not None else None

        # Conexion residual: preserva informacion del nodo original.
        h = F.relu(self.conv(x, edge_index, edge_attr=edge_attr))
        x = x + h                                  # residual

        return self.lin_out(x)
