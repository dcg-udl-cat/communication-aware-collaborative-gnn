"""Compresion de features por PCA ajustado solo sobre datos de entrenamiento.

Por que PCA antes del grafo:
  - Reduce 3072 dims (flatten CIFAR) a ~64 dims compactos.
  - El KNN opera en espacio de menor dimension: menos ruido, mas rapido.
  - El modelo GNN recibe entradas mas pequenas: menos parametros,
    menor footprint de memoria, mas adecuado para despliegue en el borde.

Regla de no-leakage:
  - fit() se llama SOLO sobre train_features.
  - transform() se aplica por separado a train y test.
  - Ajustar sobre train+test filtraria informacion de test hacia el
    espacio de proyeccion, invalidando la evaluacion inductiva.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class PCACompressor:
    """Wrapper de sklearn.PCA que mantiene las garantias de no-leakage.

    Uso tipico:
        compressor = PCACompressor(n_components=64)
        train_features = compressor.fit_transform(train_features)
        test_features  = compressor.transform(test_features)

    El objeto es serializable con pickle para guardarlo junto al modelo
    y poder transformar nuevos datos en produccion sin reajustar.
    """

    def __init__(self, n_components: int) -> None:
        if n_components <= 0:
            raise ValueError(f"n_components debe ser > 0, recibido: {n_components}")
        self.n_components = n_components
        self._pca: PCA | None = None
        self._scaler: StandardScaler | None = None

    # ------------------------------------------------------------------
    # API principal
    # ------------------------------------------------------------------

    def fit_transform(self, train_features: torch.Tensor) -> torch.Tensor:
        """Ajusta PCA sobre train_features y devuelve la proyeccion.

        Args:
            train_features: Tensor (N_train, D) float32. Solo datos de train.

        Returns:
            Tensor (N_train, n_components) float32.
        """
        _validate_features(train_features, "train_features")

        effective_components = min(self.n_components, *train_features.shape)
        if effective_components < self.n_components:
            # Aviso: n_components se recorta al maximo posible.
            print(
                f"[PCACompressor] n_components={self.n_components} > "
                f"min(N={train_features.shape[0]}, D={train_features.shape[1]}). "
                f"Se usara n_components={effective_components}."
            )

        self._pca = PCA(n_components=effective_components, random_state=0)
        raw_np = train_features.detach().cpu().numpy().astype(np.float32)
        projected = self._pca.fit_transform(raw_np)
        # PATCH 2: standardize PCA output (zero-mean, unit-variance per component)
        # before L2-normalization. PCA components have different variances
        # (eigenvalues), so without standardization the first few components
        # dominate distances and gradients regardless of their class-discriminative
        # content. StandardScaler fitted on train only.
        self._scaler = StandardScaler()
        projected = self._scaler.fit_transform(projected)
        tensor = torch.tensor(projected, dtype=torch.float32)
        return F.normalize(tensor, p=2, dim=1)

    def transform(self, features: torch.Tensor) -> torch.Tensor:
        """Proyecta features usando el PCA ya ajustado.

        Solo puede llamarse despues de fit_transform.

        Args:
            features: Tensor (N, D) float32. Puede ser train, test o nuevos datos.

        Returns:
            Tensor (N, n_components) float32.

        Raises:
            RuntimeError: Si se llama antes de fit_transform.
        """
        if self._pca is None:
            raise RuntimeError(
                "PCACompressor.transform() llamado antes de fit_transform(). "
                "Llama primero a fit_transform(train_features)."
            )
        _validate_features(features, "features")

        projected = self._pca.transform(
            features.detach().cpu().numpy().astype(np.float32)
        )
        if self._scaler is not None:
            projected = self._scaler.transform(projected)
        tensor = torch.tensor(projected, dtype=torch.float32)
        return F.normalize(tensor, p=2, dim=1)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        return self._pca is not None and self._scaler is not None

    @property
    def explained_variance_ratio(self) -> torch.Tensor | None:
        """Fraccion de varianza explicada por cada componente."""
        if self._pca is None:
            return None
        return torch.tensor(self._pca.explained_variance_ratio_, dtype=torch.float32)

    @property
    def total_explained_variance(self) -> float | None:
        """Suma de varianza explicada (0-1). Util para elegir n_components."""
        if self._pca is None:
            return None
        return float(self._pca.explained_variance_ratio_.sum())

    @property
    def output_dim(self) -> int | None:
        """Dimension de salida efectiva tras el ajuste."""
        if self._pca is None:
            return None
        return self._pca.n_components_


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_features(features: torch.Tensor, name: str) -> None:
    if features.ndim != 2:
        raise ValueError(
            f"{name} debe tener forma (N, D), recibido: {tuple(features.shape)}"
        )
    if features.shape[0] == 0:
        raise ValueError(f"{name} no puede estar vacio.")
