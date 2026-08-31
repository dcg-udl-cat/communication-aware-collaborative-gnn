"""Tests para PCACompressor: no-leakage, dimensiones, reproducibilidad."""

import torch
import pytest


class TestPCACompressorNoLeakage:
    """Garantias de que el PCA se ajusta SOLO sobre train."""

    def _make_features(self, n, d):
        return torch.rand(n, d)

    def test_transform_before_fit_raises(self):
        from src.data.pca_compressor import PCACompressor
        comp = PCACompressor(n_components=8)
        with pytest.raises(RuntimeError, match="fit_transform"):
            comp.transform(self._make_features(5, 32))

    def test_fit_only_on_train(self):
        """Verificacion del protocolo: fit sobre train, transform sobre test por separado.

        Si aplicaramos fit_transform sobre train+test concatenados, el PCA veria
        la distribucion de test y filtraria informacion hacia el espacio de
        proyeccion. El test verifica que ambos splits producen dimensiones
        correctas cuando se procesan de forma independiente.
        """
        from src.data.pca_compressor import PCACompressor
        n_train, n_test, d, k = 50, 20, 64, 8
        train_f = self._make_features(n_train, d)
        test_f = self._make_features(n_test, d)

        comp = PCACompressor(n_components=k)
        train_out = comp.fit_transform(train_f)
        test_out = comp.transform(test_f)

        assert train_out.shape == (n_train, k)
        assert test_out.shape == (n_test, k)

    def test_test_transform_does_not_change_train_projection(self):
        """Transformar test no debe alterar la proyeccion de train."""
        from src.data.pca_compressor import PCACompressor
        train_f = self._make_features(30, 32)
        test_f = self._make_features(10, 32)

        comp = PCACompressor(n_components=4)
        train_out_before = comp.fit_transform(train_f)
        _ = comp.transform(test_f)
        train_out_after = comp.transform(train_f)

        assert torch.allclose(train_out_before, train_out_after, atol=1e-5), \
            "Transformar test no debe modificar la proyeccion de train"


class TestPCACompressorDimensions:
    def test_output_dim_matches_n_components(self):
        from src.data.pca_compressor import PCACompressor
        f = torch.rand(40, 128)
        comp = PCACompressor(n_components=16)
        out = comp.fit_transform(f)
        assert out.shape == (40, 16)
        assert comp.output_dim == 16

    def test_n_components_clamped_to_min_n_d(self):
        """Si n_components > min(N, D), se recorta automaticamente sin error."""
        from src.data.pca_compressor import PCACompressor
        # 10 muestras, 8 dims: no pueden existir mas de 8 componentes utiles
        f = torch.rand(10, 8)
        comp = PCACompressor(n_components=100)
        out = comp.fit_transform(f)
        assert out.shape[1] <= min(10, 8)

    def test_zero_components_raises(self):
        from src.data.pca_compressor import PCACompressor
        with pytest.raises(ValueError, match="n_components"):
            PCACompressor(n_components=0)

    def test_output_dtype_float32(self):
        from src.data.pca_compressor import PCACompressor
        f = torch.rand(20, 32)
        comp = PCACompressor(n_components=4)
        out = comp.fit_transform(f)
        assert out.dtype == torch.float32

    def test_wrong_ndim_raises(self):
        from src.data.pca_compressor import PCACompressor
        comp = PCACompressor(n_components=4)
        with pytest.raises(ValueError, match="forma"):
            comp.fit_transform(torch.rand(10))

    def test_empty_input_raises(self):
        from src.data.pca_compressor import PCACompressor
        comp = PCACompressor(n_components=4)
        with pytest.raises(ValueError, match="vacio"):
            comp.fit_transform(torch.rand(0, 32))


class TestPCACompressorStats:
    def test_explained_variance_between_0_and_1(self):
        from src.data.pca_compressor import PCACompressor
        f = torch.rand(50, 64)
        comp = PCACompressor(n_components=8)
        comp.fit_transform(f)
        assert comp.total_explained_variance is not None
        assert 0.0 <= comp.total_explained_variance <= 1.0

    def test_more_components_more_variance(self):
        """Mas componentes deben explicar igual o mas varianza."""
        from src.data.pca_compressor import PCACompressor
        torch.manual_seed(0)
        f = torch.rand(60, 64)
        comp4 = PCACompressor(n_components=4)
        comp16 = PCACompressor(n_components=16)
        comp4.fit_transform(f)
        comp16.fit_transform(f)
        assert comp16.total_explained_variance >= comp4.total_explained_variance

    def test_stats_none_before_fit(self):
        from src.data.pca_compressor import PCACompressor
        comp = PCACompressor(n_components=8)
        assert not comp.is_fitted
        assert comp.total_explained_variance is None
        assert comp.output_dim is None
        assert comp.explained_variance_ratio is None

    def test_is_fitted_after_fit(self):
        from src.data.pca_compressor import PCACompressor
        comp = PCACompressor(n_components=4)
        comp.fit_transform(torch.rand(20, 16))
        assert comp.is_fitted


class TestPCAInPipeline:
    """Verifica que PCA se integra correctamente en _build_graph."""

    def _make_cifar_data(self):
        """CifarData sintetico para test de integracion."""
        from data.cifar_loader import CifarData, CifarSplit
        return CifarData(
            train=CifarSplit(
                images=torch.rand(40, 3, 32, 32),
                labels=torch.randint(0, 10, (40,)),
            ),
            test=CifarSplit(
                images=torch.rand(10, 3, 32, 32),
                labels=torch.randint(0, 10, (10,)),
            ),
            class_names=tuple(str(i) for i in range(10)),
        )

    def test_pca_reduces_feature_dim(self, monkeypatch):
        """Con USE_PCA=True, el grafo debe tener features de PCA_COMPONENTS dims."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        import config
        monkeypatch.setattr(config, "USE_PCA", True)
        monkeypatch.setattr(config, "PCA_COMPONENTS", 8)

        import importlib
        import main
        importlib.reload(main)

        cifar_data = self._make_cifar_data()
        graph, compressor = main._build_graph(
            cifar_data=cifar_data,
            k_neighbors=3,
            images_per_node=4,
        )
        assert graph.num_node_features == 8
        assert compressor is not None
        assert compressor.output_dim == 8

    def test_no_pca_keeps_raw_dim(self, monkeypatch):
        """Con USE_PCA=False, el grafo debe tener features de 3072 dims."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        import config
        monkeypatch.setattr(config, "USE_PCA", False)

        import importlib
        import main
        importlib.reload(main)

        cifar_data = self._make_cifar_data()
        graph, compressor = main._build_graph(
            cifar_data=cifar_data,
            k_neighbors=3,
            images_per_node=4,
        )
        assert graph.num_node_features == 3 * 32 * 32
        assert compressor is None


# ---------------------------------------------------------------------------
# Patch 1: PCA output must be L2-normalized
# ---------------------------------------------------------------------------

class TestPCAOutputNormalized:
    """PCA output must be unit-norm after both fit_transform and transform."""

    def test_fit_transform_output_is_unit_norm(self):
        from src.data.pca_compressor import PCACompressor
        f = torch.rand(40, 128)
        comp = PCACompressor(n_components=16)
        out = comp.fit_transform(f)
        norms = out.norm(dim=1)
        assert torch.allclose(norms, torch.ones(40), atol=1e-5), \
            "fit_transform debe devolver vectores L2-normalizados"

    def test_transform_output_is_unit_norm(self):
        from src.data.pca_compressor import PCACompressor
        train = torch.rand(40, 128)
        test = torch.rand(15, 128)
        comp = PCACompressor(n_components=16)
        comp.fit_transform(train)
        out = comp.transform(test)
        norms = out.norm(dim=1)
        assert torch.allclose(norms, torch.ones(15), atol=1e-5), \
            "transform debe devolver vectores L2-normalizados"

    def test_normalization_does_not_break_no_leakage(self):
        """Normalizacion no debe cambiar el hecho de que train y test
        se transforman de forma independiente."""
        from src.data.pca_compressor import PCACompressor
        train = torch.rand(30, 64)
        test = torch.rand(10, 64)
        comp = PCACompressor(n_components=8)
        train_out = comp.fit_transform(train)
        test_out = comp.transform(test)
        # Retransformar train no debe cambiar nada
        train_retransformed = comp.transform(train)
        assert torch.allclose(train_out, train_retransformed, atol=1e-5)
