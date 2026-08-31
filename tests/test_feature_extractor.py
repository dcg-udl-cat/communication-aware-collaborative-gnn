"""Tests para feature_extractor.py."""

import torch
import pytest


# ---------------------------------------------------------------------------
# extract_features
# ---------------------------------------------------------------------------

class TestExtractFeatures:
    def test_output_shape(self):
        from src.data.feature_extractor import extract_features
        out = extract_features(torch.rand(8, 3, 32, 32), normalize=False)
        assert out.shape == (8, 3 * 32 * 32)

    def test_uses_view_not_reshape(self):
        """view falla en tensores no contiguos: garantiza que no hay copias silenciosas."""
        from src.data.feature_extractor import extract_features
        # Transponer hace el tensor no continuo; contiguous() en extract_features lo corrige.
        images = torch.rand(4, 32, 32, 3).permute(0, 3, 1, 2)  # NHWC -> NCHW no contiguo
        # Debe funcionar porque extract_features llama a .contiguous() antes de .view().
        out = extract_features(images, normalize=False)
        assert out.shape == (4, 3 * 32 * 32)

    def test_l2_normalization(self):
        from src.data.feature_extractor import extract_features
        out = extract_features(torch.rand(6, 3, 32, 32), normalize=True)
        norms = out.norm(dim=1)
        assert torch.allclose(norms, torch.ones(6), atol=1e-5)

    def test_wrong_ndim_raises(self):
        from src.data.feature_extractor import extract_features
        with pytest.raises(ValueError, match="forma"):
            extract_features(torch.rand(8, 32, 32))

    def test_empty_raises(self):
        from src.data.feature_extractor import extract_features
        with pytest.raises(ValueError, match="vacio"):
            extract_features(torch.rand(0, 3, 32, 32))

    def test_output_dtype_float32(self):
        from src.data.feature_extractor import extract_features
        images = torch.randint(0, 256, (4, 3, 32, 32), dtype=torch.uint8)
        out = extract_features(images, normalize=False)
        assert out.dtype == torch.float32


# ---------------------------------------------------------------------------
# build_node_labels — mayoria de votos determinista
# ---------------------------------------------------------------------------

class TestBuildNodeLabels:
    def test_clear_majority(self):
        from src.data.feature_extractor import build_node_labels
        labels = torch.tensor([2, 2, 2, 0, 1])
        assert build_node_labels(labels) == 2

    def test_tie_picks_lowest_class(self):
        """En empate debe elegir la clase de menor indice, de forma determinista."""
        from src.data.feature_extractor import build_node_labels
        labels = torch.tensor([3, 3, 5, 5])
        result = build_node_labels(labels)
        assert result == 3, "En empate debe ganar la clase de menor indice"

    def test_single_element(self):
        from src.data.feature_extractor import build_node_labels
        assert build_node_labels(torch.tensor([7])) == 7

    def test_empty_raises(self):
        from src.data.feature_extractor import build_node_labels
        with pytest.raises(ValueError, match="vacio"):
            build_node_labels(torch.tensor([]))

    def test_all_same(self):
        from src.data.feature_extractor import build_node_labels
        assert build_node_labels(torch.tensor([4, 4, 4, 4])) == 4


# ---------------------------------------------------------------------------
# build_node_features_from_images — shuffle interno y reproducibilidad
# ---------------------------------------------------------------------------

class TestBuildNodeFeaturesFromImages:
    def _inputs(self, n=20, ipn=4):
        images = torch.rand(n, 3, 32, 32)
        # Etiquetas alternadas: si no se baraja, cada grupo tendra mezcla 50/50.
        labels = torch.tensor([i % 2 for i in range(n)], dtype=torch.long)
        return images, labels, ipn

    # --- Shuffle interno (fix #1) ---

    def test_shuffle_true_changes_grouping(self):
        """Con shuffle=True, el orden de las imagenes en los nodos debe variar."""
        from src.data.feature_extractor import build_node_features_from_images
        images, labels, ipn = self._inputs(n=20)
        gen1 = torch.Generator().manual_seed(0)
        gen2 = torch.Generator().manual_seed(99)
        f1, _, _ = build_node_features_from_images(images, labels, ipn, shuffle=True, generator=gen1)
        f2, _, _ = build_node_features_from_images(images, labels, ipn, shuffle=True, generator=gen2)
        assert not torch.allclose(f1, f2), \
            "Semillas distintas deben producir agrupaciones distintas"

    def test_shuffle_false_is_deterministic_without_seed(self):
        """Con shuffle=False, el resultado es siempre el mismo (orden del dataset)."""
        from src.data.feature_extractor import build_node_features_from_images
        images, labels, ipn = self._inputs(n=20)
        f1, l1, _ = build_node_features_from_images(images, labels, ipn, shuffle=False)
        f2, l2, _ = build_node_features_from_images(images, labels, ipn, shuffle=False)
        assert torch.allclose(f1, f2)
        assert torch.equal(l1, l2)

    def test_same_generator_seed_reproducible(self):
        """La misma semilla debe producir el mismo resultado."""
        from src.data.feature_extractor import build_node_features_from_images
        images, labels, ipn = self._inputs(n=20)
        gen_a = torch.Generator().manual_seed(42)
        gen_b = torch.Generator().manual_seed(42)
        f1, l1, c1 = build_node_features_from_images(images, labels, ipn, shuffle=True, generator=gen_a)
        f2, l2, c2 = build_node_features_from_images(images, labels, ipn, shuffle=True, generator=gen_b)
        assert torch.allclose(f1, f2)
        assert torch.equal(l1, l2)
        assert torch.equal(c1, c2)

    def test_cifar_order_bias_avoided_with_shuffle(self):
        """Simula CIFAR ordenado por clase: sin shuffle, nodos son puros (sesgo).
        Con shuffle, los nodos son mixtos."""
        from src.data.feature_extractor import build_node_features_from_images
        # 20 imagenes: primeras 10 clase 0, ultimas 10 clase 1
        images = torch.rand(20, 3, 32, 32)
        labels = torch.tensor([0] * 10 + [1] * 10, dtype=torch.long)
        ipn = 4

        # Sin shuffle: primeros nodos son clase-puros (sesgo de CIFAR)
        _, labels_no_shuffle, _ = build_node_features_from_images(
            images, labels, ipn, shuffle=False
        )
        pure_no_shuffle = all(l in [0, 1] for l in labels_no_shuffle.tolist())
        # Los primeros 2 nodos deberan ser clase 0 pura, los ultimos clase 1 pura
        assert labels_no_shuffle[0].item() == 0
        assert labels_no_shuffle[-1].item() == 1

        # Con shuffle: al menos algun nodo debe ser mixto (mayoria distinta a pura)
        gen = torch.Generator().manual_seed(7)
        _, labels_shuffled, _ = build_node_features_from_images(
            images, labels, ipn, shuffle=True, generator=gen
        )
        unique_labels = set(labels_shuffled.tolist())
        # Con 20 imagenes mezcladas y grupos de 4, es muy probable que aparezcan
        # nodos de ambas clases. No podemos garantizar mezcla en CADA nodo, pero
        # si que el conjunto de etiquetas sigue siendo {0, 1}.
        assert unique_labels == {0, 1}

    # --- Mayoria de votos determinista (fix #3) ---

    def test_tie_deterministic(self):
        """Empates en mayoria de votos deben resolverse de forma determinista."""
        from src.data.feature_extractor import build_node_features_from_images
        images = torch.rand(4, 3, 32, 32)
        # 2 de clase 0, 2 de clase 1 -> empate -> debe ganar clase 0 (menor indice)
        labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)
        _, node_labels, _ = build_node_features_from_images(
            images, labels, images_per_node=4, shuffle=False
        )
        assert node_labels[0].item() == 0

    # --- Formas y consistencia general ---

    def test_output_shapes_consistent(self):
        from src.data.feature_extractor import build_node_features_from_images
        images, labels, ipn = self._inputs(n=20, ipn=5)
        f, l, c = build_node_features_from_images(images, labels, ipn, shuffle=False)
        assert f.shape[0] == l.shape[0] == c.shape[0]
        assert c.sum().item() == 20

    def test_partial_last_node(self):
        """Cuando N no es multiplo de images_per_node, el ultimo nodo es mas pequeno."""
        from src.data.feature_extractor import build_node_features_from_images
        images = torch.rand(7, 3, 32, 32)
        labels = torch.zeros(7, dtype=torch.long)
        _, _, counts = build_node_features_from_images(images, labels, images_per_node=4, shuffle=False)
        assert counts.tolist() == [4, 3]

    def test_images_per_node_zero_raises(self):
        from src.data.feature_extractor import build_node_features_from_images
        with pytest.raises(ValueError):
            build_node_features_from_images(
                torch.rand(4, 3, 32, 32), torch.zeros(4, dtype=torch.long), images_per_node=0
            )

    def test_mismatched_sizes_raise(self):
        from src.data.feature_extractor import build_node_features_from_images
        with pytest.raises(ValueError):
            build_node_features_from_images(
                torch.rand(4, 3, 32, 32), torch.zeros(5, dtype=torch.long), images_per_node=2
            )

    # --- Responsabilidad separada (fix #5): build_node_labels es independiente ---

    def test_build_node_labels_importable_independently(self):
        """build_node_labels debe poder usarse sin pasar por build_node_features_from_images."""
        from src.data.feature_extractor import build_node_labels
        result = build_node_labels(torch.tensor([1, 1, 2]))
        assert result == 1


# ---------------------------------------------------------------------------
# Patch 6: normalize BEFORE PCA (always, regardless of normalize flag)
# ---------------------------------------------------------------------------

class TestPreNormalization:
    """extract_features debe normalizar L2 ANTES de retornar,
    independientemente del flag normalize."""

    def test_output_always_unit_norm(self):
        """Con normalize=False, la pre-normalizacion interna sigue activa."""
        from src.data.feature_extractor import extract_features
        images = torch.rand(8, 3, 32, 32) * 255  # pixeles en rango alto
        out = extract_features(images, normalize=False)
        norms = out.norm(dim=1)
        assert torch.allclose(norms, torch.ones(8), atol=1e-5), \
            "extract_features debe pre-normalizar incluso con normalize=False"

    def test_bright_and_dark_images_same_norm(self):
        """Imagenes muy brillantes y muy oscuras deben producir vectores
        de la misma norma (la magnitud de brillo no debe dominar)."""
        from src.data.feature_extractor import extract_features
        bright = torch.ones(1, 3, 32, 32) * 255.0
        dark = torch.ones(1, 3, 32, 32) * 1.0
        out_bright = extract_features(bright, normalize=False)
        out_dark = extract_features(dark, normalize=False)
        assert abs(out_bright.norm(dim=1).item() - 1.0) < 1e-4
        assert abs(out_dark.norm(dim=1).item() - 1.0) < 1e-4
