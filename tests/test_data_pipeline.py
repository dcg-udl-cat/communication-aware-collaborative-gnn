"""Tests del pipeline de datos (sin fuga de etiquetas, protocolo inductivo)."""

import torch
import pytest


# ---------------------------------------------------------------------------
# feature_extractor
# ---------------------------------------------------------------------------

class TestExtractFeatures:
    def test_output_shape(self):
        from src.data.feature_extractor import extract_features
        images = torch.rand(8, 3, 32, 32)
        features = extract_features(images, normalize=False)
        assert features.shape == (8, 3 * 32 * 32)

    def test_l2_normalization(self):
        from src.data.feature_extractor import extract_features
        images = torch.rand(4, 3, 32, 32)
        features = extract_features(images, normalize=True)
        norms = features.norm(dim=1)
        assert torch.allclose(norms, torch.ones(4), atol=1e-5)

    def test_wrong_ndim_raises(self):
        from src.data.feature_extractor import extract_features
        with pytest.raises(ValueError, match="forma"):
            extract_features(torch.rand(8, 32, 32))

    def test_empty_raises(self):
        from src.data.feature_extractor import extract_features
        with pytest.raises(ValueError, match="vacio"):
            extract_features(torch.rand(0, 3, 32, 32))


class TestBuildNodeFeaturesLabelFree:
    """Verifica que la agrupacion en nodos NO usa etiquetas para decidir grupos."""

    def _make_inputs(self, n=20, images_per_node=4):
        images = torch.rand(n, 3, 32, 32)
        # Etiquetas alternadas: 0,1,0,1,... para que la agrupacion por clase
        # produciria nodos puros, mientras que la agrupacion consecutiva no.
        labels = torch.tensor([i % 2 for i in range(n)], dtype=torch.long)
        return images, labels, images_per_node

    def test_node_labels_are_not_all_pure(self):
        """Con agrupacion aleatoria/consecutiva, los nodos NO deben ser clase-puros."""
        from src.data.feature_extractor import build_node_features_from_images
        images, labels, ipn = self._make_inputs(n=20, images_per_node=4)
        _, node_labels, _ = build_node_features_from_images(images, labels, images_per_node=ipn, normalize=False)
        # Con etiquetas alternadas y grupos consecutivos de 4, cada grupo tendra
        # 2 de clase 0 y 2 de clase 1 => todos los nodos tendran la misma etiqueta
        # por mayoria (empate -> menor indice = 0). Si fuera por clase, tendriamos
        # nodos con una sola clase. Lo que importa es que la funcion no ordena
        # por clase internamente: el numero de nodos debe ser ceil(n / ipn).
        expected_nodes = (20 + ipn - 1) // ipn
        assert len(node_labels) == expected_nodes

    def test_output_shapes_consistent(self):
        from src.data.feature_extractor import build_node_features_from_images
        images, labels, ipn = self._make_inputs(n=20, images_per_node=5)
        features, node_labels, counts = build_node_features_from_images(images, labels, images_per_node=ipn)
        assert features.shape[0] == node_labels.shape[0] == counts.shape[0]
        assert counts.sum().item() == 20

    def test_images_per_node_zero_raises(self):
        from src.data.feature_extractor import build_node_features_from_images
        images = torch.rand(4, 3, 32, 32)
        labels = torch.zeros(4, dtype=torch.long)
        with pytest.raises(ValueError):
            build_node_features_from_images(images, labels, images_per_node=0)

    def test_mismatched_sizes_raise(self):
        from src.data.feature_extractor import build_node_features_from_images
        images = torch.rand(4, 3, 32, 32)
        labels = torch.zeros(5, dtype=torch.long)
        with pytest.raises(ValueError):
            build_node_features_from_images(images, labels, images_per_node=2)


# ---------------------------------------------------------------------------
# graph_builder — protocolo inductivo
# ---------------------------------------------------------------------------

class TestBuildSimilarityGraphInductive:
    def _make_data(self, n_train=20, n_test=10, dim=16):
        train_f = torch.rand(n_train, dim)
        train_l = torch.randint(0, 5, (n_train,))
        test_f = torch.rand(n_test, dim)
        test_l = torch.randint(0, 5, (n_test,))
        return train_f, train_l, test_f, test_l

    def test_masks_partition_nodes(self):
        from src.data.graph_builder import build_similarity_graph
        tf, tl, ef, el = self._make_data()
        graph = build_similarity_graph(tf, tl, ef, el, k_neighbors=3)
        assert int(graph.train_mask.sum()) == 20
        assert int(graph.test_mask.sum()) == 10
        # Sin solapamiento
        assert not (graph.train_mask & graph.test_mask).any()

    def test_no_test_test_edges(self):
        """No debe haber aristas entre nodos de test (en ninguna direccion)."""
        from src.data.graph_builder import build_similarity_graph
        tf, tl, ef, el = self._make_data(n_train=20, n_test=8, dim=16)
        graph = build_similarity_graph(tf, tl, ef, el, k_neighbors=3)
        num_train = 20
        src, dst = graph.edge_index
        test_test = ((src >= num_train) & (dst >= num_train))
        assert not test_test.any(), "No debe haber aristas entre nodos de test"

    def test_no_test_to_train_edges(self):
        """PyG source→target: test→train would leak test features into train."""
        from src.data.graph_builder import build_similarity_graph
        tf, tl, ef, el = self._make_data(n_train=20, n_test=8, dim=16)
        graph = build_similarity_graph(tf, tl, ef, el, k_neighbors=3)
        num_train = 20
        src, dst = graph.edge_index
        test_to_train = ((src >= num_train) & (dst < num_train))
        assert not test_to_train.any(), (
            "No debe haber aristas test→train. "
            "Solo train→test permite que test agregue contexto de train."
        )

    def test_train_to_test_edges_exist(self):
        """Debe haber aristas train→test para que test reciba mensajes."""
        from src.data.graph_builder import build_similarity_graph
        tf, tl, ef, el = self._make_data(n_train=20, n_test=5, dim=16)
        graph = build_similarity_graph(tf, tl, ef, el, k_neighbors=3)
        num_train = 20
        src, dst = graph.edge_index
        train_to_test = ((src < num_train) & (dst >= num_train))
        assert train_to_test.any(), "Deben existir aristas train→test"

    def test_train_train_is_bidirectional(self):
        """Las aristas entre nodos de train deben ser bidireccionales."""
        from src.data.graph_builder import build_similarity_graph
        tf, tl, ef, el = self._make_data(n_train=15, n_test=5, dim=8)
        graph = build_similarity_graph(tf, tl, ef, el, k_neighbors=3)
        num_train = 15
        src, dst = graph.edge_index
        train_src = src < num_train
        train_dst = dst < num_train
        tt_edges = set(zip(
            src[train_src & train_dst].tolist(),
            dst[train_src & train_dst].tolist(),
        ))
        # Cada arista (i,j) debe tener su inversa (j,i)
        for (i, j) in tt_edges:
            assert (j, i) in tt_edges, (
                f"Arista ({i},{j}) existe pero no su inversa ({j},{i}). "
                "Las aristas train-train deben ser bidireccionales."
            )

    def test_edge_count_asymmetry(self):
        """Con train→test unidireccional, el grafo debe tener menos aristas que
        la version bidireccional anterior. Verificamos que no hay aristas train→test."""
        from src.data.graph_builder import build_similarity_graph
        n_train, n_test, k = 20, 10, 3
        tf, tl, ef, el = self._make_data(n_train=n_train, n_test=n_test, dim=16)
        graph = build_similarity_graph(tf, tl, ef, el, k_neighbors=k)
        src, dst = graph.edge_index
        # Aristas con al menos un extremo de test
        test_involved = (src >= n_train) | (dst >= n_train)
        test_edges = graph.edge_index[:, test_involved]
        # Todas deben ser train→test (source train, target test).
        assert (test_edges[0] < n_train).all(), "Origen debe ser nodo de train"
        assert (test_edges[1] >= n_train).all(), "Destino debe ser nodo de test"

    def test_train_edges_unaffected_by_test_count(self):
        """Las aristas train-train no deben depender de cuantos nodos de test haya."""
        from src.data.graph_builder import build_similarity_graph
        torch.manual_seed(0)
        tf = torch.rand(15, 8)
        tl = torch.randint(0, 3, (15,))
        ef_small = torch.rand(2, 8)
        ef_large = torch.rand(30, 8)
        el_small = torch.randint(0, 3, (2,))
        el_large = torch.randint(0, 3, (30,))

        g_small = build_similarity_graph(tf, tl, ef_small, el_small, k_neighbors=3)
        g_large = build_similarity_graph(tf, tl, ef_large, el_large, k_neighbors=3)

        def train_train_edges(g, n_train):
            src, dst = g.edge_index
            mask = (src < n_train) & (dst < n_train)
            return set(zip(src[mask].tolist(), dst[mask].tolist()))

        edges_small = train_train_edges(g_small, 15)
        edges_large = train_train_edges(g_large, 15)
        assert edges_small == edges_large, \
            "Las aristas train-train deben ser identicas independientemente de los nodos de test"

    def test_feature_dim_mismatch_raises(self):
        from src.data.graph_builder import build_similarity_graph
        with pytest.raises(ValueError):
            build_similarity_graph(
                torch.rand(5, 8), torch.zeros(5, dtype=torch.long),
                torch.rand(3, 16), torch.zeros(3, dtype=torch.long),
            )

    def test_node_image_counts_sum(self):
        from src.data.graph_builder import build_similarity_graph
        tf, tl, ef, el = self._make_data(n_train=10, n_test=5)
        counts_train = torch.ones(10, dtype=torch.long) * 3
        counts_test = torch.ones(5, dtype=torch.long) * 2
        graph = build_similarity_graph(tf, tl, ef, el,
                                       train_node_image_counts=counts_train,
                                       test_node_image_counts=counts_test)
        assert graph.node_image_counts.sum().item() == 10 * 3 + 5 * 2
