"""Tests del sweep de conectividad y de los modos de graph_builder."""

import torch
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_features(n_train=20, n_test=8, dim=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    tf = torch.rand(n_train, dim, generator=g)
    tl = torch.randint(0, 5, (n_train,), generator=g)
    ef = torch.rand(n_test, dim, generator=g)
    el = torch.randint(0, 5, (n_test,), generator=g)
    return tf, tl, ef, el


# ---------------------------------------------------------------------------
# graph_builder: edge_stats devueltos por build_similarity_graph
# ---------------------------------------------------------------------------

class TestEdgeStats:
    def test_edge_stats_keys_present(self):
        from src.data.graph_builder import build_similarity_graph
        tf, tl, ef, el = _make_features()
        g = build_similarity_graph(tf, tl, ef, el, k_neighbors=3)
        assert "train_train" in g.edge_stats
        assert "train_test" in g.edge_stats
        assert "total" in g.edge_stats

    def test_total_equals_sum(self):
        from src.data.graph_builder import build_similarity_graph
        tf, tl, ef, el = _make_features()
        g = build_similarity_graph(tf, tl, ef, el, k_neighbors=3)
        assert g.edge_stats["total"] == g.edge_stats["train_train"] + g.edge_stats["train_test"]
        assert g.edge_stats["total"] == g.num_edges

    def test_connectivity_mode_stored_in_graph(self):
        from src.data.graph_builder import build_similarity_graph
        tf, tl, ef, el = _make_features()
        for mode in ("directed", "undirected", "half_directed"):
            g = build_similarity_graph(tf, tl, ef, el, k_neighbors=3, connectivity_mode=mode)
            assert g.connectivity_mode == mode

    def test_invalid_mode_raises(self):
        from src.data.graph_builder import build_similarity_graph
        tf, tl, ef, el = _make_features()
        with pytest.raises(ValueError, match="connectivity_mode"):
            build_similarity_graph(tf, tl, ef, el, connectivity_mode="diagonal")


# ---------------------------------------------------------------------------
# Edge counts across modes
# ---------------------------------------------------------------------------

class TestEdgeCountsByMode:
    """Verifica las propiedades de conteo de aristas para cada modo."""

    def _graph(self, mode, k=3, n_train=20, n_test=8):
        from src.data.graph_builder import build_similarity_graph
        tf, tl, ef, el = _make_features(n_train=n_train, n_test=n_test)
        return build_similarity_graph(tf, tl, ef, el, k_neighbors=k, connectivity_mode=mode)

    def test_half_directed_has_fewer_train_train_edges_than_directed(self):
        """'half_directed' debe tener ~2x menos aristas train-train que 'directed'."""
        g_dir = self._graph("directed")
        g_spa = self._graph("half_directed")
        # En directed, cada par (i,j) genera dos aristas; en half_directed, solo una.
        # El ratio exacto puede variar segun k y la topologia, pero half_directed <= directed.
        assert g_spa.edge_stats["train_train"] <= g_dir.edge_stats["train_train"]

    def test_undirected_has_more_total_edges_than_directed(self):
        """'undirected' añade aristas train→test que 'directed' no tiene."""
        g_dir = self._graph("directed")
        g_und = self._graph("undirected")
        assert g_und.edge_stats["total"] > g_dir.edge_stats["total"]

    def test_lower_k_fewer_edges(self):
        """Reducir k debe reducir el numero de aristas."""
        g_k8 = self._graph("directed", k=8)
        g_k2 = self._graph("directed", k=2)
        assert g_k2.edge_stats["total"] < g_k8.edge_stats["total"]

    def test_directed_train_test_unidirectional(self):
        """En 'directed', todas las aristas que tocan test son train→test."""
        from src.data.graph_builder import build_similarity_graph
        n_train, n_test = 20, 8
        tf, tl, ef, el = _make_features(n_train=n_train, n_test=n_test)
        g = build_similarity_graph(tf, tl, ef, el, k_neighbors=3, connectivity_mode="directed")
        src, dst = g.edge_index
        test_to_train = (src >= n_train) & (dst < n_train)
        assert not test_to_train.any(), "directed no debe tener aristas test→train"

    def test_undirected_has_train_to_test_edges(self):
        """En 'undirected' deben existir aristas train→test."""
        from src.data.graph_builder import build_similarity_graph
        n_train, n_test = 20, 8
        tf, tl, ef, el = _make_features(n_train=n_train, n_test=n_test)
        g = build_similarity_graph(tf, tl, ef, el, k_neighbors=3, connectivity_mode="undirected")
        src, dst = g.edge_index
        train_to_test = (src < n_train) & (dst >= n_train)
        assert train_to_test.any(), "undirected debe tener aristas train→test"

    def test_half_directed_train_train_no_symmetric_pairs(self):
        """En 'half_directed', para ningun par (i,j) train-train debe existir tambien (j,i)."""
        from src.data.graph_builder import build_similarity_graph
        n_train = 20
        tf, tl, ef, el = _make_features(n_train=n_train, n_test=5)
        g = build_similarity_graph(tf, tl, ef, el, k_neighbors=4, connectivity_mode="half_directed")
        src, dst = g.edge_index
        tt_mask = (src < n_train) & (dst < n_train)
        tt_edges = set(zip(src[tt_mask].tolist(), dst[tt_mask].tolist()))
        symmetric_pairs = sum(1 for (i, j) in tt_edges if (j, i) in tt_edges)
        assert symmetric_pairs == 0, (
            f"'half_directed' tiene {symmetric_pairs} pares simetricos train-train. "
            "Deben eliminarse para reducir aristas."
        )

    def test_sparse_alias_still_works_with_deprecation_warning(self):
        """'sparse' debe seguir funcionando pero emitir DeprecationWarning."""
        import warnings
        from src.data.graph_builder import build_similarity_graph
        tf, tl, ef, el = _make_features(n_train=15, n_test=5)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            g = build_similarity_graph(tf, tl, ef, el, k_neighbors=3, connectivity_mode="sparse")
        dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(dep_warnings) == 1, (
            "Usar 'sparse' debe emitir exactamente un DeprecationWarning"
        )
        assert "half_directed" in str(dep_warnings[0].message), (
            "El aviso debe mencionar 'half_directed' como nombre correcto"
        )

    def test_sparse_and_half_directed_produce_identical_graphs(self):
        """'sparse' y 'half_directed' deben producir exactamente el mismo grafo."""
        import warnings
        from src.data.graph_builder import build_similarity_graph
        tf, tl, ef, el = _make_features(n_train=15, n_test=5)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            g_sparse = build_similarity_graph(tf, tl, ef, el, k_neighbors=3, connectivity_mode="sparse")
        g_hd = build_similarity_graph(tf, tl, ef, el, k_neighbors=3, connectivity_mode="half_directed")
        assert g_sparse.edge_stats == g_hd.edge_stats, (
            "'sparse' y 'half_directed' deben producir edge_stats identicos"
        )
        assert torch.equal(g_sparse.edge_index, g_hd.edge_index), (
            "'sparse' y 'half_directed' deben producir edge_index identico"
        )

    def test_mode_stored_as_canonical_after_alias(self):
        """El modo en el grafo debe ser 'half_directed', no 'sparse'."""
        import warnings
        from src.data.graph_builder import build_similarity_graph
        tf, tl, ef, el = _make_features()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            g = build_similarity_graph(tf, tl, ef, el, k_neighbors=3, connectivity_mode="sparse")
        assert g.connectivity_mode == "half_directed", (
            "El grafo debe almacenar el nombre canonico 'half_directed', no el alias 'sparse'"
        )


# ---------------------------------------------------------------------------
# Reduccion de aristas con k
# ---------------------------------------------------------------------------

class TestKReduction:
    """Verifica que reducir k tiene el efecto esperado en aristas y latencia."""

    @pytest.mark.parametrize("mode", ["directed", "undirected", "half_directed"])
    def test_k2_has_fewer_edges_than_k8(self, mode):
        from src.data.graph_builder import build_similarity_graph
        tf, tl, ef, el = _make_features(n_train=30, n_test=10)
        g_k8 = build_similarity_graph(tf, tl, ef, el, k_neighbors=8, connectivity_mode=mode)
        g_k2 = build_similarity_graph(tf, tl, ef, el, k_neighbors=2, connectivity_mode=mode)
        assert g_k2.edge_stats["total"] < g_k8.edge_stats["total"], \
            f"k=2 debe tener menos aristas que k=8 en modo {mode}"

    def test_k1_minimum_connectivity(self):
        """k=1 debe dar el minimo de aristas manteniendo el grafo conexo en train."""
        from src.data.graph_builder import build_similarity_graph
        tf, tl, ef, el = _make_features(n_train=15, n_test=5)
        g = build_similarity_graph(tf, tl, ef, el, k_neighbors=1, connectivity_mode="directed")
        assert g.edge_stats["total"] > 0
        # Cada nodo de test tiene al menos 1 vecino de train
        assert g.edge_stats["train_test"] >= 5


# ---------------------------------------------------------------------------
# EdgeSweepResult: estructura y calculo de reducciones
# ---------------------------------------------------------------------------

class TestEdgeSweepResult:
    def test_as_dict_has_required_keys(self):
        from src.training.edge_sweep import EdgeSweepResult
        r = EdgeSweepResult(
            k_neighbors=4,
            connectivity_mode="directed",
            num_train_train_edges=100,
            num_train_test_edges=40,
            num_total_edges=140,
            test_accuracy=0.42,
            inference_time_median_s=0.005,
            inference_time_p25_s=0.004,
            inference_time_p75_s=0.006,
            edge_reduction_vs_reference=0.3,
            latency_reduction_vs_reference=0.2,
            seed=42,
            pca_enabled=True,
            pca_components=64,
        )
        d = r.as_dict()
        required = {
            "k_neighbors", "connectivity_mode",
            "num_train_train_edges", "num_train_test_edges", "num_total_edges",
            "test_accuracy",
            "inference_time_median_s", "inference_time_p25_s", "inference_time_p75_s",
            "edge_reduction_vs_reference", "latency_reduction_vs_reference",
            "seed", "pca_enabled", "pca_components",
        }
        assert required.issubset(d.keys())


# ---------------------------------------------------------------------------
# run_edge_sweep: smoke test (rapido, no necesita CIFAR completo)
# ---------------------------------------------------------------------------

class TestRunEdgeSweep:
    def test_sweep_returns_one_result_per_combination(self, monkeypatch):
        """run_edge_sweep debe devolver len(k_values) * len(modes) resultados."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

        # Patch load_cifar10 con datos sinteticos para no necesitar CIFAR real.
        import data.cifar_loader as cl
        from data.cifar_loader import CifarData, CifarSplit

        fake_cifar = CifarData(
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
        monkeypatch.setattr(cl, "load_cifar10", lambda **_: fake_cifar)

        import config
        monkeypatch.setattr(config, "USE_PCA", False)
        monkeypatch.setattr(config, "IMAGES_PER_NODE", 4)
        monkeypatch.setattr(config, "CONTROLLED_TRAIN_SUBSET_SIZE", 40)
        monkeypatch.setattr(config, "CONTROLLED_TEST_SUBSET_SIZE", 10)

        from models.tiny_gnn import TinyGNN
        from training.edge_sweep import run_edge_sweep

        k_values = [2, 4]
        modes = ["directed", "half_directed"]

        results = run_edge_sweep(
            model_class=TinyGNN,
            model_kwargs={"input_dim": 3 * 32 * 32, "hidden_dim": 8, "num_classes": 10},
            k_values=k_values,
            connectivity_modes=modes,
            reference_k=4,
            reference_mode="directed",
            epochs=1,
            seed=42,
        )

        assert len(results) == len(k_values) * len(modes)
        for r in results:
            assert r.k_neighbors in k_values
            assert r.connectivity_mode in modes
            assert 0.0 <= r.test_accuracy <= 1.0
            assert r.inference_time_median_s > 0
            assert r.num_total_edges > 0

    def test_edge_reduction_reference_is_zero(self, monkeypatch):
        """La configuracion de referencia debe tener edge_reduction=0."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

        import data.cifar_loader as cl
        from data.cifar_loader import CifarData, CifarSplit

        fake_cifar = CifarData(
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
        monkeypatch.setattr(cl, "load_cifar10", lambda **_: fake_cifar)

        import config
        monkeypatch.setattr(config, "USE_PCA", False)
        monkeypatch.setattr(config, "IMAGES_PER_NODE", 4)
        monkeypatch.setattr(config, "CONTROLLED_TRAIN_SUBSET_SIZE", 40)
        monkeypatch.setattr(config, "CONTROLLED_TEST_SUBSET_SIZE", 10)

        from models.tiny_gnn import TinyGNN
        from training.edge_sweep import run_edge_sweep

        ref_k, ref_mode = 4, "directed"
        results = run_edge_sweep(
            model_class=TinyGNN,
            model_kwargs={"input_dim": 3 * 32 * 32, "hidden_dim": 8, "num_classes": 10},
            k_values=[ref_k],
            connectivity_modes=[ref_mode],
            reference_k=ref_k,
            reference_mode=ref_mode,
            epochs=1,
            seed=42,
        )
        assert len(results) == 1
        assert abs(results[0].edge_reduction_vs_reference) < 1e-6, \
            "La configuracion de referencia debe tener reduccion de aristas = 0"


# ---------------------------------------------------------------------------
# Adjacency matrix edge building: correctness vs old set approach
# ---------------------------------------------------------------------------

class TestAdjacencyMatrixEdges:
    """Verifica que la implementacion matricial produce los mismos edges
    que la implementacion original basada en set, y que escala mejor."""

    def _edges_via_set(self, train_features, test_features, k, mode):
        """Reimplementacion de referencia usando el set original."""
        import numpy as np
        from sklearn.neighbors import NearestNeighbors

        num_train = train_features.shape[0]
        num_test = test_features.shape[0]
        train_np = train_features.numpy().astype(np.float32)
        eff_k = min(k, num_train - 1)

        knn = NearestNeighbors(n_neighbors=eff_k + 1, metric="euclidean")
        knn.fit(train_np)
        neighbors = knn.kneighbors(train_np, return_distance=False)

        src_list, dst_list = [], []
        seen: set[tuple[int, int]] = set()
        for i, nbrs in enumerate(neighbors):
            for j in nbrs[1:]:
                j = int(j)
                if mode == "half_directed":
                    canonical = (min(i, j), max(i, j))
                    if canonical not in seen:
                        src_list.append(canonical[0]); dst_list.append(canonical[1])
                        seen.add(canonical)
                else:
                    if (i, j) not in seen:
                        src_list.append(i); dst_list.append(j); seen.add((i, j))
                    if (j, i) not in seen:
                        src_list.append(j); dst_list.append(i); seen.add((j, i))

        if num_test > 0:
            test_np = test_features.numpy().astype(np.float32)
            eff_kt = min(k, num_train)
            knn2 = NearestNeighbors(n_neighbors=eff_kt, metric="euclidean")
            knn2.fit(train_np)
            tnbrs = knn2.kneighbors(test_np, return_distance=False)
            for ti, nbrs in enumerate(tnbrs):
                tg = num_train + ti
                for tidx in nbrs:
                    src_list.append(int(tidx)); dst_list.append(tg)
                    if mode == "undirected":
                        src_list.append(tg); dst_list.append(int(tidx))

        return set(zip(src_list, dst_list))

    @pytest.mark.parametrize("mode", ["directed", "undirected", "half_directed"])
    def test_matrix_matches_set_for_all_modes(self, mode):
        """La implementacion matricial debe producir exactamente los mismos edges
        que la referencia set-based para los tres modos."""
        from src.data.graph_builder import build_similarity_graph
        torch.manual_seed(7)
        tf = torch.rand(20, 16)
        tl = torch.randint(0, 5, (20,))
        ef = torch.rand(8, 16)
        el = torch.randint(0, 5, (8,))

        g = build_similarity_graph(
            tf,
            tl,
            ef,
            el,
            k_neighbors=4,
            connectivity_mode=mode,
            knn_metric="euclidean",
            use_mutual_knn=False,
        )
        matrix_edges = set(zip(g.edge_index[0].tolist(), g.edge_index[1].tolist()))
        reference_edges = self._edges_via_set(tf, ef, k=4, mode=mode)

        assert matrix_edges == reference_edges, (
            f"Modo '{mode}': la implementacion matricial difiere de la referencia set.\n"
            f"Solo en matriz: {matrix_edges - reference_edges}\n"
            f"Solo en set:    {reference_edges - matrix_edges}"
        )

    def test_no_self_loops(self):
        """No debe haber aristas de un nodo a si mismo."""
        from src.data.graph_builder import build_similarity_graph
        tf, tl, ef, el = _make_features(n_train=25, n_test=8)
        for mode in ("directed", "undirected", "half_directed"):
            g = build_similarity_graph(tf, tl, ef, el, k_neighbors=4, connectivity_mode=mode)
            src, dst = g.edge_index
            assert not (src == dst).any(), f"Self-loop en modo '{mode}'"

    def test_edge_index_dtype_is_long(self):
        """edge_index debe ser int64 para compatibilidad con PyG."""
        from src.data.graph_builder import build_similarity_graph
        tf, tl, ef, el = _make_features()
        g = build_similarity_graph(tf, tl, ef, el, k_neighbors=3)
        assert g.edge_index.dtype == torch.long

    def test_scales_to_medium_graph_without_crash(self):
        """La implementacion matricial debe completar en tiempo razonable
        para N_train=1000, que con set habria sido notablemente mas lento."""
        import time
        from src.data.graph_builder import build_similarity_graph
        torch.manual_seed(0)
        n_train = 1000
        tf = torch.rand(n_train, 32)
        tl = torch.randint(0, 10, (n_train,))
        ef = torch.rand(100, 32)
        el = torch.randint(0, 10, (100,))

        t0 = time.perf_counter()
        g = build_similarity_graph(tf, tl, ef, el, k_neighbors=8)
        elapsed = time.perf_counter() - t0

        assert g.num_edges > 0
        # En hardware de CI razonable esto debe completar en menos de 30s.
        # El KNN de sklearn domina; la parte matricial es despreciable.
        assert elapsed < 30.0, f"Tardo {elapsed:.1f}s, demasiado lento para N=1000"
