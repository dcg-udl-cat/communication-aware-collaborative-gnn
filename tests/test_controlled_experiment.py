"""Tests del experimento controlado A/B/C."""

import torch
import pytest
from torch_geometric.data import Data


def _make_synthetic_cifar(n_train=60, n_test=20):
    """CifarData sintetico para tests sin descargar CIFAR."""
    from data.cifar_loader import CifarData, CifarSplit
    return CifarData(
        train=CifarSplit(
            images=torch.rand(n_train, 3, 32, 32),
            labels=torch.randint(0, 10, (n_train,)),
        ),
        test=CifarSplit(
            images=torch.rand(n_test, 3, 32, 32),
            labels=torch.randint(0, 10, (n_test,)),
        ),
        class_names=tuple(str(i) for i in range(10)),
    )


# ---------------------------------------------------------------------------
# images_per_node = 1 forzado
# ---------------------------------------------------------------------------

class TestImagesPerNodeDefault:
    def test_config_default_is_one(self):
        """IMAGES_PER_NODE debe ser 1 en config."""
        import config
        assert config.IMAGES_PER_NODE == 1, (
            f"IMAGES_PER_NODE={config.IMAGES_PER_NODE}. "
            "Debe ser 1: un tiny GNN no puede recuperarse de nodos ruidosos."
        )

    def test_controlled_experiment_uses_one(self):
        """El experimento controlado siempre usa images_per_node=1."""
        import training.controlled_experiment as ce
        assert ce._IMAGES_PER_NODE == 1


# ---------------------------------------------------------------------------
# KNN metrica coseno
# ---------------------------------------------------------------------------

class TestKNNMetric:
    def test_config_default_is_cosine(self):
        import config
        assert config.KNN_METRIC == "cosine"

    def test_cosine_graph_stored_on_data(self):
        from data.graph_builder import build_similarity_graph
        tf = torch.rand(20, 16)
        tl = torch.randint(0, 5, (20,))
        ef = torch.rand(8, 16)
        el = torch.randint(0, 5, (8,))
        g = build_similarity_graph(tf, tl, ef, el, k_neighbors=3, knn_metric="cosine")
        assert g.knn_metric == "cosine"

    def test_euclidean_graph_different_edges_than_cosine(self):
        """Coseno y euclidiano deben producir grafos distintos en la mayoria de casos."""
        import torch.nn.functional as F
        from data.graph_builder import build_similarity_graph
        torch.manual_seed(42)
        # Features no-uniformes donde coseno y euclidiano difieren
        tf = torch.rand(30, 32) * torch.arange(1, 31).float().unsqueeze(1)
        tl = torch.randint(0, 5, (30,))
        ef = torch.rand(10, 32)
        el = torch.randint(0, 5, (10,))
        g_cos = build_similarity_graph(tf, tl, ef, el, k_neighbors=4, knn_metric="cosine")
        g_euc = build_similarity_graph(tf, tl, ef, el, k_neighbors=4, knn_metric="euclidean")
        edges_cos = set(zip(g_cos.edge_index[0].tolist(), g_cos.edge_index[1].tolist()))
        edges_euc = set(zip(g_euc.edge_index[0].tolist(), g_euc.edge_index[1].tolist()))
        assert edges_cos != edges_euc, (
            "Coseno y euclidiano producen el mismo grafo. "
            "Las features pueden ser demasiado uniformes para este test."
        )

    def test_invalid_metric_raises(self):
        from data.graph_builder import build_similarity_graph
        tf = torch.rand(10, 8)
        tl = torch.randint(0, 3, (10,))
        ef = torch.rand(5, 8)
        el = torch.randint(0, 3, (5,))
        with pytest.raises(Exception):
            build_similarity_graph(tf, tl, ef, el, knn_metric="manhattan_nonsense")


# ---------------------------------------------------------------------------
# _replace_edges_with_random
# ---------------------------------------------------------------------------

class TestReplaceEdgesWithRandom:
    def _make_graph(self, n_train=20, n_test=8, k=3):
        from data.graph_builder import build_similarity_graph
        tf = torch.rand(n_train, 16)
        tl = torch.randint(0, 5, (n_train,))
        ef = torch.rand(n_test, 16)
        el = torch.randint(0, 5, (n_test,))
        return build_similarity_graph(tf, tl, ef, el, k_neighbors=k)

    def test_same_number_of_edges(self):
        from training.controlled_experiment import _replace_edges_with_random
        g = self._make_graph()
        rg = _replace_edges_with_random(g, seed=42)
        assert rg.num_edges == g.num_edges, (
            "El grafo aleatorio debe tener exactamente el mismo numero de aristas"
        )

    def test_different_topology(self):
        """El grafo aleatorio debe tener una topologia diferente al KNN."""
        from training.controlled_experiment import _replace_edges_with_random
        g = self._make_graph(n_train=30, n_test=10, k=5)
        rg = _replace_edges_with_random(g, seed=42)
        edges_orig = set(zip(g.edge_index[0].tolist(), g.edge_index[1].tolist()))
        edges_rand = set(zip(rg.edge_index[0].tolist(), rg.edge_index[1].tolist()))
        assert edges_orig != edges_rand, "El grafo aleatorio debe diferir del KNN"

    def test_no_test_to_train_edges_in_random(self):
        """El protocolo inductivo debe respetarse incluso con aristas aleatorias."""
        from training.controlled_experiment import _replace_edges_with_random
        n_train = 20
        g = self._make_graph(n_train=n_train, n_test=8, k=3)
        rg = _replace_edges_with_random(g, seed=0)
        src, dst = rg.edge_index
        test_to_train = (src >= n_train) & (dst < n_train)
        assert not test_to_train.any(), (
            "El grafo aleatorio no debe tener aristas test→train"
        )

    def test_same_seed_reproducible(self):
        from training.controlled_experiment import _replace_edges_with_random
        g = self._make_graph()
        rg1 = _replace_edges_with_random(g, seed=7)
        rg2 = _replace_edges_with_random(g, seed=7)
        assert torch.equal(rg1.edge_index, rg2.edge_index)

    def test_different_seeds_different_graphs(self):
        from training.controlled_experiment import _replace_edges_with_random
        g = self._make_graph(n_train=30, n_test=10, k=5)
        rg1 = _replace_edges_with_random(g, seed=1)
        rg2 = _replace_edges_with_random(g, seed=2)
        edges1 = set(zip(rg1.edge_index[0].tolist(), rg1.edge_index[1].tolist()))
        edges2 = set(zip(rg2.edge_index[0].tolist(), rg2.edge_index[1].tolist()))
        assert edges1 != edges2


# ---------------------------------------------------------------------------
# SetupResult y ControlledExperimentResult
# ---------------------------------------------------------------------------

class TestSetupResult:
    def _make_result(self, setup, acc):
        from training.controlled_experiment import SetupResult
        return SetupResult(
            setup=setup, test_accuracy=acc, val_accuracy=acc,
            train_accuracy=acc, inference_time_median_s=0.01,
            num_parameters=100, num_edges=50, edge_type="knn", seed=42,
        )

    def test_as_dict_keys(self):
        r = self._make_result("B_knn_gnn", 0.4)
        d = r.as_dict()
        required = {
            "setup", "test_accuracy", "val_accuracy", "train_accuracy",
            "inference_time_median_s", "num_parameters", "num_edges",
            "edge_type", "seed",
        }
        assert required.issubset(d.keys())


class TestInterpretation:
    def _make_experiment(self, acc_a, acc_b, acc_c):
        from training.controlled_experiment import SetupResult, ControlledExperimentResult

        def r(setup, acc, edge_type):
            return SetupResult(
                setup=setup, test_accuracy=acc, val_accuracy=acc,
                train_accuracy=acc, inference_time_median_s=0.01,
                num_parameters=100, num_edges=50, edge_type=edge_type, seed=42,
            )

        return ControlledExperimentResult(
            setup_a=r("A_mlp", acc_a, "none"),
            setup_b=r("B_knn_gnn", acc_b, "knn"),
            setup_c=r("C_random_gnn", acc_c, "random"),
            seeds_used=[42],
        )

    def test_b_beats_both_is_positive(self):
        exp = self._make_experiment(acc_a=0.30, acc_b=0.40, acc_c=0.25)
        assert "✅" in exp.interpretation

    def test_b_worse_than_a_is_warning(self):
        exp = self._make_experiment(acc_a=0.35, acc_b=0.20, acc_c=0.22)
        assert "⚠️" in exp.interpretation

    def test_b_equals_a_is_negative(self):
        exp = self._make_experiment(acc_a=0.30, acc_b=0.305, acc_c=0.25)
        assert "❌" in exp.interpretation

    def test_b_beats_a_but_not_c_is_shrug(self):
        exp = self._make_experiment(acc_a=0.25, acc_b=0.40, acc_c=0.395)
        assert "🤷" in exp.interpretation

    def test_b_over_a_computed_correctly(self):
        exp = self._make_experiment(acc_a=0.30, acc_b=0.45, acc_c=0.20)
        assert abs(exp.b_over_a - 0.15) < 1e-6

    def test_b_over_c_computed_correctly(self):
        exp = self._make_experiment(acc_a=0.30, acc_b=0.45, acc_c=0.20)
        assert abs(exp.b_over_c - 0.25) < 1e-6


# ---------------------------------------------------------------------------
# run_controlled_experiment: smoke test con datos sinteticos
# ---------------------------------------------------------------------------

class TestRunControlledExperiment:
    def test_three_setups_run_and_return_valid_result(self, monkeypatch):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

        import data.cifar_loader as cl
        monkeypatch.setattr(cl, "load_cifar10", lambda **_: _make_synthetic_cifar(60, 20))

        import config
        monkeypatch.setattr(config, "USE_PCA", False)
        monkeypatch.setattr(config, "CONTROLLED_TRAIN_SUBSET_SIZE", 60)
        monkeypatch.setattr(config, "CONTROLLED_TEST_SUBSET_SIZE", 20)

        from models.baselines import SimpleMLPBaseline
        from models.tiny_gnn import TinyGNN
        from training.controlled_experiment import run_controlled_experiment

        result = run_controlled_experiment(
            mlp_class=SimpleMLPBaseline,
            gnn_class=TinyGNN,
            mlp_kwargs={"input_dim": 3 * 32 * 32, "hidden_dim": 8, "num_classes": 10},
            gnn_kwargs={"input_dim": 3 * 32 * 32, "hidden_dim": 8, "num_classes": 10},
            k_neighbors=3,
            epochs=1,
            seeds=[42],
        )

        for setup in (result.setup_a, result.setup_b, result.setup_c):
            assert 0.0 <= setup.test_accuracy <= 1.0
            assert 0.0 <= setup.val_accuracy <= 1.0
            assert setup.num_parameters > 0
            assert setup.inference_time_median_s > 0

        # Setup A (MLP) ignora el grafo — num_edges es del grafo KNN
        # (el MLP recibe el mismo grafo pero no usa las aristas).
        assert result.setup_b.num_edges > 0, "Setup B debe tener aristas KNN"
        assert result.setup_c.num_edges > 0, "Setup C debe tener aristas aleatorias"
        assert result.setup_b.num_edges == result.setup_c.num_edges, (
            "B y C deben tener el mismo numero de aristas"
        )

        # La interpretacion debe ser una cadena no vacia
        assert isinstance(result.interpretation, str)
        assert len(result.interpretation) > 10

    def test_summary_contains_all_setups(self, monkeypatch):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

        import data.cifar_loader as cl
        monkeypatch.setattr(cl, "load_cifar10", lambda **_: _make_synthetic_cifar(60, 20))

        import config
        monkeypatch.setattr(config, "USE_PCA", False)
        monkeypatch.setattr(config, "CONTROLLED_TRAIN_SUBSET_SIZE", 60)
        monkeypatch.setattr(config, "CONTROLLED_TEST_SUBSET_SIZE", 20)

        from models.baselines import SimpleMLPBaseline
        from models.tiny_gnn import TinyGNN
        from training.controlled_experiment import run_controlled_experiment

        result = run_controlled_experiment(
            mlp_class=SimpleMLPBaseline,
            gnn_class=TinyGNN,
            mlp_kwargs={"input_dim": 3 * 32 * 32, "hidden_dim": 8, "num_classes": 10},
            gnn_kwargs={"input_dim": 3 * 32 * 32, "hidden_dim": 8, "num_classes": 10},
            k_neighbors=3,
            epochs=1,
            seeds=[42],
        )
        summary = result.summary()
        assert "Setup A" in summary or "A_mlp" in summary or "A (MLP" in summary
        assert "Setup B" in summary or "B_knn" in summary or "B (GNN" in summary
        assert "Setup C" in summary or "C_random" in summary or "C (GNN" in summary
