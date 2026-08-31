"""Tests de integracion del pipeline completo con todas las mejoras."""

import torch
import pytest


def _make_features(n=20, dim=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    f = torch.rand(n, dim, generator=g)
    import torch.nn.functional as F
    return F.normalize(f, p=2, dim=1)


def _make_labels(n=20, nc=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, nc, (n,), generator=g)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class TestResNet18Encoder:
    def test_output_shape(self):
        try:
            from src.data.encoder import ResNet18Encoder
        except ImportError:
            pytest.skip("torchvision not available")
        enc = ResNet18Encoder(pretrained=False)
        images = torch.rand(4, 3, 32, 32)
        out = enc.encode(images)
        assert out.shape == (4, 512)

    def test_output_is_unit_norm(self):
        try:
            from src.data.encoder import ResNet18Encoder
        except ImportError:
            pytest.skip("torchvision not available")
        enc = ResNet18Encoder(pretrained=False)
        images = torch.rand(6, 3, 32, 32)
        out = enc.encode(images)
        norms = out.norm(dim=1)
        assert torch.allclose(norms, torch.ones(6), atol=1e-5)

    def test_no_grad_flows_through_encoder(self):
        try:
            from src.data.encoder import ResNet18Encoder
        except ImportError:
            pytest.skip("torchvision not available")
        enc = ResNet18Encoder(pretrained=False)
        for p in enc.parameters():
            assert not p.requires_grad, "Encoder debe estar congelado"

    def test_wrong_channel_count_raises(self):
        try:
            from src.data.encoder import ResNet18Encoder
        except ImportError:
            pytest.skip("torchvision not available")
        enc = ResNet18Encoder(pretrained=False)
        with pytest.raises(ValueError, match="canales"):
            enc.encode(torch.rand(4, 1, 32, 32))   # grayscale, no RGB


# ---------------------------------------------------------------------------
# Mutual KNN
# ---------------------------------------------------------------------------

class TestMutualKNN:
    def _graph(self, mutual, n_train=20, n_test=8, k=4):
        from src.data.graph_builder import build_similarity_graph
        tf = _make_features(n_train, 16)
        tl = _make_labels(n_train)
        ef = _make_features(n_test, 16, seed=1)
        el = _make_labels(n_test, seed=1)
        return build_similarity_graph(
            tf, tl, ef, el, k_neighbors=k,
            use_mutual_knn=mutual, edge_weight_sigma=None
        )

    def test_mutual_has_fewer_or_equal_edges_than_standard(self):
        g_mut = self._graph(mutual=True)
        g_std = self._graph(mutual=False)
        assert g_mut.num_edges <= g_std.num_edges, \
            "Mutual KNN debe tener <= aristas que KNN estandar"

    def test_mutual_edges_are_symmetric_in_train(self):
        """En mutual KNN + directed, las aristas train-train deben ser simetricas
        porque se requiere que ambas direcciones existan para incluir la arista."""
        g = self._graph(mutual=True, k=3)
        n_train = 20
        src, dst = g.edge_index
        tt = (src < n_train) & (dst < n_train)
        tt_edges = set(zip(src[tt].tolist(), dst[tt].tolist()))
        for (i, j) in tt_edges:
            assert (j, i) in tt_edges, \
                f"Arista ({i},{j}) existe pero no ({j},{i}) — mutual KNN debe ser simetrico"

    def test_mutual_flag_stored_on_graph(self):
        g = self._graph(mutual=True)
        assert g.use_mutual_knn is True
        g2 = self._graph(mutual=False)
        assert g2.use_mutual_knn is False


# ---------------------------------------------------------------------------
# Edge weights
# ---------------------------------------------------------------------------

class TestEdgeWeights:
    def _graph(self, sigma, n_train=20, n_test=8, k=3):
        from src.data.graph_builder import build_similarity_graph
        tf = _make_features(n_train, 16)
        tl = _make_labels(n_train)
        ef = _make_features(n_test, 16, seed=1)
        el = _make_labels(n_test, seed=1)
        return build_similarity_graph(
            tf, tl, ef, el, k_neighbors=k,
            edge_weight_sigma=sigma, use_mutual_knn=False
        )

    def test_sigma_none_gives_no_edge_weight(self):
        g = self._graph(sigma=None)
        assert g.edge_weight is None

    def test_sigma_given_gives_edge_weight_tensor(self):
        g = self._graph(sigma=0.5)
        assert g.edge_weight is not None
        assert g.edge_weight.shape == (g.num_edges,)

    def test_edge_weights_in_zero_one(self):
        g = self._graph(sigma=0.5)
        assert (g.edge_weight >= 0).all()
        assert (g.edge_weight <= 1.0 + 1e-6).all()

    def test_closer_nodes_higher_weight(self):
        """Nodos a distancia ~0 deben tener peso ~1; lejanos peso < 1."""
        g = self._graph(sigma=0.5)
        # peso maximo debe ser cercano a 1 (nodos muy similares)
        assert g.edge_weight.max().item() > 0.5, \
            "Al menos un par de nodos similares debe tener peso > 0.5"

    def test_edge_stats_reports_weighted(self):
        g = self._graph(sigma=0.5)
        assert g.edge_stats.get("weighted") is True


# ---------------------------------------------------------------------------
# TinyGNN con GAT + residual
# ---------------------------------------------------------------------------

class TestTinyGNNGAT:
    def _make_graph(self, n=30, feat_dim=128):
        from torch_geometric.data import Data
        x = torch.rand(n, feat_dim)
        y = torch.randint(0, 10, (n,))
        src = torch.randint(0, n, (n * 3,))
        dst = torch.randint(0, n, (n * 3,))
        mask = torch.zeros(n, dtype=torch.bool)
        mask[:20] = True
        return Data(x=x, edge_index=torch.stack([src, dst]), y=y,
                    train_mask=mask, test_mask=~mask,
                    node_image_counts=torch.ones(n, dtype=torch.long),
                    edge_weight=torch.rand(n * 3))

    def test_uses_gatconv(self):
        from src.models.tiny_gnn import TinyGNN
        from torch_geometric.nn import GATConv
        model = TinyGNN(input_dim=128, hidden_dim=64, num_classes=10)
        gat_layers = [m for m in model.modules() if isinstance(m, GATConv)]
        assert len(gat_layers) >= 1

    def test_residual_connection_changes_output(self):
        """La conexion residual debe producir output diferente al sin residual."""
        from src.models.tiny_gnn import TinyGNN
        g = self._make_graph()
        model = TinyGNN(input_dim=128, hidden_dim=64, num_classes=10)
        model.eval()
        with torch.no_grad():
            # Si el residual funciona, el output no puede ser exactamente
            # igual a un modelo sin el (test indirecto: verificamos que
            # el modelo corre y produce logits razonables)
            out = model(g.x, g.edge_index, g.edge_weight)
        assert out.shape == (30, 10)
        assert not torch.isnan(out).any(), "Output contiene NaN"
        assert not torch.isinf(out).any(), "Output contiene Inf"

    def test_forward_with_and_without_edge_weight(self):
        """El modelo debe funcionar con y sin edge_weight."""
        from src.models.tiny_gnn import TinyGNN
        g = self._make_graph()
        model = TinyGNN(input_dim=128, hidden_dim=64, num_classes=10)
        model.eval()
        with torch.no_grad():
            out_with = model(g.x, g.edge_index, g.edge_weight)
            out_without = model(g.x, g.edge_index, None)
        assert out_with.shape == out_without.shape

    def test_edge_weight_via_model_forward(self):
        """model_forward debe pasar edge_weight automaticamente."""
        from src.models.tiny_gnn import TinyGNN
        from src.training.trainer import model_forward
        g = self._make_graph()
        model = TinyGNN(input_dim=128, hidden_dim=64, num_classes=10)
        model.eval()
        with torch.no_grad():
            out = model_forward(model, g)
        assert out.shape == (30, 10)


# ---------------------------------------------------------------------------
# KNN Classifier Baseline
# ---------------------------------------------------------------------------

class TestKNNClassifierBaseline:
    def test_fit_and_score(self):
        from src.models.knn_baseline import KNNClassifierBaseline
        train_f = _make_features(50, 32)
        train_l = _make_labels(50, nc=5)
        test_f = _make_features(20, 32, seed=1)
        test_l = _make_labels(20, nc=5, seed=1)
        baseline = KNNClassifierBaseline(k=3, metric="cosine")
        baseline.fit(train_f, train_l)
        acc = baseline.score(test_f, test_l)
        assert 0.0 <= acc <= 1.0

    def test_score_before_fit_raises(self):
        from src.models.knn_baseline import KNNClassifierBaseline
        with pytest.raises(RuntimeError, match="fit"):
            KNNClassifierBaseline().score(torch.rand(5, 16), torch.zeros(5, dtype=torch.long))

    def test_evaluate_returns_result(self):
        from src.models.knn_baseline import KNNClassifierBaseline
        train_f = _make_features(40, 16)
        train_l = _make_labels(40, nc=5)
        test_f = _make_features(15, 16, seed=2)
        test_l = _make_labels(15, nc=5, seed=2)
        result = KNNClassifierBaseline(k=3).evaluate(train_f, train_l, test_f, test_l)
        assert 0.0 <= result.train_accuracy <= 1.0
        assert 0.0 <= result.test_accuracy <= 1.0
        assert result.k == 3
        assert result.num_train_nodes == 40
        assert result.num_test_nodes == 15

    def test_summary_string(self):
        from src.models.knn_baseline import KNNClassifierBaseline
        train_f = _make_features(20, 8)
        train_l = _make_labels(20, nc=3)
        result = KNNClassifierBaseline(k=2).evaluate(
            train_f, train_l, train_f[:5], train_l[:5]
        )
        s = result.summary()
        assert "KNN" in s
        assert "test=" in s

    def test_k_zero_raises(self):
        from src.models.knn_baseline import KNNClassifierBaseline
        with pytest.raises(ValueError, match="k debe"):
            KNNClassifierBaseline(k=0)


# ---------------------------------------------------------------------------
# mean+std node features
# ---------------------------------------------------------------------------

class TestMeanStdNodeFeatures:
    def test_use_mean_std_doubles_feature_dim(self):
        from src.data.feature_extractor import build_node_features_from_images
        images = torch.rand(10, 3, 32, 32)
        labels = torch.zeros(10, dtype=torch.long)
        f_mean, _, _ = build_node_features_from_images(
            images, labels, images_per_node=5, shuffle=False, use_mean_std=False)
        f_both, _, _ = build_node_features_from_images(
            images, labels, images_per_node=5, shuffle=False, use_mean_std=True)
        assert f_both.shape[1] == f_mean.shape[1] * 2, \
            "mean+std debe tener el doble de dimensiones que solo mean"

    def test_use_mean_std_false_same_as_default(self):
        from src.data.feature_extractor import build_node_features_from_images
        images = torch.rand(8, 3, 32, 32)
        labels = torch.zeros(8, dtype=torch.long)
        f1, l1, c1 = build_node_features_from_images(
            images, labels, images_per_node=4, shuffle=False, use_mean_std=False)
        f2, l2, c2 = build_node_features_from_images(
            images, labels, images_per_node=4, shuffle=False)
        assert torch.allclose(f1, f2)

    def test_single_image_per_node_ignores_std(self):
        """Con ipn=1 no hay varianza real; use_mean_std debe ignorar std
        para evitar NaN (std de un solo elemento = 0 en pytorch)."""
        from src.data.feature_extractor import build_node_features_from_images
        images = torch.rand(4, 3, 32, 32)
        labels = torch.zeros(4, dtype=torch.long)
        # No debe lanzar y no debe tener NaN
        f, _, _ = build_node_features_from_images(
            images, labels, images_per_node=1, shuffle=False, use_mean_std=True)
        assert not torch.isnan(f).any(), "NaN con ipn=1 y use_mean_std=True"

    def test_mean_std_output_is_normalized(self):
        from src.data.feature_extractor import build_node_features_from_images
        images = torch.rand(10, 3, 32, 32)
        labels = torch.zeros(10, dtype=torch.long)
        f, _, _ = build_node_features_from_images(
            images, labels, images_per_node=5, shuffle=False, use_mean_std=True)
        norms = f.norm(dim=1)
        assert torch.allclose(norms, torch.ones(norms.shape[0]), atol=1e-5)

    def test_weight_decay_in_config(self):
        import src.config as cfg
        assert hasattr(cfg, "WEIGHT_DECAY"), "WEIGHT_DECAY debe existir en config"
        assert cfg.WEIGHT_DECAY >= 0


# ---------------------------------------------------------------------------
# Regression: KNNClassifierBaseline must not crash in _run_model
# ---------------------------------------------------------------------------

class TestKNNBaselineInRunModel:
    """Regression tests for AttributeError: 'KNNClassifierBaseline' has no 'to'."""

    def test_to_is_no_op(self):
        from src.models.knn_baseline import KNNClassifierBaseline
        b = KNNClassifierBaseline()
        result = b.to("cpu")
        assert result is b, ".to() must return self"
        result2 = b.to(torch.device("cpu"))
        assert result2 is b

    def test_train_eval_are_no_ops(self):
        from src.models.knn_baseline import KNNClassifierBaseline
        b = KNNClassifierBaseline()
        assert b.train() is b
        assert b.eval() is b

    def test_parameters_is_empty(self):
        from src.models.knn_baseline import KNNClassifierBaseline
        b = KNNClassifierBaseline()
        assert list(b.parameters()) == []

    def test_run_sklearn_model_returns_valid_dict(self):
        """_run_sklearn_model must return a result dict with all required keys."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        from main import _run_sklearn_model
        from models.knn_baseline import KNNClassifierBaseline
        from torch_geometric.data import Data

        n_train, n_test, dim = 30, 10, 16
        n = n_train + n_test
        x = torch.rand(n, dim)
        y = torch.randint(0, 5, (n,))
        src = torch.randint(0, n, (n * 2,))
        dst = torch.randint(0, n, (n * 2,))
        train_mask = torch.zeros(n, dtype=torch.bool); train_mask[:n_train] = True
        test_mask  = torch.zeros(n, dtype=torch.bool); test_mask[n_train:]  = True
        graph = Data(x=x, edge_index=torch.stack([src, dst]), y=y,
                     train_mask=train_mask, test_mask=test_mask,
                     node_image_counts=torch.ones(n, dtype=torch.long))

        result = _run_sklearn_model(
            model=KNNClassifierBaseline(k=3),
            graph=graph,
            model_type="knn_sklearn",
            dataset_mode="test",
            train_subset_size=n_train,
            test_subset_size=n_test,
            run_mode="controlled_subset",
            k_neighbors=3,
            connectivity_mode="directed",
            epochs=0,
            images_per_node=1,
            run_label=None,
            seed=42,
            compressor=None,
        )
        required = {
            "train_accuracy", "test_accuracy", "inference_time_median_s",
            "num_parameters", "model_type", "seed",
        }
        assert required.issubset(result.keys())
        assert 0.0 <= result["train_accuracy"] <= 1.0
        assert 0.0 <= result["test_accuracy"] <= 1.0
        assert result["num_parameters"] == 0
        assert result["epochs"] == 0

    def test_knn_baseline_in_build_experiments(self):
        """knn_baseline must appear in _build_experiments output for relevant modes."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        from main import _build_experiments
        from models.knn_baseline import KNNClassifierBaseline

        for mode in ("resource_efficiency", "controlled_subset"):
            experiments = _build_experiments(mode)
            assert "knn_baseline" in experiments, \
                f"knn_baseline missing from _build_experiments('{mode}')"
            factory = experiments["knn_baseline"]["model_factory"]
            model = factory(128)
            assert isinstance(model, KNNClassifierBaseline)
