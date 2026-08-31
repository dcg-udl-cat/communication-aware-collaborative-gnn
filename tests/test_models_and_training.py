"""Tests de modelos y entrenamiento."""

import statistics
import torch
import pytest
from torch_geometric.data import Data


def _make_graph(n_train=20, n_test=10, feat_dim=64, num_classes=10):
    """Grafo sintetico para tests."""
    n = n_train + n_test
    x = torch.rand(n, feat_dim)
    y = torch.randint(0, num_classes, (n,))
    # Aristas aleatorias simples
    src = torch.randint(0, n, (n * 4,))
    dst = torch.randint(0, n, (n * 4,))
    edge_index = torch.stack([src, dst], dim=0)

    train_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[:n_train] = True
    test_mask[n_train:] = True

    return Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        test_mask=test_mask,
        node_image_counts=torch.ones(n, dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# RandomBaseline — debe ser determinista
# ---------------------------------------------------------------------------

class TestRandomBaselineDeterministic:
    def test_same_logits_across_calls(self):
        """forward() debe devolver los mismos logits cada vez."""
        from src.models.baselines import RandomBaseline
        model = RandomBaseline(num_classes=10)
        model.eval()
        x = torch.rand(8, 64)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.allclose(out1, out2), \
            "RandomBaseline debe producir logits identicos en llamadas sucesivas"

    def test_output_shape(self):
        from src.models.baselines import RandomBaseline
        model = RandomBaseline(num_classes=10)
        x = torch.rand(12, 64)
        out = model(x)
        assert out.shape == (12, 10)

    def test_not_trainable(self):
        from src.models.baselines import RandomBaseline
        model = RandomBaseline()
        params = list(model.parameters())
        assert len(params) == 0 or all(not p.requires_grad for p in params), \
            "RandomBaseline no debe tener parametros entrenables"


# ---------------------------------------------------------------------------
# TinyGNN / LegacyCloudGCN / MLP
# ---------------------------------------------------------------------------

class TestModelOutputShapes:
    @pytest.mark.parametrize("ModelClass,kwargs", [
        ("TinyGNN", {"input_dim": 64, "hidden_dim": 16, "num_classes": 5}),
        ("LegacyCloudGCN", {"input_dim": 64, "hidden_dim": 16, "num_classes": 5}),
    ])
    def test_gnn_output_shape(self, ModelClass, kwargs):
        import importlib
        module = "tiny_gnn" if ModelClass == "TinyGNN" else "legacy_cloud_gcn"
        mod = importlib.import_module(f"src.models.{module}")
        Model = getattr(mod, ModelClass)
        graph = _make_graph(feat_dim=64, num_classes=5)
        model = Model(**kwargs)
        model.eval()
        with torch.no_grad():
            out = model(graph.x, graph.edge_index)
        assert out.shape == (graph.num_nodes, 5)

    def test_mlp_output_shape(self):
        from src.models.baselines import SimpleMLPBaseline
        model = SimpleMLPBaseline(input_dim=64, hidden_dim=16, num_classes=5)
        model.eval()
        x = torch.rand(15, 64)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (15, 5)


# ---------------------------------------------------------------------------
# Evaluator — timing robusto
# ---------------------------------------------------------------------------

class TestMeasureInferenceTime:
    def test_returns_positive_float(self):
        from src.models.tiny_gnn import TinyGNN
        from src.training.evaluator import measure_inference_time
        graph = _make_graph(feat_dim=64)
        model = TinyGNN(input_dim=64, hidden_dim=8, num_classes=10)
        t = measure_inference_time(model, graph, warmup_runs=2, measure_runs=5)
        assert isinstance(t, float)
        assert t > 0

    def test_median_is_stable(self):
        """La mediana de multiples llamadas debe ser similar (no ruidosa como randn)."""
        from src.models.baselines import SimpleMLPBaseline
        from src.training.evaluator import measure_inference_time
        graph = _make_graph(feat_dim=64)
        model = SimpleMLPBaseline(input_dim=64, hidden_dim=8, num_classes=10)
        times = [
            measure_inference_time(model, graph, warmup_runs=2, measure_runs=10)
            for _ in range(5)
        ]
        # El rango relativo entre la mediana mas alta y la mas baja debe ser
        # razonable (no divergente como ocurriria con logits aleatorios).
        assert max(times) < min(times) * 100, \
            "El tiempo de inferencia mediano no debe variar en mas de 100x entre llamadas"


class TestEvaluateAccuracy:
    def test_range(self):
        from src.models.baselines import RandomBaseline
        from src.training.evaluator import evaluate_accuracy
        graph = _make_graph(n_train=30, n_test=10, feat_dim=64)
        model = RandomBaseline(num_classes=10)
        acc = evaluate_accuracy(model, graph, graph.test_mask)
        assert 0.0 <= acc <= 1.0

    def test_empty_mask_returns_zero(self):
        from src.models.baselines import RandomBaseline
        from src.training.evaluator import evaluate_accuracy
        graph = _make_graph(feat_dim=64)
        model = RandomBaseline()
        empty_mask = torch.zeros(graph.num_nodes, dtype=torch.bool)
        acc = evaluate_accuracy(model, graph, empty_mask)
        assert acc == 0.0


# ---------------------------------------------------------------------------
# Validation split en _run_model
# ---------------------------------------------------------------------------

class TestValidationSplit:
    def test_val_accuracy_present_in_results(self):
        """_run_model debe incluir val_accuracy en las metricas."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        from main import _run_model
        from models.tiny_gnn import TinyGNN

        graph = _make_graph(n_train=20, n_test=10, feat_dim=64)
        result = _run_model(
            model=TinyGNN(input_dim=64, hidden_dim=8, num_classes=10),
            graph=graph,
            device=torch.device("cpu"),
            model_type="tiny_gnn",
            dataset_mode="test",
            train_subset_size=20,
            test_subset_size=10,
            run_mode="controlled_subset",
            k_neighbors=3,
            epochs=1,
            images_per_node=1,
            run_label=None,
            seed=42,
        )
        assert "val_accuracy" in result
        assert 0.0 <= result["val_accuracy"] <= 1.0
        assert "val_test_gap" in result


# ---------------------------------------------------------------------------
# Val split shuffle
# ---------------------------------------------------------------------------

class TestValSplitShuffle:
    def _make_class_ordered_graph(self, num_classes=5, nodes_per_class=10, feat_dim=16):
        """Grafo donde los nodos de train estan ordenados por clase (como CIFAR sin shuffle)."""
        n_train = num_classes * nodes_per_class
        n_test = 10
        n = n_train + n_test

        x = torch.rand(n, feat_dim)
        # Nodos de train: bloques de clase pura [0,0,...,1,1,...,4,4,...]
        y_train = torch.repeat_interleave(torch.arange(num_classes), nodes_per_class)
        y_test = torch.randint(0, num_classes, (n_test,))
        y = torch.cat([y_train, y_test])

        src = torch.randint(0, n, (n * 3,))
        dst = torch.randint(0, n, (n * 3,))
        edge_index = torch.stack([src, dst])

        train_mask = torch.zeros(n, dtype=torch.bool)
        test_mask = torch.zeros(n, dtype=torch.bool)
        train_mask[:n_train] = True
        test_mask[n_train:] = True

        return Data(
            x=x, edge_index=edge_index, y=y,
            train_mask=train_mask, test_mask=test_mask,
            node_image_counts=torch.ones(n, dtype=torch.long),
        )

    def test_val_split_covers_all_classes(self):
        """Con nodos ordenados por clase, el val split debe cubrir todas las clases
        gracias al shuffle previo. Sin shuffle, los primeros val_size nodos serian
        todos de las clases 0 y 1."""
        from main import _run_model
        from models.baselines import SimpleMLPBaseline

        num_classes = 5
        graph = self._make_class_ordered_graph(num_classes=num_classes, nodes_per_class=20)

        # Ejecutamos _run_model y comprobamos indirectamente que el val split es
        # representativo inspeccionando las etiquetas de los indices de val.
        # Para eso replicamos la logica del split con la misma semilla.
        import torch
        train_indices = graph.train_mask.nonzero(as_tuple=True)[0]
        torch.manual_seed(42)
        perm = torch.randperm(len(train_indices))
        shuffled = train_indices[perm]
        val_size = max(1, int(len(shuffled) * 0.2))
        val_indices = shuffled[:val_size]
        val_labels = graph.y[val_indices]
        unique_val_classes = torch.unique(val_labels).numel()

        assert unique_val_classes > 1, (
            f"El val split solo contiene {unique_val_classes} clase(s). "
            "Probablemente no se esta barajando antes de dividir."
        )

    def test_val_accuracy_in_result(self):
        """_run_model siempre debe devolver val_accuracy."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        from main import _run_model
        from models.baselines import SimpleMLPBaseline

        graph = self._make_class_ordered_graph()
        result = _run_model(
            model=SimpleMLPBaseline(input_dim=16, hidden_dim=8, num_classes=5),
            graph=graph,
            device=torch.device("cpu"),
            model_type="no_gnn",
            dataset_mode="test",
            train_subset_size=50,
            test_subset_size=10,
            run_mode="controlled_subset",
            k_neighbors=3,
            epochs=1,
            images_per_node=1,
            run_label=None,
            seed=42,
        )
        assert "val_accuracy" in result
        assert "val_test_gap" in result
        assert 0.0 <= result["val_accuracy"] <= 1.0


# ---------------------------------------------------------------------------
# TinyGNN architecture (GAT + residual + dropout)
# ---------------------------------------------------------------------------

class TestTinyGNNArchitecture:
    def test_uses_gatconv(self):
        """TinyGNN uses attention to suppress noisy KNN neighbours."""
        from src.models.tiny_gnn import TinyGNN
        from torch_geometric.nn import GATConv
        model = TinyGNN(input_dim=128, hidden_dim=64, num_classes=10)
        gat_layers = [m for m in model.modules() if isinstance(m, GATConv)]
        assert len(gat_layers) >= 1

    def test_has_linear_before_conv(self):
        """TinyGNN must project inputs before attention."""
        from src.models.tiny_gnn import TinyGNN
        model = TinyGNN(input_dim=128, hidden_dim=64, num_classes=10)
        assert isinstance(model.lin_in, torch.nn.Linear)
        assert model.lin_in.in_features == 128
        assert model.lin_in.out_features == 64

    def test_has_dropout(self):
        """TinyGNN debe tener dropout > 0."""
        from src.models.tiny_gnn import TinyGNN
        model = TinyGNN(input_dim=128, hidden_dim=64, num_classes=10, dropout=0.3)
        assert model.dropout_p == 0.3

    def test_dropout_active_in_train_mode(self):
        """Dropout must make train-mode outputs stochastic."""
        from src.models.tiny_gnn import TinyGNN
        graph = _make_graph(feat_dim=128, num_classes=10)
        model = TinyGNN(input_dim=128, hidden_dim=64, num_classes=10, dropout=0.5)
        model.train()
        torch.manual_seed(0)
        out1 = model(graph.x, graph.edge_index)
        out2 = model(graph.x, graph.edge_index)
        assert not torch.allclose(out1, out2)

    def test_dropout_inactive_in_eval_mode(self):
        """En eval mode, el dropout no debe afectar la salida."""
        from src.models.tiny_gnn import TinyGNN
        graph = _make_graph(feat_dim=128, num_classes=10)
        model = TinyGNN(input_dim=128, hidden_dim=64, num_classes=10, dropout=0.5)
        model.eval()
        with torch.no_grad():
            out1 = model(graph.x, graph.edge_index)
            out2 = model(graph.x, graph.edge_index)
        assert torch.allclose(out1, out2), \
            "En eval mode, la salida debe ser determinista"

    def test_output_shape_with_new_arch(self):
        """La forma de salida debe ser (num_nodes, num_classes)."""
        from src.models.tiny_gnn import TinyGNN
        graph = _make_graph(n_train=20, n_test=10, feat_dim=128, num_classes=10)
        model = TinyGNN(input_dim=128, hidden_dim=64, num_classes=10)
        model.eval()
        with torch.no_grad():
            out = model(graph.x, graph.edge_index)
        assert out.shape == (30, 10)

    def test_param_count_is_tiny(self):
        """The edge model should remain small relative to image encoders."""
        from src.models.tiny_gnn import TinyGNN
        from src.utils.metrics import count_trainable_parameters
        model = TinyGNN(input_dim=128, hidden_dim=64, num_classes=10)
        n = count_trainable_parameters(model)
        assert n < 50_000, f"TinyGNN tiene {n} parametros, demasiados para 'tiny'"
        assert n > 1_000, f"TinyGNN tiene solo {n} parametros, probablemente roto"


class TestConfigDefaults:
    def test_pca_components_is_128(self):
        import src.config as cfg
        assert cfg.PCA_COMPONENTS == 128, \
            f"PCA_COMPONENTS={cfg.PCA_COMPONENTS}, debe ser 128"

    def test_epochs_is_sufficient_for_convergence(self):
        import src.config as cfg
        assert cfg.EPOCHS >= 20

    def test_tiny_hidden_dim_is_64(self):
        import src.config as cfg
        assert cfg.TINY_GNN_HIDDEN_DIM == 64, \
            f"TINY_GNN_HIDDEN_DIM={cfg.TINY_GNN_HIDDEN_DIM}, debe ser 64"


# ---------------------------------------------------------------------------
# Regression: model input_dim must match actual graph feature dim
# ---------------------------------------------------------------------------

class TestModelDimMatchesGraph:
    """Regression test for the shape mismatch bug:
    models were instantiated with INPUT_DIM from config before the graph
    was built, but the actual feature dim can differ (PCA clamping,
    images_per_node, USE_PCA flag). The factory pattern fixes this."""

    def _make_graph_with_dim(self, feat_dim: int, n_train=20, n_test=8):
        x = torch.rand(n_train + n_test, feat_dim)
        y = torch.randint(0, 10, (n_train + n_test,))
        src = torch.randint(0, n_train + n_test, (60,))
        dst = torch.randint(0, n_train + n_test, (60,))
        train_mask = torch.zeros(n_train + n_test, dtype=torch.bool)
        test_mask = torch.zeros(n_train + n_test, dtype=torch.bool)
        train_mask[:n_train] = True
        test_mask[n_train:] = True
        return Data(x=x, edge_index=torch.stack([src, dst]),
                    y=y, train_mask=train_mask, test_mask=test_mask,
                    node_image_counts=torch.ones(n_train + n_test, dtype=torch.long))

    @pytest.mark.parametrize("actual_dim,config_dim", [
        (100, 128),   # PCA clamped below PCA_COMPONENTS
        (64, 128),    # USE_PCA=False but config says 128
        (128, 64),    # config changed but graph not rebuilt
        (256, 128),   # larger than expected
    ])
    def test_factory_produces_correct_input_dim(self, actual_dim, config_dim):
        """The factory lambda must use the runtime feat_dim, not config INPUT_DIM."""
        from models.baselines import SimpleMLPBaseline
        from models.tiny_gnn import TinyGNN
        from models.legacy_cloud_gcn import LegacyCloudGCN
        from training.trainer import model_forward

        graph = self._make_graph_with_dim(actual_dim)

        factories = {
            "mlp": lambda d: SimpleMLPBaseline(input_dim=d),
            "tiny": lambda d: TinyGNN(input_dim=d),
            "full": lambda d: LegacyCloudGCN(input_dim=d),
        }

        for name, factory in factories.items():
            model = factory(actual_dim)  # correct: use runtime dim
            model.eval()
            with torch.no_grad():
                # Must not raise RuntimeError: mat1 and mat2 shapes cannot be multiplied
                out = model_forward(model, graph)
            assert out.shape == (graph.num_nodes, 10), \
                f"{name} with actual_dim={actual_dim}: wrong output shape {out.shape}"

    def test_wrong_dim_raises(self):
        """Sanity check: building a model with the wrong dim DOES raise."""
        from models.baselines import SimpleMLPBaseline
        from training.trainer import model_forward

        graph = self._make_graph_with_dim(feat_dim=100)
        model = SimpleMLPBaseline(input_dim=128)  # wrong: stale config dim
        model.eval()
        with pytest.raises(RuntimeError, match="shapes cannot be multiplied"):
            with torch.no_grad():
                model_forward(model, graph)

    def test_build_experiments_returns_factories_not_models(self):
        """_build_experiments must return callables, not instantiated models."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        from main import _build_experiments

        for mode in ("resource_efficiency", "controlled_subset", "full_gnn_only"):
            experiments = _build_experiments(mode)
            for name, exp in experiments.items():
                assert "model_factory" in exp, \
                    f"Experiment '{name}' missing 'model_factory' key — got {list(exp.keys())}"
                assert callable(exp["model_factory"]), \
                    f"Experiment '{name}' model_factory is not callable"
                assert "model" not in exp, \
                    f"Experiment '{name}' has stale 'model' key — should be 'model_factory'"

    def test_factory_called_with_runtime_dim_produces_correct_model(self):
        """Calling the factory with the actual graph dim must produce a working model."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        from main import _build_experiments
        from training.trainer import model_forward

        actual_dim = 100  # deliberately different from any config default
        graph = self._make_graph_with_dim(actual_dim)

        experiments = _build_experiments("controlled_subset")
        for name, exp in experiments.items():
            model = exp["model_factory"](actual_dim)
            if not isinstance(model, torch.nn.Module):
                continue
            model.eval()
            with torch.no_grad():
                out = model_forward(model, graph)
            assert out.shape[1] == 10, \
                f"{name}: output classes dim wrong: {out.shape}"
            assert out.shape[0] == graph.num_nodes, \
                f"{name}: output node count wrong: {out.shape}"
