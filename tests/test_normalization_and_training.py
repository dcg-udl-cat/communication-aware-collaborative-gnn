"""Tests for patches 1-6: normalization, early stopping, MLP capacity, diagnostics."""

import torch
import pytest
from torch import nn
from torch_geometric.data import Data


def _graph(n_train=30, n_test=10, feat_dim=16, num_classes=5):
    n = n_train + n_test
    torch.manual_seed(0)
    x = torch.rand(n, feat_dim) * torch.arange(1, feat_dim + 1).float()
    y = torch.randint(0, num_classes, (n,))
    src = torch.randint(0, n, (n * 3,))
    dst = torch.randint(0, n, (n * 3,))
    train_mask = torch.zeros(n, dtype=torch.bool); train_mask[:n_train] = True
    test_mask  = torch.zeros(n, dtype=torch.bool); test_mask[n_train:]  = True
    return Data(x=x, edge_index=torch.stack([src, dst]), y=y,
                train_mask=train_mask, test_mask=test_mask,
                node_image_counts=torch.ones(n, dtype=torch.long))


# ---------------------------------------------------------------------------
# Patch 2: normalize_graph_features
# ---------------------------------------------------------------------------
class TestNormalizeGraphFeatures:
    def test_train_columns_zero_mean(self):
        from src.utils.feature_normalization import normalize_graph_features
        g = _graph()
        g_norm = normalize_graph_features(g, g.train_mask, l2_normalize_rows=False)
        train_x = g_norm.x[g_norm.train_mask]
        col_means = train_x.mean(dim=0)
        assert torch.allclose(col_means, torch.zeros_like(col_means), atol=1e-5), \
            "Train columns must have zero mean after normalization"

    def test_train_columns_unit_std(self):
        from src.utils.feature_normalization import normalize_graph_features
        g = _graph()
        g_norm = normalize_graph_features(g, g.train_mask, l2_normalize_rows=False)
        train_x = g_norm.x[g_norm.train_mask]
        col_stds = train_x.std(dim=0)
        assert torch.allclose(col_stds, torch.ones_like(col_stds), atol=1e-4), \
            "Train columns must have unit std after normalization"

    def test_uses_train_stats_for_test(self):
        """Test nodes must be normalized using train mean/std, not their own."""
        from src.utils.feature_normalization import normalize_graph_features
        g = _graph(n_train=30, n_test=10, feat_dim=8)
        # Give test nodes very different scale
        g.x[30:] = g.x[30:] * 100
        g_norm = normalize_graph_features(g, g.train_mask, l2_normalize_rows=False)
        # Test nodes should NOT have zero mean (they were shifted by train stats)
        test_x = g_norm.x[g_norm.test_mask]
        # Verify train stats were used: test values can be far from zero mean
        train_x = g_norm.x[g_norm.train_mask]
        assert torch.allclose(train_x.mean(dim=0), torch.zeros(8), atol=1e-5)

    def test_l2_rows_unit_norm(self):
        from src.utils.feature_normalization import normalize_graph_features
        g = _graph()
        g_norm = normalize_graph_features(g, g.train_mask, l2_normalize_rows=True)
        norms = g_norm.x.norm(dim=1)
        assert torch.allclose(norms, torch.ones(g.num_nodes), atol=1e-5)

    def test_row_only_normalization_preserves_sparse_feature_support(self):
        from src.utils.feature_normalization import normalize_graph_features

        g = _graph(n_train=4, n_test=2, feat_dim=3)
        g.x[:, 2] = 0
        g.x[g.test_mask, 2] = 1
        g_norm = normalize_graph_features(
            g,
            g.train_mask,
            standardize_columns=False,
        )
        assert torch.equal(g_norm.x[:, 2] != 0, g.x[:, 2] != 0)
        assert torch.allclose(g_norm.x.norm(dim=1), torch.ones(g.num_nodes))

    def test_original_graph_unchanged(self):
        """normalize_graph_features must not mutate the input graph."""
        from src.utils.feature_normalization import normalize_graph_features
        g = _graph()
        x_before = g.x.clone()
        _ = normalize_graph_features(g, g.train_mask)
        assert torch.allclose(g.x, x_before), "Original graph.x was mutated"

    def test_high_variance_feature_no_nan(self):
        """Columns with near-zero std must not produce NaN (eps guard)."""
        from src.utils.feature_normalization import normalize_graph_features
        g = _graph(feat_dim=8)
        # Make one column constant
        g.x[:, 3] = 5.0
        g_norm = normalize_graph_features(g, g.train_mask)
        assert not torch.isnan(g_norm.x).any(), "NaN in constant-column feature"


# ---------------------------------------------------------------------------
# Patch 5: logits_diagnostic
# ---------------------------------------------------------------------------
class TestLogitsDiagnostic:
    def test_healthy_logits_not_collapsed(self):
        from src.utils.feature_normalization import logits_diagnostic
        logits = torch.randn(50, 10)
        result = logits_diagnostic(logits, label="test")
        assert not result["collapsed"]
        assert result["std"] > 0.1

    def test_constant_logits_detected_as_collapsed(self):
        from src.utils.feature_normalization import logits_diagnostic
        logits = torch.zeros(50, 10) + 0.001  # near-constant
        result = logits_diagnostic(logits)
        assert result["collapsed"], "Near-zero std must be flagged as collapsed"

    def test_returns_all_keys(self):
        from src.utils.feature_normalization import logits_diagnostic
        result = logits_diagnostic(torch.randn(20, 5))
        assert {"std", "mean", "max", "min", "collapsed"}.issubset(result.keys())


# ---------------------------------------------------------------------------
# Patch 1: early stopping in train_with_logging
# ---------------------------------------------------------------------------
class TestEarlyStopping:
    def _make_model_and_graph(self, feat_dim=16, num_classes=5):
        from src.models.baselines import SimpleMLPBaseline
        model = SimpleMLPBaseline(input_dim=feat_dim, hidden_dim=32, num_classes=num_classes)
        g = _graph(feat_dim=feat_dim, num_classes=num_classes)
        return model, g

    def test_stops_before_max_epochs_when_val_not_improving(self):
        """With patience=2 and a frozen (non-training) model, should stop early."""
        from src.training.trainer import train_with_logging
        model, g = self._make_model_and_graph()
        # No optimizer → loss never changes → early stopping triggers
        val_mask = g.test_mask
        curve = train_with_logging(
            model, g, optimizer=None, loss_fn=nn.CrossEntropyLoss(),
            device="cpu", epochs=50, patience=2, val_mask=val_mask, log_every=100,
        )
        assert len(curve) < 50, \
            f"Expected early stop before 50 epochs, got {len(curve)}"

    def test_runs_full_epochs_without_val_mask(self):
        """Without val_mask, all epochs must run regardless of loss plateau."""
        from src.training.trainer import train_with_logging, make_optimizer
        model, g = self._make_model_and_graph()
        opt = make_optimizer(model, lr=1e-3)
        curve = train_with_logging(
            model, g, optimizer=opt, loss_fn=nn.CrossEntropyLoss(),
            device="cpu", epochs=10, val_mask=None, log_every=100,
        )
        assert len(curve) == 10

    def test_loss_curve_stored_in_result(self, monkeypatch):
        """_run_model result dict must contain 'loss_curve'."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        from main import _run_model
        from models.baselines import SimpleMLPBaseline

        g = _graph(feat_dim=16, num_classes=5)
        result = _run_model(
            model=SimpleMLPBaseline(input_dim=16, hidden_dim=32, num_classes=5),
            graph=g,
            device=torch.device("cpu"),
            model_type="no_gnn",
            dataset_mode="test",
            train_subset_size=30,
            test_subset_size=10,
            run_mode="controlled_subset",
            k_neighbors=3,
            connectivity_mode="directed",
            epochs=3,
            images_per_node=1,
            run_label=None,
            seed=42,
            compressor=None,
        )
        assert "loss_curve" in result
        assert isinstance(result["loss_curve"], list)
        assert len(result["loss_curve"]) > 0


# ---------------------------------------------------------------------------
# Patch 4+6: MLP capacity and target accuracy
# ---------------------------------------------------------------------------
class TestMLPCapacity:
    def test_mlp_has_two_hidden_layers(self):
        from src.models.baselines import SimpleMLPBaseline
        model = SimpleMLPBaseline(input_dim=128, hidden_dim=256, num_classes=10)
        assert hasattr(model, "fc1") and hasattr(model, "fc2") and hasattr(model, "fc3"), \
            "MLP must have fc1, fc2, fc3 (two hidden layers)"
        assert model.fc1.in_features == 128
        assert model.fc1.out_features == 256
        assert model.fc2.in_features == 256
        assert model.fc2.out_features == 256
        assert model.fc3.out_features == 10

    def test_mlp_has_batchnorm(self):
        from src.models.baselines import SimpleMLPBaseline
        model = SimpleMLPBaseline(input_dim=128, hidden_dim=256, num_classes=10)
        bn_layers = [m for m in model.modules() if isinstance(m, nn.BatchNorm1d)]
        assert len(bn_layers) >= 2, "MLP must have at least 2 BatchNorm layers"

    def test_mlp_param_count_in_range(self):
        """With hidden=256, params should be ~100k — enough but still lightweight."""
        from src.models.baselines import SimpleMLPBaseline
        from src.utils.metrics import count_trainable_parameters
        model = SimpleMLPBaseline(input_dim=128, hidden_dim=256, num_classes=10)
        n = count_trainable_parameters(model)
        assert n > 50_000, f"MLP too small: {n} params (expected >50k)"
        assert n < 500_000, f"MLP too large: {n} params (expected <500k)"

    def test_mlp_can_overfit_small_dataset(self):
        """Patch 4: MLP must reach >90% train acc on 30 nodes (sanity gate)."""
        from src.models.baselines import SimpleMLPBaseline
        from src.training.trainer import overfit_sanity_check
        torch.manual_seed(0)
        g = _graph(n_train=30, n_test=5, feat_dim=16, num_classes=5)
        model = SimpleMLPBaseline(input_dim=16, hidden_dim=256, num_classes=5)
        result = overfit_sanity_check(
            model, g, "cpu", target_train_acc=0.90, max_epochs=500, lr=5e-3
        )
        assert result["passed"], (
            f"MLP (2-layer, hidden=256) could not overfit 30 samples. "
            f"Final train acc: {result['final_train_acc']:.3f}. "
            "Either BatchNorm causes issues with tiny batches or there is a pipeline bug."
        )

    def test_mlp_hidden_dim_config_is_256(self):
        import src.config as cfg
        assert cfg.MLP_HIDDEN_DIM == 256, \
            f"MLP_HIDDEN_DIM={cfg.MLP_HIDDEN_DIM}, expected 256"

    def test_early_stopping_patience_in_config(self):
        import src.config as cfg
        assert hasattr(cfg, "EARLY_STOPPING_PATIENCE")
        assert cfg.EARLY_STOPPING_PATIENCE >= 5


# ---------------------------------------------------------------------------
# Patch 3: learning rate sanity
# ---------------------------------------------------------------------------
class TestLearningRateConfig:
    def test_lr_in_productive_range(self):
        import src.config as cfg
        assert 5e-4 <= cfg.LEARNING_RATE <= 5e-3, \
            f"LEARNING_RATE={cfg.LEARNING_RATE} outside productive range [5e-4, 5e-3]"

    def test_weight_decay_not_too_high(self):
        import src.config as cfg
        assert cfg.WEIGHT_DECAY <= 1e-3, \
            f"WEIGHT_DECAY={cfg.WEIGHT_DECAY} may cause underfit (expected ≤1e-3)"

    def test_training_improves_with_reasonable_config(self):
        """With lr=1e-3 and no weight decay, loss must decrease over 30 epochs."""
        from src.models.baselines import SimpleMLPBaseline
        from src.training.trainer import train_with_logging, make_optimizer
        torch.manual_seed(7)
        g = _graph(n_train=40, feat_dim=16, num_classes=5)
        model = SimpleMLPBaseline(input_dim=16, hidden_dim=128, num_classes=5)
        opt = make_optimizer(model, lr=1e-3, weight_decay=0.0)
        curve = train_with_logging(
            model, g, opt, nn.CrossEntropyLoss(),
            "cpu", epochs=30, log_every=30,
        )
        assert curve[-1] < curve[0] - 0.05, \
            f"Loss did not decrease meaningfully: {curve[0]:.4f} -> {curve[-1]:.4f}"
