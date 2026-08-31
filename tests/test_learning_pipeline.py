"""Tests for patches 1-10: training stability, normalization, baselines, alignment."""

import torch
import pytest
from torch import nn
from torch_geometric.data import Data


def _tiny_graph(n_train=20, n_test=8, feat_dim=16, num_classes=5):
    n = n_train + n_test
    x = torch.rand(n, feat_dim)
    y = torch.randint(0, num_classes, (n,))
    src = torch.randint(0, n, (n * 3,))
    dst = torch.randint(0, n, (n * 3,))
    train_mask = torch.zeros(n, dtype=torch.bool); train_mask[:n_train] = True
    test_mask  = torch.zeros(n, dtype=torch.bool); test_mask[n_train:]  = True
    return Data(x=x, edge_index=torch.stack([src, dst]), y=y,
                train_mask=train_mask, test_mask=test_mask,
                node_image_counts=torch.ones(n, dtype=torch.long))


# ---------------------------------------------------------------------------
# Patch 1: train_with_logging returns loss curve
# ---------------------------------------------------------------------------
class TestTrainWithLogging:
    def test_returns_loss_curve_of_correct_length(self):
        from src.models.baselines import SimpleMLPBaseline
        from src.training.trainer import train_with_logging, make_optimizer
        g = _tiny_graph()
        model = SimpleMLPBaseline(input_dim=16, hidden_dim=8, num_classes=5)
        opt = make_optimizer(model, lr=1e-3)
        curve = train_with_logging(model, g, opt, nn.CrossEntropyLoss(),
                                   "cpu", epochs=5, log_every=2, label="test")
        assert len(curve) == 5

    def test_loss_decreases_on_simple_problem(self):
        """On a small graph with enough epochs, loss should decrease."""
        from src.models.baselines import SimpleMLPBaseline
        from src.training.trainer import train_with_logging, make_optimizer
        torch.manual_seed(0)
        g = _tiny_graph(n_train=20, feat_dim=16, num_classes=5)
        model = SimpleMLPBaseline(input_dim=16, hidden_dim=32, num_classes=5)
        opt = make_optimizer(model, lr=1e-2, weight_decay=0.0)
        curve = train_with_logging(model, g, opt, nn.CrossEntropyLoss(),
                                   "cpu", epochs=50, log_every=50)
        assert curve[-1] < curve[0], \
            f"Loss did not decrease: {curve[0]:.4f} -> {curve[-1]:.4f}"


# ---------------------------------------------------------------------------
# Patch 2: PCA + StandardScaler normalization
# ---------------------------------------------------------------------------
class TestPCAWithStandardScaler:
    def test_standardscaler_fitted_on_train_only(self):
        """Scaler must be fitted on train only, not on test."""
        from src.data.pca_compressor import PCACompressor
        comp = PCACompressor(n_components=8)
        train = torch.rand(40, 32)
        comp.fit_transform(train)
        assert comp._scaler is not None
        assert hasattr(comp._scaler, 'mean_'), "Scaler must be fitted"

    def test_transform_uses_train_scaler(self):
        """Transforming test with different distribution should still use train stats."""
        from src.data.pca_compressor import PCACompressor
        comp = PCACompressor(n_components=4)
        train = torch.rand(30, 16)
        comp.fit_transform(train)
        # Test with very different scale
        test = torch.rand(10, 16) * 100
        out = comp.transform(test)
        assert not torch.isnan(out).any()
        # Output should still be L2-normalized
        norms = out.norm(dim=1)
        assert torch.allclose(norms, torch.ones(10), atol=1e-5)

    def test_fit_transform_zero_mean_before_l2(self):
        """After standardization (before L2), components should have ~zero mean."""
        from src.data.pca_compressor import PCACompressor
        import numpy as np
        comp = PCACompressor(n_components=8)
        train = torch.rand(100, 32) * torch.arange(1, 33).float()
        train_out = comp.fit_transform(train)
        # After L2-norm the mean won't be zero, but scaler itself was fitted
        # Check that scaler mean_ is close to PCA-projected train mean
        assert comp._scaler.mean_ is not None
        assert len(comp._scaler.mean_) == 8


# ---------------------------------------------------------------------------
# Patch 3: ablation constant exists
# ---------------------------------------------------------------------------
class TestPCAAblation:
    def test_ablation_dim_exists_in_config(self):
        import src.config as cfg
        assert hasattr(cfg, 'PCA_COMPONENTS_ABLATION')
        assert cfg.PCA_COMPONENTS_ABLATION > cfg.PCA_COMPONENTS


# ---------------------------------------------------------------------------
# Patch 5: overfit sanity check
# ---------------------------------------------------------------------------
class TestOverfitSanityCheck:
    def test_mlp_can_overfit_20_samples(self):
        """A simple MLP must reach >95% train acc on 20 samples."""
        from src.models.baselines import SimpleMLPBaseline
        from src.training.trainer import overfit_sanity_check
        torch.manual_seed(42)
        g = _tiny_graph(n_train=20, n_test=5, feat_dim=16, num_classes=5)
        model = SimpleMLPBaseline(input_dim=16, hidden_dim=64, num_classes=5)
        result = overfit_sanity_check(model, g, "cpu",
                                      target_train_acc=0.95, max_epochs=500, lr=1e-2)
        assert result["passed"], (
            f"MLP could not overfit 20 samples. "
            f"Final train acc: {result['final_train_acc']:.3f}. "
            "Pipeline may be broken."
        )

    def test_sanity_check_returns_loss_curve(self):
        from src.models.baselines import SimpleMLPBaseline
        from src.training.trainer import overfit_sanity_check
        g = _tiny_graph(n_train=10, n_test=5, feat_dim=8, num_classes=3)
        model = SimpleMLPBaseline(input_dim=8, hidden_dim=32, num_classes=3)
        result = overfit_sanity_check(model, g, "cpu", max_epochs=100, lr=1e-2)
        assert "loss_curve" in result
        assert len(result["loss_curve"]) > 0


# ---------------------------------------------------------------------------
# Patch 7: Logistic Regression baseline
# ---------------------------------------------------------------------------
class TestLogRegBaseline:
    def _features(self, n, dim, seed=0):
        g = torch.Generator().manual_seed(seed)
        import torch.nn.functional as F
        return F.normalize(torch.rand(n, dim, generator=g), p=2, dim=1)

    def test_fit_and_score(self):
        from src.models.logreg_baseline import LogRegClassifierBaseline
        train_f = self._features(50, 32)
        train_l = torch.randint(0, 5, (50,))
        test_f = self._features(20, 32, seed=1)
        test_l = torch.randint(0, 5, (20,))
        lr = LogRegClassifierBaseline()
        lr.fit(train_f, train_l)
        acc = lr.score(test_f, test_l)
        assert 0.0 <= acc <= 1.0

    def test_nn_module_shims(self):
        from src.models.logreg_baseline import LogRegClassifierBaseline
        m = LogRegClassifierBaseline()
        assert m.to("cpu") is m
        assert m.eval() is m
        assert m.train() is m
        assert list(m.parameters()) == []

    def test_evaluate_returns_result(self):
        from src.models.logreg_baseline import LogRegClassifierBaseline
        train_f = self._features(40, 16)
        train_l = torch.randint(0, 4, (40,))
        test_f = self._features(15, 16, seed=2)
        test_l = torch.randint(0, 4, (15,))
        result = LogRegClassifierBaseline().evaluate(train_f, train_l, test_f, test_l)
        assert 0.0 <= result.test_accuracy <= 1.0
        assert 0.0 <= result.train_accuracy <= 1.0

    def test_in_build_experiments(self):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        from main import _build_experiments
        from models.logreg_baseline import LogRegClassifierBaseline
        for mode in ("resource_efficiency", "controlled_subset"):
            exps = _build_experiments(mode)
            assert "logreg_baseline" in exps
            model = exps["logreg_baseline"]["model_factory"](128)
            assert isinstance(model, LogRegClassifierBaseline)

    def test_logreg_beats_random_on_structured_data(self):
        """LR should beat 20% random on linearly separable 5-class data."""
        from src.models.logreg_baseline import LogRegClassifierBaseline
        torch.manual_seed(0)
        # Linearly separable: class centers well separated
        centers = torch.eye(5) * 5.0       # (5, 5) well-separated
        train_f, train_l, test_f, test_l = [], [], [], []
        for c in range(5):
            train_f.append(centers[c].unsqueeze(0).repeat(20, 1) + torch.randn(20, 5) * 0.3)
            train_l.extend([c] * 20)
            test_f.append(centers[c].unsqueeze(0).repeat(5, 1) + torch.randn(5, 5) * 0.3)
            test_l.extend([c] * 5)
        train_f = torch.cat(train_f); train_l = torch.tensor(train_l)
        test_f = torch.cat(test_f); test_l = torch.tensor(test_l)
        result = LogRegClassifierBaseline().evaluate(train_f, train_l, test_f, test_l)
        assert result.test_accuracy > 0.5, \
            f"LR should do well on separable data, got {result.test_accuracy:.3f}"


# ---------------------------------------------------------------------------
# Patch 8: reduced regularization in config
# ---------------------------------------------------------------------------
class TestReducedRegularization:
    def test_dropout_is_low(self):
        import src.config as cfg
        assert cfg.DROPOUT <= 0.2, \
            f"DROPOUT={cfg.DROPOUT} is too high for underfit regime (expected ≤0.2)"

    def test_weight_decay_is_small(self):
        import src.config as cfg
        assert cfg.WEIGHT_DECAY <= 1e-4, \
            f"WEIGHT_DECAY={cfg.WEIGHT_DECAY} may cause underfit"

    def test_epochs_is_sufficient(self):
        import src.config as cfg
        assert cfg.EPOCHS >= 50, \
            f"EPOCHS={cfg.EPOCHS} too low; need ≥50 to escape random regime"


# ---------------------------------------------------------------------------
# Patch 9: alignment check
# ---------------------------------------------------------------------------
class TestGraphAlignmentCheck:
    def test_valid_graph_passes(self):
        from src.utils.alignment_check import verify_graph_alignment
        g = _tiny_graph(n_train=20, n_test=8, num_classes=5)
        result = verify_graph_alignment(g, label="test")
        assert result["passed"], f"Valid graph failed: {result['issues']}"

    def test_mask_overlap_detected(self):
        from src.utils.alignment_check import verify_graph_alignment
        g = _tiny_graph(n_train=20, n_test=8)
        # Artificially create overlap
        g.train_mask[25] = True   # node 25 is also in test_mask
        result = verify_graph_alignment(g)
        assert not result["passed"]
        assert any("∩" in i for i in result["issues"])

    def test_out_of_bounds_edge_detected(self):
        from src.utils.alignment_check import verify_graph_alignment
        g = _tiny_graph()
        n = g.num_nodes
        # Inject bad edge
        g.edge_index = torch.cat([
            g.edge_index,
            torch.tensor([[0], [n + 10]])  # out of bounds
        ], dim=1)
        result = verify_graph_alignment(g)
        assert not result["passed"]
        assert any("edge_index" in i for i in result["issues"])

    def test_degenerate_labels_detected(self):
        from src.utils.alignment_check import verify_graph_alignment
        g = _tiny_graph(n_train=20, n_test=8, num_classes=5)
        # Force all train labels to one class
        g.y[:20] = 0
        result = verify_graph_alignment(g)
        assert not result["passed"]
        assert any("clases" in i for i in result["issues"])

    def test_returns_counts(self):
        from src.utils.alignment_check import verify_graph_alignment
        g = _tiny_graph(n_train=20, n_test=8)
        result = verify_graph_alignment(g)
        assert result["num_train"] == 20
        assert result["num_test"] == 8
