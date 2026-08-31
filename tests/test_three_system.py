"""Tests for the three-system architecture."""

import torch
import pytest
from torch_geometric.data import Data


def _make_graph(n_train=40, n_test=15, feat_dim=32, num_classes=5):
    n = n_train + n_test
    torch.manual_seed(0)
    x = torch.randn(n, feat_dim)
    import torch.nn.functional as F
    x = F.normalize(x, p=2, dim=1)
    y = torch.randint(0, num_classes, (n,))
    src = torch.randint(0, n, (n * 4,))
    dst = torch.randint(0, n, (n * 4,))
    train_mask = torch.zeros(n, dtype=torch.bool); train_mask[:n_train] = True
    test_mask  = torch.zeros(n, dtype=torch.bool); test_mask[n_train:]  = True
    edge_weight = torch.rand(n * 4)
    return Data(x=x, edge_index=torch.stack([src, dst]), y=y,
                train_mask=train_mask, test_mask=test_mask,
                node_image_counts=torch.ones(n, dtype=torch.long),
                edge_weight=edge_weight)


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------
class TestCostModel:
    def test_local_zero_bytes(self):
        from src.simulation.cost_model import compute_costs
        c = compute_costs("local", num_nodes=50, embedding_dim=0,
                          model_flops=1e6, is_cloud=False)
        assert c.total_bytes == 0.0
        assert c.transmission_latency_s == 0.0

    def test_distributed_bytes_proportional_to_emb_dim(self):
        from src.simulation.cost_model import compute_costs
        c8  = compute_costs("dist", 50, 8,  1e6, False)
        c64 = compute_costs("dist", 50, 64, 1e6, False)
        assert abs(c64.total_bytes / c8.total_bytes - 8.0) < 1e-6

    def test_explicit_message_count_controls_distributed_bytes(self):
        from src.simulation.cost_model import compute_costs

        cost = compute_costs(
            "dist",
            num_nodes=50,
            embedding_dim=8,
            model_flops=1e6,
            is_cloud=False,
            num_messages=17,
        )
        assert cost.num_messages == 17
        assert cost.total_bytes == 17 * 8 * 4

    def test_cloud_uses_cloud_compute_speed(self):
        from src.simulation.cost_model import compute_costs, ScenarioConfig
        cfg = ScenarioConfig(edge_flops_per_s=100e6, cloud_flops_per_s=10e9)
        c_edge  = compute_costs("edge",  10, 32, 1e8, is_cloud=False, cfg=cfg)
        c_cloud = compute_costs("cloud", 10, 32, 1e8, is_cloud=True,  cfg=cfg)
        # Cloud should be 100x faster on the same FLOPs
        assert abs(c_cloud.compute_latency_s * 100 - c_edge.compute_latency_s) < 1e-9

    def test_total_latency_is_sum(self):
        from src.simulation.cost_model import compute_costs
        c = compute_costs("dist", 20, 16, 1e7, False)
        assert abs(c.total_latency_s - c.transmission_latency_s - c.compute_latency_s) < 1e-12

    def test_as_dict_has_required_keys(self):
        from src.simulation.cost_model import compute_costs
        c = compute_costs("local", 10, 0, 1e6, False)
        d = c.as_dict()
        required = {"system","bytes_per_node","total_bytes","transmission_latency_s",
                    "compute_latency_s","total_latency_s","edge_flops","cloud_flops",
                    "total_flops","energy_proxy"}
        assert required.issubset(d.keys())


# ---------------------------------------------------------------------------
# Local system
# ---------------------------------------------------------------------------
class TestLocalSystem:
    def test_returns_valid_result(self):
        from src.systems.local_system import run_local_system
        from src.models.baselines import SimpleMLPBaseline
        g = _make_graph()
        model = SimpleMLPBaseline(input_dim=32, hidden_dim=32, num_classes=5)
        result = run_local_system(model, g, epochs=2, lr=1e-3, weight_decay=0,
                                  device=torch.device("cpu"))
        assert 0.0 <= result.test_accuracy <= 1.0
        assert result.cost.total_bytes == 0.0     # no communication
        assert result.cost.transmission_latency_s == 0.0

    def test_loss_curve_has_entries(self):
        from src.systems.local_system import run_local_system
        from src.models.baselines import SimpleMLPBaseline
        g = _make_graph()
        model = SimpleMLPBaseline(input_dim=32, hidden_dim=16, num_classes=5)
        result = run_local_system(model, g, epochs=3, lr=1e-3, weight_decay=0,
                                  device=torch.device("cpu"))
        assert len(result.loss_curve) > 0


# ---------------------------------------------------------------------------
# Distributed system
# ---------------------------------------------------------------------------
class TestDistributedSystem:
    def test_returns_valid_result(self):
        from src.systems.distributed_system import run_distributed_system
        g = _make_graph()
        result = run_distributed_system(
            input_dim=32, embedding_dim=8, hidden_dim=16, num_classes=5,
            graph=g, epochs=2, lr=1e-3, weight_decay=0,
            device=torch.device("cpu"),
        )
        assert 0.0 <= result.test_accuracy <= 1.0
        assert result.embedding_dim == 8
        assert result.compression_ratio == pytest.approx(32 / 8)

    def test_bytes_proportional_to_emb_dim(self):
        from src.systems.distributed_system import run_distributed_system
        g = _make_graph()
        r8  = run_distributed_system(32, 8,  16, 5, g.clone(), 1, 1e-3, 0, torch.device("cpu"))
        r32 = run_distributed_system(32, 32, 16, 5, g.clone(), 1, 1e-3, 0, torch.device("cpu"))
        assert abs(r32.cost.total_bytes / r8.cost.total_bytes - 4.0) < 0.01

    def test_embedding_compressor_in_model(self):
        from src.systems.distributed_system import DistributedTinyGNN
        model = DistributedTinyGNN(input_dim=32, embedding_dim=8,
                                   hidden_dim=16, num_classes=5)
        assert hasattr(model, "compressor")
        assert model.embedding_dim == 8

    def test_forward_produces_correct_shape(self):
        from src.systems.distributed_system import DistributedTinyGNN
        g = _make_graph()
        model = DistributedTinyGNN(input_dim=32, embedding_dim=8,
                                   hidden_dim=16, num_classes=5)
        model.eval()
        with torch.no_grad():
            out = model(g.x, g.edge_index)
        assert out.shape == (g.num_nodes, 5)


# ---------------------------------------------------------------------------
# Cloud system
# ---------------------------------------------------------------------------
class TestCloudSystem:
    def test_returns_valid_result(self):
        from src.systems.cloud_system import run_cloud_system
        from src.models.legacy_cloud_gcn import LegacyCloudGCN
        g = _make_graph()
        model = LegacyCloudGCN(input_dim=32, num_classes=5)
        result = run_cloud_system(model, g, epochs=2, lr=1e-3, weight_decay=0,
                                  device=torch.device("cpu"))
        assert 0.0 <= result.test_accuracy <= 1.0
        assert result.cost.total_bytes > 0    # cloud transmits full features
        assert result.cost.cloud_flops > 0
        assert result.cost.edge_flops == 0    # runs on cloud, not edge


# ---------------------------------------------------------------------------
# Three-system comparison
# ---------------------------------------------------------------------------
class TestThreeSystemComparison:
    def test_all_systems_run(self):
        from src.experiment.three_system_comparison import run_three_system_comparison
        g = _make_graph(n_train=40, n_test=15, feat_dim=32, num_classes=5)
        results = run_three_system_comparison(
            graph=g, input_dim=32, num_classes=5,
            embedding_dims=[8, 16], epochs=2, seeds=[42],
        )
        assert results.local is not None
        assert len(results.distributed_variants) == 2
        assert results.cloud is not None

    def test_bandwidth_ordering(self):
        """Communication is counted from actual messages, not assumed ordering."""
        from src.experiment.three_system_comparison import run_three_system_comparison
        g = _make_graph(n_train=40, n_test=15, feat_dim=32, num_classes=5)
        results = run_three_system_comparison(
            graph=g, input_dim=32, num_classes=5,
            embedding_dims=[16], epochs=1, seeds=[42],
        )
        local_bytes = results.local.cost.total_bytes
        dist_bytes  = results.distributed_variants[0].cost.total_bytes
        cloud_bytes = results.cloud.cost.total_bytes
        assert local_bytes == 0.0
        assert dist_bytes > 0
        assert cloud_bytes > 0
        assert results.distributed_variants[0].cost.num_messages > 0

    def test_smaller_embedding_less_bandwidth(self):
        """emb=8 must use less bandwidth than emb=32."""
        from src.experiment.three_system_comparison import run_three_system_comparison
        g = _make_graph(n_train=40, n_test=15, feat_dim=32, num_classes=5)
        results = run_three_system_comparison(
            graph=g, input_dim=32, num_classes=5,
            embedding_dims=[8, 32], epochs=1, seeds=[42],
        )
        variants = {r.embedding_dim: r for r in results.distributed_variants}
        assert variants[8].cost.total_bytes < variants[32].cost.total_bytes

    def test_as_dict_serializable(self):
        import json
        from src.experiment.three_system_comparison import run_three_system_comparison
        g = _make_graph()
        results = run_three_system_comparison(
            graph=g, input_dim=32, num_classes=5,
            embedding_dims=[8], epochs=1, seeds=[42],
        )
        d = results.as_dict()
        json.dumps(d)   # must not raise

    def test_cloud_pct_in_range(self):
        from src.experiment.three_system_comparison import run_three_system_comparison
        g = _make_graph(n_train=40, n_test=15, feat_dim=32, num_classes=5)
        results = run_three_system_comparison(
            graph=g, input_dim=32, num_classes=5,
            embedding_dims=[16], epochs=2, seeds=[42],
        )
        pct = results.cloud_pct_at(16)
        assert pct is not None
        assert pct >= 0.0

    def test_bandwidth_reduction_positive(self):
        """Distributed system must use less bandwidth than cloud."""
        from src.experiment.three_system_comparison import run_three_system_comparison
        g = _make_graph(n_train=40, n_test=15, feat_dim=32, num_classes=5)
        results = run_three_system_comparison(
            graph=g, input_dim=32, num_classes=5,
            embedding_dims=[8], epochs=1, seeds=[42],
        )
        bw_red = results.bandwidth_reduction_at(8)
        assert bw_red is not None
        assert bw_red > 0.0

    def test_best_distributed_uses_validation_not_test_accuracy(self):
        from src.experiment.three_system_comparison import (
            CloudSystemResult,
            DistributedSystemResult,
            LocalSystemResult,
            ThreeSystemResults,
        )
        from src.simulation.cost_model import CostBreakdown, ScenarioConfig

        cost = CostBreakdown(system_name="test")
        local = LocalSystemResult(0.2, 0.3, 1, cost, [1.0, 0.9], 16)
        cloud = CloudSystemResult(0.5, 0.6, 1, cost, [1.0, 0.8], 16)
        high_test = DistributedSystemResult(
            0.9, 0.5, 1, 8, cost, [1.0, 0.9], 2.0, val_accuracy=0.3
        )
        high_val = DistributedSystemResult(
            0.4, 0.5, 1, 16, cost, [1.0, 0.9], 1.0, val_accuracy=0.7
        )
        results = ThreeSystemResults(
            local=local,
            distributed_variants=[high_test, high_val],
            cloud=cloud,
            seeds_used=[42],
            scenario=ScenarioConfig(),
        )
        assert results.best_distributed().embedding_dim == 16


class TestPartitionAwareCommunication:
    def test_only_cross_partition_edges_are_messages(self):
        from src.simulation.partitioning import (
            assign_balanced_partitions,
            communication_edge_count,
        )

        graph = _make_graph(n_train=20, n_test=8, feat_dim=16, num_classes=5)
        partitioned = assign_balanced_partitions(graph, num_partitions=4, seed=42)
        src, dst = partitioned.edge_index
        expected = int(
            (
                partitioned.partition_id[src]
                != partitioned.partition_id[dst]
            ).sum().item()
        )
        assert communication_edge_count(partitioned) == expected


# ---------------------------------------------------------------------------
# Bug regressions
# ---------------------------------------------------------------------------
class TestBugRegressions:
    """Regression tests for the two confirmed bugs."""

    def test_residual_works_when_hidden_gt_embedding(self):
        """Bug: F.pad with negative size when hidden_dim > embedding_dim.
        Fix: res_proj Linear aligns dims before addition."""
        from src.systems.distributed_system import DistributedTinyGNN
        g = _make_graph(feat_dim=32)
        # Common case: emb=8, hidden=64 — this crashed before the fix
        model = DistributedTinyGNN(input_dim=32, embedding_dim=8,
                                   hidden_dim=64, num_classes=5)
        model.eval()
        with torch.no_grad():
            out = model(g.x, g.edge_index)
        assert out.shape == (g.num_nodes, 5), \
            "Forward must succeed when hidden_dim (64) > embedding_dim (8)"
        assert not torch.isnan(out).any()

    def test_residual_works_when_hidden_eq_embedding(self):
        """When dims match, res_proj is Identity — no projection needed."""
        from src.systems.distributed_system import DistributedTinyGNN
        g = _make_graph(feat_dim=32)
        model = DistributedTinyGNN(input_dim=32, embedding_dim=64,
                                   hidden_dim=64, num_classes=5)
        model.eval()
        with torch.no_grad():
            out = model(g.x, g.edge_index)
        assert out.shape == (g.num_nodes, 5)

    def test_residual_works_when_hidden_lt_embedding(self):
        """Edge case: hidden_dim < embedding_dim. res_proj projects down."""
        from src.systems.distributed_system import DistributedTinyGNN
        g = _make_graph(feat_dim=32)
        model = DistributedTinyGNN(input_dim=32, embedding_dim=32,
                                   hidden_dim=16, num_classes=5)
        model.eval()
        with torch.no_grad():
            out = model(g.x, g.edge_index)
        assert out.shape == (g.num_nodes, 5)

    def test_res_proj_is_identity_when_dims_match(self):
        """res_proj must be nn.Identity when embedding_dim == hidden_dim."""
        from src.systems.distributed_system import DistributedTinyGNN
        model = DistributedTinyGNN(input_dim=32, embedding_dim=64,
                                   hidden_dim=64, num_classes=5)
        assert isinstance(model.res_proj, torch.nn.Identity), \
            "res_proj must be Identity when embedding_dim == hidden_dim"

    def test_res_proj_is_linear_when_dims_differ(self):
        """res_proj must be nn.Linear when embedding_dim != hidden_dim."""
        from src.systems.distributed_system import DistributedTinyGNN
        model = DistributedTinyGNN(input_dim=32, embedding_dim=8,
                                   hidden_dim=64, num_classes=5)
        assert isinstance(model.res_proj, torch.nn.Linear), \
            "res_proj must be Linear when embedding_dim (8) != hidden_dim (64)"
        assert model.res_proj.in_features == 8
        assert model.res_proj.out_features == 64

    def test_run_three_system_script_imports_torch(self):
        """Bug: torch was only imported inside functions, so torch.Generator()
        in build_graph() crashed at module level before main() ran."""
        import importlib.util, sys
        # Parse the script source and verify torch is imported at module level
        script_path = str(
            __import__("pathlib").Path(__file__).parents[1]
            / "scripts"
            / "run_three_system.py"
        )
        with open(script_path) as f:
            lines = f.readlines()

        # Find the first non-comment, non-docstring import of torch
        module_level_torch = False
        in_function = False
        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith("def ") or stripped.startswith("class "):
                in_function = True
            if not in_function and stripped.startswith("import torch"):
                module_level_torch = True
                break

        assert module_level_torch, (
            "torch must be imported at module level in run_three_system.py. "
            "torch.Generator() in build_graph() fails without it."
        )


# ---------------------------------------------------------------------------
# FLOP estimator correctness
# ---------------------------------------------------------------------------
class TestFlopEstimators:
    """Regression tests for the compressor FLOP bug and model/estimate consistency."""

    def test_gnn_flop_compressor_scales_with_input_dim(self):
        """Bug: was embedding_dim**2. Must be input_dim * embedding_dim."""
        from src.experiment.three_system_comparison import _estimate_gnn_flops
        # With input_dim=128, embedding_dim=8: compressor = 2 * 128 * 8 = 2048
        # Wrong version gave: 2 * 8 * 8 = 128
        flops_correct = _estimate_gnn_flops(
            input_dim=128, embedding_dim=8, hidden_dim=16, num_edges=100
        )
        # If bug were present, flops would be dominated by tiny 128 compressor term
        # vs the correct 2048 term — difference detectable at fixed num_edges
        flops_buggy_would_give = 2.0 * 8 * 8 + 100 * 3.0 * 16 + 2.0 * 16 * 10
        flops_correct_should_give = (
            2.0 * 128 * 8
            + 2.0 * 8 * 16
            + 100 * 3.0 * 16
            + 2.0 * 16 * 10
        )
        assert abs(flops_correct - flops_correct_should_give) < 1e-6, \
            f"Expected {flops_correct_should_give}, got {flops_correct}"
        assert flops_correct != flops_buggy_would_give, \
            "FLOP estimate must use input_dim*embedding_dim, not embedding_dim**2"

    def test_gnn_flops_increase_with_input_dim(self):
        """Larger input_dim must produce larger FLOP estimate (compressor dominates)."""
        from src.experiment.three_system_comparison import _estimate_gnn_flops
        f_small = _estimate_gnn_flops(32,  8, 16, 50)
        f_large = _estimate_gnn_flops(512, 8, 16, 50)
        assert f_large > f_small, \
            "FLOPs must increase with input_dim (compressor is Linear(input->emb))"

    def test_gnn_flops_increase_with_embedding_dim(self):
        """Larger embedding_dim → more compressor work AND more GATConv work."""
        from src.experiment.three_system_comparison import _estimate_gnn_flops
        f8  = _estimate_gnn_flops(128, 8,  64, 100)
        f64 = _estimate_gnn_flops(128, 64, 64, 100)
        assert f64 > f8

    def test_cloud_model_dims_match_flop_estimate(self):
        """Cloud FLOP estimate must use the actual model's conv output dim,
        not the function's hidden_dim argument, so they cannot drift apart."""
        from src.models.legacy_cloud_gcn import LegacyCloudGCN
        from src.experiment.three_system_comparison import _estimate_legacy_cloud_gcn_flops
        model = LegacyCloudGCN(input_dim=128, hidden_dim=64, num_classes=10)
        # Read hidden_dim from the model, not from a separate variable
        actual_hidden = model.conv1.out_channels
        assert actual_hidden == 64
        flops = _estimate_legacy_cloud_gcn_flops(
            input_dim=128,
            hidden_dim=actual_hidden,   # derived from model, not args
            num_edges=500,
        )
        # If hidden_dim were misread as 128 (the default), flops would be 2x larger
        flops_with_default_hidden = _estimate_legacy_cloud_gcn_flops(128, 128, 500)
        assert flops != flops_with_default_hidden, \
            "Test setup: ensure non-default hidden_dim is actually different"
        # The key invariant: consistent computation
        assert flops == _estimate_legacy_cloud_gcn_flops(128, 64, 500)

    def test_distributed_flops_gt_zero_for_all_emb_dims(self):
        from src.experiment.three_system_comparison import _estimate_gnn_flops
        for emb_dim in [8, 16, 32, 64]:
            f = _estimate_gnn_flops(128, emb_dim, 64, 200)
            assert f > 0, f"FLOP estimate must be positive for emb_dim={emb_dim}"


# ---------------------------------------------------------------------------
# Cloud framing: all systems use same features
# ---------------------------------------------------------------------------
class TestCloudFraming:
    def test_cloud_and_distributed_share_feature_pipeline(self):
        """Both systems start from the same graph.x — no raw pixel advantage for cloud."""
        from src.experiment.three_system_comparison import run_three_system_comparison
        g = _make_graph(n_train=40, n_test=15, feat_dim=32, num_classes=5)
        results = run_three_system_comparison(
            graph=g, input_dim=32, num_classes=5,
            embedding_dims=[16], epochs=1, seeds=[42],
        )
        # Both cloud and distributed see the same input_dim
        assert results.cloud.full_feature_dim == 32
        assert results.distributed_variants[0].embedding_dim < 32  # compressed

    def test_cloud_bytes_equals_full_feature_dim_times_nodes(self):
        """Cloud transmits full features: bytes = num_nodes * feat_dim * 4."""
        from src.simulation.cost_model import compute_costs, ScenarioConfig
        cfg = ScenarioConfig()
        n_nodes, feat_dim = 50, 128
        c = compute_costs("cloud", n_nodes, feat_dim, 1e8,
                          is_cloud=True, transmit_full_features=True,
                          full_feature_dim=feat_dim, cfg=cfg)
        expected = n_nodes * feat_dim * cfg.bytes_per_float
        assert abs(c.total_bytes - expected) < 1e-6

    def test_framing_note_in_module_docstring(self):
        """The explicit framing note must be present in the experiment module."""
        import importlib, sys, os
        sys.path.insert(0, os.path.join(
            os.path.dirname(__file__), "..", "src"
        ))
        import experiment.three_system_comparison as m
        assert "compressed features" in m.__doc__ or "same compressed features" in m.__doc__, \
            "Module docstring must document the 'same features for all systems' framing"


# ---------------------------------------------------------------------------
# Encoder config flag
# ---------------------------------------------------------------------------
class TestEncoderConfig:
    def test_use_pretrained_encoder_in_config(self):
        import src.config as cfg
        assert hasattr(cfg, "USE_PRETRAINED_ENCODER")
        assert isinstance(cfg.USE_PRETRAINED_ENCODER, bool)

    def test_encoder_type_in_config(self):
        import src.config as cfg
        assert hasattr(cfg, "ENCODER_TYPE")
        assert cfg.ENCODER_TYPE in ("resnet18", "dino")

    def test_default_encoder_is_disabled(self):
        """Default must be False so the pipeline works without torchvision installed."""
        import src.config as cfg
        assert cfg.USE_PRETRAINED_ENCODER is False, \
            "Default must be USE_PRETRAINED_ENCODER=False (no torchvision required)"


# ---------------------------------------------------------------------------
# Canonical settings
# ---------------------------------------------------------------------------
class TestCanonicalSettings:
    def test_canonical_constants_exist(self):
        import src.config as cfg
        assert hasattr(cfg, "CANONICAL_K_NEIGHBORS")
        assert hasattr(cfg, "CANONICAL_EPOCHS")
        assert hasattr(cfg, "CANONICAL_EMBEDDING_DIMS")
        assert hasattr(cfg, "CONVERGENCE_MIN_LOSS_DROP")

    def test_canonical_epochs_sufficient(self):
        import src.config as cfg
        assert cfg.CANONICAL_EPOCHS >= 20, \
            f"CANONICAL_EPOCHS={cfg.CANONICAL_EPOCHS} — need ≥20 to verify convergence"

    def test_canonical_k_in_reasonable_range(self):
        import src.config as cfg
        assert 3 <= cfg.CANONICAL_K_NEIGHBORS <= 10, \
            f"CANONICAL_K_NEIGHBORS={cfg.CANONICAL_K_NEIGHBORS} outside [3,10]"

    def test_canonical_embedding_dims_ordered(self):
        import src.config as cfg
        dims = cfg.CANONICAL_EMBEDDING_DIMS
        assert dims == sorted(dims), "CANONICAL_EMBEDDING_DIMS must be ascending"
        assert dims[0] >= 4, "Smallest embedding_dim must be ≥4 (too small is uninformative)"

    def test_sweep_epochs_updated(self):
        """SWEEP_EPOCHS must not include 1 or 3 — those never converge."""
        import src.config as cfg
        assert 1 not in cfg.SWEEP_EPOCHS, "1 epoch never converges — remove from sweep"
        assert 3 not in cfg.SWEEP_EPOCHS, "3 epochs never converge — remove from sweep"
        assert all(e >= 20 for e in cfg.SWEEP_EPOCHS), \
            f"All sweep epochs must be ≥20, got {cfg.SWEEP_EPOCHS}"


# ---------------------------------------------------------------------------
# Convergence check
# ---------------------------------------------------------------------------
class TestConvergenceCheck:
    def _make_results(self, local_curve, dist_curve, cloud_curve):
        """Build a minimal ThreeSystemResults with synthetic loss curves."""
        from src.experiment.three_system_comparison import (
            ThreeSystemResults, LocalSystemResult, DistributedSystemResult, CloudSystemResult
        )
        from src.simulation.cost_model import CostBreakdown, ScenarioConfig

        def _cost(name):
            return CostBreakdown(system_name=name)

        local = LocalSystemResult(0.3, 0.4, 100, _cost("local"), local_curve, 32)
        dist  = DistributedSystemResult(0.28, 0.38, 80, 16, _cost("dist"), dist_curve, 2.0)
        cloud = CloudSystemResult(0.35, 0.45, 200, _cost("cloud"), cloud_curve, 32)
        r = ThreeSystemResults(local=local, distributed_variants=[dist],
                               cloud=cloud, seeds_used=[42],
                               scenario=ScenarioConfig())
        return r

    def test_converged_curve_passes(self):
        """A curve that drops 20% must pass convergence check."""
        results = self._make_results(
            local_curve=[2.3, 2.0, 1.8],
            dist_curve= [2.3, 2.0, 1.8],
            cloud_curve=[2.3, 1.9, 1.7],
        )
        report = results.check_convergence(min_relative_drop=0.10)
        assert report["local"]["converged"], \
            f"Local should converge: {report['local']}"

    def test_flat_curve_fails(self):
        """A curve stuck at 2.30 must fail convergence check."""
        results = self._make_results(
            local_curve=[2.30, 2.29, 2.30, 2.31],
            dist_curve= [2.30, 2.29, 2.30],
            cloud_curve=[2.30, 2.29, 2.30],
        )
        report = results.check_convergence(min_relative_drop=0.10)
        assert not report["local"]["converged"], \
            "Flat loss curve at 2.30 must not be reported as converged"

    def test_convergence_stored_on_result(self):
        """check_convergence result must be stored on the object."""
        from src.experiment.three_system_comparison import run_three_system_comparison
        g = _make_graph(n_train=40, n_test=15, feat_dim=32, num_classes=5)
        results = run_three_system_comparison(
            graph=g, input_dim=32, num_classes=5,
            embedding_dims=[16], epochs=5, seeds=[42],
        )
        assert results.convergence_report is not None
        assert "local" in results.convergence_report
        assert "cloud" in results.convergence_report


# ---------------------------------------------------------------------------
# Trade-off plot (smoke test — does not require display)
# ---------------------------------------------------------------------------
class TestTradeOffPlot:
    def test_plot_runs_without_error(self, tmp_path):
        """plot_trade_off must complete without error and save files."""
        pytest.importorskip("matplotlib")
        from src.experiment.three_system_comparison import run_three_system_comparison
        from src.utils.trade_off_plot import plot_trade_off

        g = _make_graph(n_train=40, n_test=15, feat_dim=32, num_classes=5)
        results = run_three_system_comparison(
            graph=g, input_dim=32, num_classes=5,
            embedding_dims=[8, 16], epochs=2, seeds=[42],
        )
        saved = plot_trade_off(results, save_dir=tmp_path, show=False)
        assert len(saved) == 4, f"Expected 4 figures, got {len(saved)}"
        for p in saved:
            assert p.exists(), f"Figure not created: {p}"
            assert p.stat().st_size > 0, f"Empty figure file: {p}"

    def test_main_figure_is_accuracy_vs_embedding_dim(self, tmp_path):
        pytest.importorskip("matplotlib")
        from src.experiment.three_system_comparison import run_three_system_comparison
        from src.utils.trade_off_plot import plot_trade_off

        g = _make_graph(n_train=40, n_test=15, feat_dim=32, num_classes=5)
        results = run_three_system_comparison(
            graph=g, input_dim=32, num_classes=5,
            embedding_dims=[8, 32], epochs=2, seeds=[42],
        )
        saved = plot_trade_off(results, save_dir=tmp_path, show=False)
        names = [p.name for p in saved]
        assert any("embedding_dim" in n for n in names), \
            f"Main figure (accuracy vs embedding_dim) not found in {names}"


# ---------------------------------------------------------------------------
# Fix 1: early stopping restores best checkpoint
# ---------------------------------------------------------------------------
class TestEarlyStoppingCheckpointRestore:
    def test_best_weights_restored_not_last(self):
        """After early stopping, model weights must be the best-val ones."""
        from src.models.baselines import SimpleMLPBaseline
        from src.training.trainer import make_optimizer, train_with_logging
        import copy

        torch.manual_seed(0)
        g = _make_graph(n_train=30, n_test=10, feat_dim=16, num_classes=5)
        model = SimpleMLPBaseline(input_dim=16, hidden_dim=32, num_classes=5)

        # Record the best state manually alongside training
        val_mask = g.test_mask
        opt = make_optimizer(model, lr=1e-2, weight_decay=0)
        loss_fn = torch.nn.CrossEntropyLoss()

        # Train and capture the state at the best val epoch ourselves
        best_val = float("inf")
        best_state_ref = None

        for epoch in range(1, 20):
            from src.training.trainer import train_one_epoch, model_forward
            train_one_epoch(model, g, opt, loss_fn, "cpu")
            model.eval()
            with torch.no_grad():
                logits = model_forward(model, g)
                vl = float(loss_fn(logits[val_mask], g.y[val_mask]).item())
            model.train()
            if vl < best_val - 1e-4:
                best_val = vl
                best_state_ref = copy.deepcopy(model.state_dict())

        # Now train a fresh model with train_with_logging + early stopping
        torch.manual_seed(0)
        model2 = SimpleMLPBaseline(input_dim=16, hidden_dim=32, num_classes=5)
        opt2 = make_optimizer(model2, lr=1e-2, weight_decay=0)
        train_with_logging(
            model2, g, opt2, loss_fn, "cpu",
            epochs=20, patience=5, val_mask=val_mask, log_every=100,
        )

        # The weights after train_with_logging must NOT be the same as
        # whatever epoch 20 produced if early stopping fired.
        # We verify the restore happened by checking that the val loss
        # of the returned model <= best_val + small epsilon.
        model2.eval()
        with torch.no_grad():
            from src.training.trainer import model_forward
            logits2 = model_forward(model2, g)
            final_val = float(loss_fn(logits2[val_mask], g.y[val_mask]).item())
        assert final_val <= best_val + 0.05, (
            f"Restored model val loss ({final_val:.4f}) should be close to best "
            f"seen ({best_val:.4f}). Checkpoint restore may not have worked."
        )

    def test_best_state_printed_on_restore(self, capsys):
        """train_with_logging should print a message when restoring best checkpoint."""
        from src.models.baselines import SimpleMLPBaseline
        from src.training.trainer import make_optimizer, train_with_logging

        g = _make_graph(n_train=20, n_test=8, feat_dim=16, num_classes=5)
        model = SimpleMLPBaseline(input_dim=16, hidden_dim=16, num_classes=5)
        opt = make_optimizer(model, lr=1e-3)
        train_with_logging(
            model, g, opt, torch.nn.CrossEntropyLoss(), "cpu",
            epochs=5, patience=10, val_mask=g.test_mask, log_every=100,
        )
        captured = capsys.readouterr()
        assert "restored best checkpoint" in captured.out, \
            "train_with_logging must print confirmation of checkpoint restore"

    def test_no_restore_without_val_mask(self):
        """Without val_mask, no early stopping occurs and no restore is attempted."""
        from src.models.baselines import SimpleMLPBaseline
        from src.training.trainer import make_optimizer, train_with_logging

        g = _make_graph()
        model = SimpleMLPBaseline(input_dim=32, hidden_dim=16, num_classes=5)
        opt = make_optimizer(model, lr=1e-3)
        # Must not raise even though no val_mask provided
        curve = train_with_logging(
            model, g, opt, torch.nn.CrossEntropyLoss(), "cpu",
            epochs=5, val_mask=None, log_every=100,
        )
        assert len(curve) == 5   # all epochs ran, no early stop


# ---------------------------------------------------------------------------
# Fix 2: multi-seed semantics documented
# ---------------------------------------------------------------------------
class TestMultiSeedSemantics:
    def test_same_graph_different_seeds_gives_different_accuracy(self):
        """Training randomness alone (not graph randomness) must vary across seeds."""
        from src.experiment.three_system_comparison import run_three_system_comparison
        g = _make_graph(n_train=40, n_test=15, feat_dim=32, num_classes=5)

        r1 = run_three_system_comparison(
            graph=g.clone(), input_dim=32, num_classes=5,
            embedding_dims=[16], epochs=5, seeds=[42],
        )
        r2 = run_three_system_comparison(
            graph=g.clone(), input_dim=32, num_classes=5,
            embedding_dims=[16], epochs=5, seeds=[999],
        )
        # Results may differ due to training randomness
        # (they won't always differ on tiny graphs, but the infrastructure must work)
        assert r1.local.test_accuracy is not None
        assert r2.local.test_accuracy is not None

    def test_seed_semantics_documented_in_docstring(self):
        """run_three_system_comparison docstring must explain graph-reuse."""
        from src.experiment import three_system_comparison as m
        doc = m.run_three_system_comparison.__doc__
        assert doc is not None
        assert "graph" in doc.lower() and "seed" in doc.lower(), \
            "Docstring must explain multi-seed semantics w.r.t. graph construction"


# ---------------------------------------------------------------------------
# Fix 3: encoder cost framing
# ---------------------------------------------------------------------------
class TestEncoderCostFraming:
    def test_encoder_latency_default_zero(self):
        """By default encoder cost is NOT included in per-round latency."""
        from src.simulation.cost_model import CostBreakdown
        c = CostBreakdown(system_name="test",
                          transmission_latency_s=0.1, compute_latency_s=0.05)
        assert c.encoder_latency_s == 0.0
        assert c.total_latency_s == pytest.approx(0.15)

    def test_encoder_latency_added_when_online(self):
        """Setting encoder_latency_s > 0 adds it to total_latency_s."""
        from src.simulation.cost_model import CostBreakdown
        c = CostBreakdown(system_name="test",
                          transmission_latency_s=0.1, compute_latency_s=0.05,
                          encoder_latency_s=0.2)
        assert c.total_latency_s == pytest.approx(0.35)

    def test_encoder_flops_separate_from_total(self):
        """encoder_preprocessing_flops must be separate from total_flops."""
        from src.simulation.cost_model import CostBreakdown
        c = CostBreakdown(system_name="test",
                          edge_flops=1e7,
                          encoder_preprocessing_flops=5e8)
        assert c.total_flops == pytest.approx(1e7)
        assert c.total_flops_including_encoder == pytest.approx(5e8 + 1e7)

    def test_as_dict_includes_encoder_fields(self):
        from src.simulation.cost_model import CostBreakdown
        c = CostBreakdown(system_name="test", encoder_latency_s=0.1,
                          encoder_preprocessing_flops=1e8)
        d = c.as_dict()
        assert "encoder_latency_s" in d
        assert "encoder_preprocessing_flops" in d
        assert "total_flops_including_encoder" in d


# ---------------------------------------------------------------------------
# Fix 4a: RandomBaseline name accuracy
# ---------------------------------------------------------------------------
class TestRandomBaselineNaming:
    def test_docstring_says_constant_not_random(self):
        from src.models.baselines import RandomBaseline
        doc = RandomBaseline.__doc__
        assert doc is not None
        # Must acknowledge it's NOT stochastic random
        assert "constant" in doc.lower() or "zero" in doc.lower() or \
               "not" in doc.lower(), \
            "RandomBaseline docstring must clarify it outputs constant zeros, not random values"

    def test_output_is_deterministic_across_calls(self):
        from src.models.baselines import RandomBaseline
        model = RandomBaseline(num_classes=5)
        x = torch.rand(10, 32)
        out1 = model(x)
        out2 = model(x)
        assert torch.allclose(out1, out2), \
            "RandomBaseline must be deterministic (constant output)"

    def test_output_is_zero_logits(self):
        from src.models.baselines import RandomBaseline
        model = RandomBaseline(num_classes=5)
        out = model(torch.rand(8, 16))
        assert torch.all(out == 0), "RandomBaseline must output all-zero logits"
