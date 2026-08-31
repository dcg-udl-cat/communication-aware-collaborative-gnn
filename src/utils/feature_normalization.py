"""Patch 2: per-feature normalization applied to graph.x before training.

Why this is needed even after PCA + L2:
  PCA+L2 normalizes each ROW (node) to unit norm.
  But the COLUMNS (features) still have different means and variances across
  nodes. A feature that encodes "presence of red" might span [-2, 8] while
  one encoding "edge sharpness" spans [-0.1, 0.1]. The first dominates
  gradient updates in the first linear layer regardless of its
  class-discriminative content.

  Column-wise standardization (zero mean, unit std per feature) followed
  by optional row L2-normalization gives the model a stable, isotropic
  input distribution where all features compete on equal footing.

  This is the single most impactful normalization step for models that
  receive PCA features — it alone can boost MLP accuracy by 5-15 points.
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data


def normalize_graph_features(
    graph: Data,
    train_mask: torch.Tensor,
    eps: float = 1e-6,
    l2_normalize_rows: bool = True,
    standardize_columns: bool = True,
) -> Data:
    """Returns a copy of graph with x column-normalized using train statistics.

    Procedure:
      1. Compute mean and std of x[train_mask] per feature column.
      2. Subtract mean, divide by std (+ eps) — applied to ALL nodes.
         Test nodes use train statistics: no leakage.
      3. Optionally L2-normalize rows so each node has unit norm.

    Args:
        graph:             PyG Data object.
        train_mask:        Boolean mask identifying train nodes.
        eps:               Stability term for division.
        l2_normalize_rows: If True, L2-normalize each row after column norm.
        standardize_columns: If False, preserve feature columns and only apply
            the requested row normalization. This is appropriate for sparse
            bag-of-words graph features whose official training mask does not
            observe every feature column.

    Returns:
        New Data with normalized x. Original graph is unchanged.
    """
    x = graph.x.float()
    if standardize_columns:
        train_x = x[train_mask.bool()]
        mean = train_x.mean(dim=0)            # (D,) computed on train only
        std = train_x.std(dim=0).clamp(min=eps)  # (D,) clamp avoids /0
        x_norm = (x - mean) / std
    else:
        x_norm = x

    if l2_normalize_rows:
        x_norm = F.normalize(x_norm, p=2, dim=1)

    new_graph = graph.clone()
    new_graph.x = x_norm
    return new_graph


def logits_diagnostic(logits: torch.Tensor, label: str = "") -> dict:
    """Patch 5: print logits statistics to detect model collapse.

    If logits.std() < 0.01 the model is outputting near-identical values
    for all nodes — it has not learned any discriminative signal.
    This happens when:
      - Features are all zeros or constant (upstream bug)
      - Gradients are vanishing (LR too low, or bad initialization)
      - Dropout is too aggressive (killing all activations)

    Returns dict with std, mean, max, min for logging.
    """
    with torch.no_grad():
        std  = float(logits.std().item())
        mean = float(logits.mean().item())
        mx   = float(logits.max().item())
        mn   = float(logits.min().item())

    prefix = f"[{label}] " if label else ""
    status = "✅ OK" if std > 0.1 else ("⚠️  LOW" if std > 0.01 else "❌ COLLAPSED")
    print(f"{prefix}logits std={std:.4f} mean={mean:.4f} "
          f"max={mx:.4f} min={mn:.4f}  {status}")
    return {"std": std, "mean": mean, "max": mx, "min": mn, "collapsed": std < 0.01}
