"""Rutinas de entrenamiento con logging de loss por época."""

import torch
from torch import nn
from torch.optim import Optimizer
from torch_geometric.data import Data

from config import EARLY_STOPPING_PATIENCE, LOG_LOSS_EVERY_N_EPOCHS, WEIGHT_DECAY


def model_forward(model: nn.Module, data: Data) -> torch.Tensor:
    """Ejecuta el forward segun si el modelo usa grafo o no.

    Intenta pasar edge_weight si el modelo lo acepta; si no, omite.
    """
    if not getattr(model, "uses_graph", True):
        return model(data.x)

    edge_weight = getattr(data, "edge_weight", None)
    if edge_weight is not None:
        try:
            return model(data.x, data.edge_index, edge_weight)
        except TypeError:
            pass
    return model(data.x, data.edge_index)


def make_optimizer(model: nn.Module, lr: float, weight_decay: float = WEIGHT_DECAY):
    """Adam con weight decay sobre parametros entrenables."""
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        return None
    return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)


def train_one_epoch(
    model: nn.Module,
    data: Data,
    optimizer: Optimizer | None,
    loss_fn: nn.Module,
    device: torch.device | str,
) -> float:
    """Entrena una epoca, devuelve la perdida."""
    model.to(device)
    data = data.to(device)

    if optimizer is None:
        model.eval()
        with torch.no_grad():
            logits = model_forward(model, data)
            loss = loss_fn(logits[data.train_mask], data.y[data.train_mask])
        return float(loss.item())

    model.train()
    optimizer.zero_grad()
    logits = model_forward(model, data)
    loss = loss_fn(logits[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss.item())


def train_with_logging(
    model: nn.Module,
    data: Data,
    optimizer: Optimizer | None,
    loss_fn: nn.Module,
    device: torch.device | str,
    epochs: int,
    log_every: int = LOG_LOSS_EVERY_N_EPOCHS,
    label: str = "",
    patience: int = EARLY_STOPPING_PATIENCE,
    val_mask: torch.Tensor | None = None,
) -> list[float]:
    """Entrena hasta `epochs` epocas con early stopping y restauracion del mejor checkpoint.

    Early stopping with best-model restore:
      When `val_mask` is provided, we track the model state_dict at the epoch
      with the lowest validation loss. If training stops early (or when it ends),
      we restore that best checkpoint before returning. This guarantees that the
      model passed back to the caller is the one with the best validation
      performance, not the one from the last training step (which may have
      overfit or diverged slightly).

      Without this restore, reported test accuracy reflects the last epoch's
      weights, which can be meaningfully worse than the best epoch's weights —
      especially with 50-100 epochs and patience=10.

    Patch 5: prints logits.std() on epoch 1 to detect model collapse.

    Returns:
        Lista de floats with training loss per epoch.
    """
    import copy
    from utils.feature_normalization import logits_diagnostic

    loss_curve: list[float] = []
    prefix = f"[{label}] " if label else ""

    best_val_loss = float("inf")
    best_state: dict | None = None     # deepcopy of state_dict at best val loss
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, data, optimizer, loss_fn, device)
        loss_curve.append(loss)

        # Patch 5: logits diagnostic on epoch 1 to detect immediate collapse
        if epoch == 1:
            model.eval()
            with torch.no_grad():
                logits = model_forward(model, data.to(device))
            logits_diagnostic(logits, label=f"{label}/e1")
            model.train()

        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            print(f"{prefix}epoch {epoch:4d}/{epochs}  loss={loss:.4f}")

        # Early stopping on val loss with checkpoint saving
        if val_mask is not None:
            model.eval()
            with torch.no_grad():
                logits = model_forward(model, data.to(device))
                val_loss = float(loss_fn(
                    logits[val_mask.to(device)],
                    data.y.to(device)[val_mask.to(device)]
                ).item())
            model.train()

            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                # Save a deep copy so subsequent updates don't overwrite it
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"{prefix}early stop at epoch {epoch} "
                          f"(val_loss={val_loss:.4f}, patience={patience})")
                    break

    # Restore best checkpoint so evaluation sees the best model, not last epoch
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"{prefix}restored best checkpoint (val_loss={best_val_loss:.4f})")

    return loss_curve


def overfit_sanity_check(
    model: nn.Module,
    data: Data,
    device: torch.device | str,
    target_train_acc: float = 0.95,
    max_epochs: int = 500,
    lr: float = 1e-2,
) -> dict:
    """PATCH 5: verifica que el pipeline puede aprender en datos pequeños.

    Entrena hasta alcanzar target_train_acc o max_epochs en un subconjunto
    minimo (primeros 20 nodos de train). Si el modelo NO alcanza ~100% en
    20 muestras, el pipeline de entrenamiento está roto.

    Por que esto importa:
        Cuando todo el sweep da loss ~2.30 (random), la causa puede ser:
        a) Features tan malas que el modelo no puede aprender de ninguna forma.
        b) Un bug sutil: labels mal alineadas, mascaras incorrectas, gradientes
           que no llegan, etc.
        Este test distingue (a) de (b) porque con 20 muestras cualquier modelo
        razonable debería poder memorizar si el pipeline es correcto.

    Returns:
        dict con 'passed', 'epochs_to_converge', 'final_train_acc', 'loss_curve'
    """
    from training.evaluator import evaluate_accuracy

    # Subgrafo con solo los primeros 20 nodos de train
    train_idx = data.train_mask.nonzero(as_tuple=True)[0]
    tiny_idx = train_idx[:min(20, len(train_idx))]

    tiny_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    tiny_mask[tiny_idx] = True

    tiny_data = data.clone()
    tiny_data.train_mask = tiny_mask

    import copy
    model_copy = copy.deepcopy(model).to(device)
    opt = make_optimizer(model_copy, lr=lr, weight_decay=0.0)
    loss_fn = nn.CrossEntropyLoss()

    loss_curve = []
    epochs_to_converge = None

    for epoch in range(1, max_epochs + 1):
        loss = train_one_epoch(model_copy, tiny_data, opt, loss_fn, device)
        loss_curve.append(loss)
        acc = evaluate_accuracy(model_copy, tiny_data.to(device), tiny_mask.to(device))
        if acc >= target_train_acc and epochs_to_converge is None:
            epochs_to_converge = epoch
            break

    final_acc = evaluate_accuracy(model_copy, tiny_data.to(device), tiny_mask.to(device))
    passed = final_acc >= target_train_acc

    status = "✅ PASSED" if passed else "❌ FAILED — pipeline may be broken"
    print(
        f"\nOverfit sanity check {status}\n"
        f"  target_train_acc={target_train_acc:.0%}  "
        f"final_train_acc={final_acc:.3f}  "
        f"epochs={'did not converge' if epochs_to_converge is None else epochs_to_converge}"
    )
    return {
        "passed": passed,
        "epochs_to_converge": epochs_to_converge,
        "final_train_acc": final_acc,
        "loss_curve": loss_curve,
    }
