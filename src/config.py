"""Configuracion centralizada del pipeline de datos."""

from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "src" / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

TRAIN_SUBSET_SIZE = 1000
TEST_SUBSET_SIZE = 300
NUM_CLASSES = 10
# ---------------------------------------------------------------------------
# CANONICAL SETTING — lock this for all clean comparisons.
# Changing ipn or k changes the graph structure, making comparisons harder
# to interpret. Run sweeps separately; use these for the main figure.
# ---------------------------------------------------------------------------
IMAGES_PER_NODE = 1            # one observation per node; grouped labels are an ablation
CANONICAL_K_NEIGHBORS = 5      # canonical k for the main experiment
CANONICAL_EPOCHS = 50          # enough to verify convergence, not so many it's slow
CANONICAL_EMBEDDING_DIMS = [8, 16, 32, 64]   # main figure x-axis
# Convergence threshold: training is considered successful if final loss
# is at least this much lower than initial loss (relative drop).
CONVERGENCE_MIN_LOSS_DROP = 0.10   # 10% relative improvement required

RUN_MODE = "resource_efficiency"

FULL_GNN_TRAIN_SUBSET_SIZE: int | None = None
FULL_GNN_TEST_SUBSET_SIZE: int | None = None

TINY_GNN_TRAIN_SUBSET_SIZE = TRAIN_SUBSET_SIZE
TINY_GNN_TEST_SUBSET_SIZE = TEST_SUBSET_SIZE

CONTROLLED_TRAIN_SUBSET_SIZE = TRAIN_SUBSET_SIZE
CONTROLLED_TEST_SUBSET_SIZE = TEST_SUBSET_SIZE

IMAGE_FEATURE_DIM = 3 * 32 * 32

# Compresion por PCA.
# Si USE_PCA=True, los modelos reciben features de PCA_COMPONENTS dims
# en lugar de IMAGE_FEATURE_DIM. El PCA se ajusta SOLO sobre train.
USE_PCA = True

# Pretrained encoder (semantic features).
# USE_PRETRAINED_ENCODER=True replaces flatten+PCA with a frozen ResNet18.
# This produces semantically coherent KNN neighborhoods — the most impactful
# single upgrade for making the graph meaningful.
# Requires torchvision. Set to False to use PCA pipeline (faster, no download).
USE_PRETRAINED_ENCODER = False  # set True for best graph quality
ENCODER_TYPE = 'resnet18'       # 'resnet18' or 'dino'
PCA_COMPONENTS = 128
PCA_COMPONENTS_ABLATION = 256  # ablation only, test AFTER normalization is fixed

INPUT_DIM = PCA_COMPONENTS if USE_PCA else IMAGE_FEATURE_DIM
FULL_GNN_HIDDEN_DIM = 128
# Legacy CIFAR note: LegacyCloudGCN uses the same feature pipeline as distributed
# (PCA-compressed or encoder). It is 'bigger model, same features, no bandwidth
# constraint' — not 'raw pixel access'. This is the honest comparison.
TINY_GNN_HIDDEN_DIM = 64  # doubled from 32; still 'tiny' but not too weak for PCA-128 input
MLP_HIDDEN_DIM = 256       # patch 6: MLP needs more capacity than GNN to compensate for no graph
DROPOUT = 0.1            # reduced from 0.3: underfit regime, not overfit

K_NEIGHBORS = 8

# Metric for KNN graph construction.
# 'cosine' works better than 'euclidean' after L2-normalised PCA features:
# it measures angle between embeddings, which is more stable in high-ish dims.
# 'euclidean' is kept as an option for ablation.
KNN_METRIC = 'cosine'

# Mutual KNN: keep edge i↔j only if both i∈KNN(j) and j∈KNN(i).
# Removes noisy asymmetric edges. Highly recommended for small noisy graphs.
USE_MUTUAL_KNN = True

# Gaussian edge weight sigma. None = binary edges (no weights).
# w = exp(-d² / sigma²). 0.5 works well for L2-normalised features.
EDGE_WEIGHT_SIGMA: float | None = 0.5

# Modo de conectividad del grafo para el sweep de optimizacion de aristas.
# Progresion de mayor a menor coste computacional:
#   "undirected"    — train↔train + train↔test  (linea base, rompe protocolo inductivo)
#   "directed"      — train↔train + train→test  (protocolo inductivo correcto) [DEFAULT]
#   "half_directed" — train→train + train→test  (minimo coste, ~0.5x aristas vs directed)
# Alias por compatibilidad: "sparse" == "half_directed" (deprecado, emite aviso).
CONNECTIVITY_MODE = "directed"

# Valores de k y modos para el sweep de conectividad.
SWEEP_K_NEIGHBORS_EDGE = [2, 4, 8]
SWEEP_CONNECTIVITY_MODES = ["directed", "undirected", "half_directed"]

NORMALIZE_FEATURES = True

RANDOM_SEED = 42

# 100 epochs minimum to escape the random-prediction regime.
# Loss ~2.30 across all configs means the optimizer is barely moving.
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10   # stop if val loss doesn't improve for N epochs
LOG_LOSS_EVERY_N_EPOCHS = 10   # print loss curve during training
LEARNING_RATE = 1e-3           # 1e-3: fast enough to escape random regime, stable enough not to diverge
WEIGHT_DECAY = 1e-5            # reduced from 1e-4: symptom is underfit, not overfit
DEVICE = "cpu"

SWEEP_RUN_MODE = "resource_efficiency"
SWEEP_K_NEIGHBORS = [5, 8]
SWEEP_EPOCHS = [20, 50]    # updated: 1/3 epochs never converge
SWEEP_IMAGES_PER_NODE = [1, 3, 5]
SWEEP_RESULTS_DIR = RESULTS_DIR / "config_sweep"


@dataclass(frozen=True)
class DataConfig:
    """Configuracion del dataset para mantener imports existentes."""

    dataset_name: str = "CIFAR-10"
    raw_dir: Path = DATA_DIR
    train_subset: int = TRAIN_SUBSET_SIZE
    test_subset: int = TEST_SUBSET_SIZE
    images_per_node: int = IMAGES_PER_NODE
    run_mode: str = RUN_MODE

    full_gnn_train_subset: int | None = FULL_GNN_TRAIN_SUBSET_SIZE
    full_gnn_test_subset: int | None = FULL_GNN_TEST_SUBSET_SIZE
    tiny_gnn_train_subset: int = TINY_GNN_TRAIN_SUBSET_SIZE
    tiny_gnn_test_subset: int = TINY_GNN_TEST_SUBSET_SIZE

    controlled_train_subset: int = CONTROLLED_TRAIN_SUBSET_SIZE
    controlled_test_subset: int = CONTROLLED_TEST_SUBSET_SIZE


@dataclass(frozen=True)
class GraphConfig:
    """Configuracion basica del grafo."""

    k_neighbors: int = K_NEIGHBORS
    feature_dim: int = IMAGE_FEATURE_DIM


@dataclass(frozen=True)
class ModelConfig:
    """Configuracion comun de los modelos GNN."""

    input_dim: int = INPUT_DIM
    num_classes: int = NUM_CLASSES
    full_hidden_dim: int = FULL_GNN_HIDDEN_DIM
    tiny_hidden_dim: int = TINY_GNN_HIDDEN_DIM
    dropout: float = DROPOUT


@dataclass(frozen=True)
class TrainingConfig:
    """Configuracion minima del entrenamiento."""

    epochs: int = EPOCHS
    learning_rate: float = LEARNING_RATE
    seed: int = RANDOM_SEED
    device: str = DEVICE
    sweep_run_mode: str = SWEEP_RUN_MODE
    sweep_k_neighbors: list[int] = field(default_factory=lambda: SWEEP_K_NEIGHBORS)
    sweep_epochs: list[int] = field(default_factory=lambda: SWEEP_EPOCHS)
    sweep_images_per_node: list[int] = field(default_factory=lambda: SWEEP_IMAGES_PER_NODE)


@dataclass(frozen=True)
class AppConfig:
    """Configuracion agrupada para utilidades existentes."""

    data: DataConfig = field(default_factory=DataConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    metrics_path: Path = RESULTS_DIR / "metrics.json"
    plots_dir: Path = PLOTS_DIR
    sweep_results_dir: Path = SWEEP_RESULTS_DIR


CONFIG = AppConfig()
