# Communication-Aware Collaborative GNN Inference

This repository studies a narrower and more defensible question than the
original prototype:

> When does collaborative GNN inference improve on local inference enough to
> justify communication between edge partitions?

The code compares:

1. **Local:** an MLP with no communication.
2. **Collaborative:** a compressed GAT whose embeddings cross graph-partition
   boundaries.
3. **Cloud:** a larger centralized GNN receiving full node features.

## Current research status

The complete row-L2 five-seed matrix passes the unchanged scientific gate on
Cora, CiteSeer, and PubMed; Flickr preserves accuracy but reaches only a
3.01-fold float32 wire-byte reduction.

Correctness-critical safeguards include:

- PyG edges now use the correct `source -> target` convention, so test nodes
  receive messages from train nodes.
- Full-graph message passing follows the standard transductive protocol, while
  training, validation selection, and final testing remain label-mask isolated.
- The default is one observation per node; random multi-image grouping is only
  an explicit ablation.
- Embedding dimensions are selected by validation accuracy, never test
  accuracy.
- The primary paper protocol row-normalizes sparse features without
  train-column standardization; the chosen preprocessing is recorded in every
  result and checkpoint key.
- Multi-seed results report standard deviation and Student-t 95% confidence
  intervals after averaging placement seeds within each training seed.
- Cross-partition embeddings are deduplicated by source and destination
  partition, serialized, and counted as exact wire bytes.
- The int8 ablation quantizes each serialized source-to-destination batch with
  its own symmetric scale; local embeddings remain float32, and reported
  accuracy is measured after destination-side dequantization.
- Python, pytest, and CI versions are aligned.

The scope is deliberately a single-workstation, partition-aware emulation of
edge collaboration. It does not report measurements from physical edge
devices, production RPC deployments, or hardware energy meters. Experiment
records include the full model/training configuration, checkpoint hash,
clean/dirty Git state, the commit, and a SHA-256 digest of the source tree.

The manuscript pipeline is `scripts/run_paper_experiments.py` and its
centralized reference is the one-layer `FullFeatureGAT` in
`src/systems/distributed_system.py`. The CIFAR-derived scripts are legacy
ablations and use a structurally different three-layer `LegacyCloudGCN`; their
numbers must not be substituted for manuscript results.

The raw numerical records supporting the paper are distributed with the
publication in Supplementary File S1, not through GitHub. Supplementary File
S1 identifies the commit and contains the direct
`scripts/run_paper_experiments.py` JSONL output plus a per-file SHA-256
manifest. Fresh local runs create new working outputs; they are not the frozen
paper evidence.

## Recommended experiments

Start with natural graphs:

```bash
python scripts/run_graph_benchmark.py \
  --dataset cora \
  --epochs 100 \
  --seeds 42 123 2024 2025 2026 \
  --num-partitions 4
```

Supported PyG benchmarks are `cora`, `citeseer`, `pubmed`, `flickr`, and
`reddit`. Dataset downloads occur on first use.

The CIFAR-derived graph remains available as a secondary/artificial ablation:

```bash
python scripts/run_three_system.py \
  --epochs 100 \
  --seeds 42 123 2024 2025 2026 \
  --num-partitions 4
```

Passing `--num-partitions 0` places every graph node on a separate logical
device, so every retained graph edge is counted as a network message.

## Experimental protocol

- PCA is fitted on training data only.
- Official validation masks are preserved; a stratified fallback is used only
  when a dataset has no validation mask.
- Early stopping restores the lowest-validation-loss checkpoint.
- Hyperparameter and embedding selection uses validation metrics.
- Test metrics are evaluated only after selection.
- The primary protocol uses the complete observed graph; `source-only` remains
  an explicit diagnostic.
- Communication uses only cross-partition edges in the selected graph view.

## Installation

Python 3.11-3.14 is supported. With [uv](https://docs.astral.sh/uv/), install
the exact locked environment:

```bash
uv sync --frozen
```

Alternatively, create a virtual environment and install the project directly:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Testing

```bash
uv run python -m pytest -q
```

When using the activated virtual environment instead of uv, run
`python -m pytest -q` directly.

## Reproducing the experiments

First run a one-seed, five-epoch smoke version of the paper pipeline:

```bash
uv run python scripts/run_paper_experiments.py --smoke --include-ablations
```

Run the complete five-seed experiment matrix, including the int8 and PCA
ablations, with:

```bash
uv run python scripts/run_paper_experiments.py --include-ablations
uv run python scripts/generate_paper_artifacts.py
```

The first command downloads benchmark datasets on first use and writes
resumable results under `artifacts/paper/results/`. The second converts those
results into tables and figures under `artifacts/paper/generated/`. These
directories contain locally reproduced outputs; the raw records cited by the
paper remain distributed only in Supplementary File S1.

## Publication target

The active target is the Future Internet special issue **Distributed Machine
Learning and Federated Edge Computing for IoT**, with a hard scientific gate
before submission.

## Post-review revision analyses

The revision preserves the frozen publication evidence and adds exploratory
paired statistics, saved-checkpoint PCA diagnostics, and a balanced METIS
placement comparison for every dataset at 2, 4, and 8 partitions. These analyses
are separate from the original five-seed decision rule.

```bash
uv run python scripts/analyze_revision_statistics.py
uv run --with pymetis==2025.2.2 python scripts/analyze_revision_placement.py
uv run python scripts/generate_revision_tables.py
```

The commands require the frozen evidence under
`artifacts/paper/results/release-v1.0.0-final/`; the statistical audit also
verifies the 40 original focused-ablation checkpoints. These inputs and the
analysis outputs are included in the revision's Supplementary File S2.
PyMETIS is an optional revision-analysis dependency, not part of the main
inference pipeline. Placement variation and training-seed uncertainty are
reported separately. The network replay uses a same-host socket pair (Unix
sockets on the reference Mac), and does not validate a TCP/IP path or physical
edge hardware. Historical raw metadata is preserved with its original wording.
