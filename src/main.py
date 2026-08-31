"""Pipeline experimental: baselines, GNNs, metricas y graficos."""

import statistics
from typing import Any

import torch
from torch import nn
from torch_geometric.data import Data

from config import (
    CONNECTIVITY_MODE,
    CONTROLLED_TEST_SUBSET_SIZE,
    CONTROLLED_TRAIN_SUBSET_SIZE,
    DEVICE,
    EPOCHS,
    IMAGES_PER_NODE,
    K_NEIGHBORS,
    KNN_METRIC,
    LEARNING_RATE,
    WEIGHT_DECAY,
    PCA_COMPONENTS,
    PLOTS_DIR,
    RANDOM_SEED,
    RESULTS_DIR,
    RUN_MODE,
    USE_PCA,
)
from data.cifar_loader import CifarData, CifarSplit, load_cifar10
from data.feature_extractor import build_node_features_from_images
from data.pca_compressor import PCACompressor
from data.graph_builder import build_similarity_graph
from models.baselines import RandomBaseline, SimpleMLPBaseline
from models.knn_baseline import KNNClassifierBaseline
from models.logreg_baseline import LogRegClassifierBaseline
from models.legacy_cloud_gcn import LegacyCloudGCN
from models.tiny_gnn import TinyGNN
from training.evaluator import evaluate_accuracy, measure_inference_time
from training.trainer import make_optimizer, train_one_epoch, train_with_logging
from utils.feature_normalization import normalize_graph_features
from utils.metrics import (
    build_metrics_dataframe,
    count_trainable_parameters,
    save_metrics,
    save_metrics_dataframe,
)
from utils.plots import save_metrics_bar_charts
from utils.seed import set_seed

# Semillas para evaluaciones multi-seed.
_EVAL_SEEDS = [42, 123, 2024]


def run(
    run_mode: str = RUN_MODE,
    k_neighbors: int = K_NEIGHBORS,
    connectivity_mode: str = CONNECTIVITY_MODE,
    epochs: int = EPOCHS,
    images_per_node: int = IMAGES_PER_NODE,
    results_dir: Any = RESULTS_DIR,
    plots_dir: Any = PLOTS_DIR,
    run_label: str | None = None,
    multi_seed: bool = False,
) -> dict[str, dict[str, Any]]:
    """Ejecuta todos los modelos definidos para el modo actual.

    Comparacion justa entre modelos:
      En 'resource_efficiency' y 'controlled_subset', TODOS los modelos
      (incluido LegacyCloudGCN) usan EL MISMO subconjunto de datos. Esto evita
      confundir capacidad del modelo con tamano del dataset.
      Para ver el techo de LegacyCloudGCN sobre CIFAR-10 completo, usa
      run_mode='full_gnn_only'.

    Protocolo inductivo:
      El KNN se ajusta SOLO sobre nodos de train. Los nodos de test se
      conectan a vecinos de train sin alterar las aristas train-train.
      Esto refleja el escenario real de borde donde los nodos de
      inferencia llegan despues del entrenamiento.

    Evaluacion robusta:
      - Split de validacion (20% de train) para detectar sobre-ajuste.
      - Si multi_seed=True, se repiten los experimentos con varias semillas
        y se reportan media +/- std de test_accuracy.
    """

    seeds = _EVAL_SEEDS if multi_seed else [RANDOM_SEED]
    all_seed_results: dict[str, list[dict[str, Any]]] = {}

    for seed in seeds:
        set_seed(seed)
        device = torch.device(DEVICE)

        experiments = _build_experiments(run_mode)
        graph_cache: dict[tuple[int | None, int | None], tuple[Data, CifarData, PCACompressor | None]] = {}
        seed_results: dict[str, dict[str, Any]] = {}

        for model_name, experiment in experiments.items():
            cache_key = (experiment["train_subset_size"], experiment["test_subset_size"])

            if cache_key not in graph_cache:
                graph_cache[cache_key] = _load_graph_for_experiment(
                    train_subset_size=experiment["train_subset_size"],
                    test_subset_size=experiment["test_subset_size"],
                    device=device,
                    k_neighbors=k_neighbors,
                    connectivity_mode=connectivity_mode,
                    images_per_node=images_per_node,
                    seed=seed,
                )

            graph, cifar_data, compressor = graph_cache[cache_key]

            # Instantiate model NOW — feat_dim is only known after the graph
            # is built (PCA can clamp n_components, images_per_node changes
            # node count, etc.). Using a factory avoids the shape mismatch
            # that occurs when INPUT_DIM in config differs from the real dim.
            feat_dim = int(graph.num_node_features)
            model = experiment["model_factory"](feat_dim)

            _print_graph_summary(
                graph=graph,
                cifar_data=cifar_data,
                model_name=model_name,
                dataset_mode=experiment["dataset_mode"],
                run_mode=run_mode,
                k_neighbors=k_neighbors,
                connectivity_mode=connectivity_mode,
                images_per_node=images_per_node,
                compressor=compressor,
            )

            seed_results[model_name] = _run_model(
                model=model,
                graph=graph,
                device=device,
                model_type=experiment["model_type"],
                dataset_mode=experiment["dataset_mode"],
                train_subset_size=experiment["train_subset_size"],
                test_subset_size=experiment["test_subset_size"],
                run_mode=run_mode,
                k_neighbors=k_neighbors,
                connectivity_mode=connectivity_mode,
                epochs=epochs,
                images_per_node=images_per_node,
                run_label=run_label,
                seed=seed,
                compressor=compressor,
            )

        for model_name, result in seed_results.items():
            all_seed_results.setdefault(model_name, []).append(result)

    results = _aggregate_seed_results(all_seed_results)

    metrics_dataframe = build_metrics_dataframe(results)
    metrics_path = results_dir / "metrics.json"
    metrics_csv_path = results_dir / "metrics.csv"
    save_metrics(metrics_path, results)
    save_metrics_dataframe(metrics_csv_path, metrics_dataframe)
    save_metrics_bar_charts(metrics_dataframe, plots_dir)

    _print_results_summary(results)
    print(f"Metricas guardadas en: {metrics_path}")
    print(f"DataFrame guardado en: {metrics_csv_path}")
    print(f"Graficos guardados en: {plots_dir}")

    return results


def _build_experiments(run_mode: str) -> dict[str, dict[str, Any]]:
    """Define que modelos y datos usar segun RUN_MODE.

    Los modelos se describen como FACTORIES (callables) en lugar de instancias.
    Esto es necesario porque el input_dim correcto solo se conoce despues de
    construir el grafo (depende de PCA_COMPONENTS, del numero de nodos de train,
    y de si USE_PCA esta activado). Instanciar los modelos antes del grafo
    produce un shape mismatch en el forward pass cuando la dimension real
    difiere de INPUT_DIM en config.

    La factory se llama en el bucle de experimentos con el feat_dim real:
        model = experiment["model_factory"](feat_dim)
    """

    valid_modes = {"resource_efficiency", "controlled_subset", "full_gnn_only"}
    if run_mode not in valid_modes:
        raise ValueError(f"RUN_MODE debe ser uno de: {valid_modes}.")

    if run_mode == "full_gnn_only":
        return {
            "full_gnn": _experiment(
                lambda d: LegacyCloudGCN(input_dim=d),
                "full_gnn", "full_cifar10", None, None,
            ),
        }

    common_train: int = CONTROLLED_TRAIN_SUBSET_SIZE
    common_test: int = CONTROLLED_TEST_SUBSET_SIZE

    return {
        "knn_baseline": _experiment(
            lambda d: KNNClassifierBaseline(k=K_NEIGHBORS, metric=KNN_METRIC),
            "knn_sklearn", "common_subset", common_train, common_test,
        ),
        "logreg_baseline": _experiment(
            lambda d: LogRegClassifierBaseline(max_iter=1000),
            "logreg_sklearn", "common_subset", common_train, common_test,
        ),
        "random_baseline": _experiment(
            lambda d: RandomBaseline(),
            "random", "common_subset", common_train, common_test,
        ),
        "mlp_baseline": _experiment(
            lambda d: SimpleMLPBaseline(input_dim=d),
            "no_gnn", "common_subset", common_train, common_test,
        ),
        "tiny_gnn": _experiment(
            lambda d: TinyGNN(input_dim=d),
            "tiny_gnn", "common_subset", common_train, common_test,
        ),
        "full_gnn": _experiment(
            lambda d: LegacyCloudGCN(input_dim=d),
            "full_gnn", "common_subset", common_train, common_test,
        ),
    }


def _experiment(
    model_factory: Any,
    model_type: str,
    dataset_mode: str,
    train_subset_size: int | None,
    test_subset_size: int | None,
) -> dict[str, Any]:
    """Descriptor de experimento. model_factory(feat_dim) -> nn.Module."""
    return {
        "model_factory": model_factory,
        "model_type": model_type,
        "dataset_mode": dataset_mode,
        "train_subset_size": train_subset_size,
        "test_subset_size": test_subset_size,
    }


def _load_graph_for_experiment(
    train_subset_size: int | None,
    test_subset_size: int | None,
    device: torch.device,
    k_neighbors: int,
    connectivity_mode: str,
    images_per_node: int,
    seed: int,
) -> tuple[Data, CifarData]:
    """Carga CIFAR-10 y construye el grafo con permutacion controlada por semilla."""

    cifar_data = load_cifar10(
        train_subset_size=train_subset_size,
        test_subset_size=test_subset_size,
    )

    # Permutacion controlada por semilla para que la asignacion de imagenes
    # a nodos sea aleatoria pero reproducible. Se crean nuevos CifarSplit
    # (frozen dataclass) con los tensores reordenados.
    gen = torch.Generator().manual_seed(seed)
    train_perm = torch.randperm(cifar_data.train.images.shape[0], generator=gen)
    test_perm = torch.randperm(cifar_data.test.images.shape[0], generator=gen)

    shuffled_cifar = CifarData(
        train=CifarSplit(
            images=cifar_data.train.images[train_perm],
            labels=cifar_data.train.labels[train_perm],
        ),
        test=CifarSplit(
            images=cifar_data.test.images[test_perm],
            labels=cifar_data.test.labels[test_perm],
        ),
        class_names=cifar_data.class_names,
    )

    # Generador separado para el shuffle interno de build_node_features_from_images,
    # derivado de la misma semilla pero independiente del randperm anterior.
    node_gen = torch.Generator().manual_seed(seed + 1000)

    graph, compressor = _build_graph(
        cifar_data=shuffled_cifar,
        k_neighbors=k_neighbors,
        connectivity_mode=connectivity_mode,
        images_per_node=images_per_node,
        generator=node_gen,
    )
    # PATCH 9: verify alignment before any training
    from utils.alignment_check import verify_graph_alignment
    check = verify_graph_alignment(
        graph, label=f"subset={train_subset_size}|seed={seed}"
    )
    if not check['passed']:
        raise RuntimeError(
            f"Graph alignment check failed: {check['issues']}"
        )
    graph = graph.to(device)
    return graph, shuffled_cifar, compressor


def _build_graph(
    cifar_data: CifarData,
    k_neighbors: int,
    images_per_node: int,
    connectivity_mode: str = CONNECTIVITY_MODE,
    generator: torch.Generator | None = None,
) -> tuple[Data, PCACompressor | None]:
    """Extrae features agrupadas, comprime con PCA y construye el grafo.

    Orden de operaciones (sin leakage):
      1. build_node_features_from_images sobre train  -> train_features (3072-D)
      2. build_node_features_from_images sobre test   -> test_features  (3072-D)
      3. PCACompressor.fit_transform(train_features)  -> train_features (64-D)
         PCACompressor.transform(test_features)       -> test_features  (64-D)
         El PCA se ajusta SOLO sobre train; test solo se transforma.
      4. build_similarity_graph con features comprimidas.

    El compressor se devuelve para que el caller pueda:
      - logear la varianza explicada,
      - serializar el PCA junto al modelo para inferencia en produccion.
    """

    train_features, train_labels, train_node_image_counts = build_node_features_from_images(
        images=cifar_data.train.images,
        labels=cifar_data.train.labels,
        images_per_node=images_per_node,
        shuffle=True,
        generator=generator,
    )
    test_features, test_labels, test_node_image_counts = build_node_features_from_images(
        images=cifar_data.test.images,
        labels=cifar_data.test.labels,
        images_per_node=images_per_node,
        shuffle=True,
        generator=generator,
    )

    compressor: PCACompressor | None = None
    if USE_PCA:
        compressor = PCACompressor(n_components=PCA_COMPONENTS)
        train_features = compressor.fit_transform(train_features)
        test_features = compressor.transform(test_features)
        print(
            f"[PCA] {train_features.shape[1]}-D features "
            f"(varianza explicada: {compressor.total_explained_variance:.1%})"
        )

    graph = build_similarity_graph(
        train_features=train_features,
        train_labels=train_labels,
        test_features=test_features,
        test_labels=test_labels,
        train_node_image_counts=train_node_image_counts,
        test_node_image_counts=test_node_image_counts,
        k_neighbors=k_neighbors,
        connectivity_mode=connectivity_mode,
    )
    return graph, compressor


def _run_model(
    model: nn.Module,
    graph: Data,
    device: torch.device,
    model_type: str,
    dataset_mode: str,
    train_subset_size: int | None,
    test_subset_size: int | None,
    run_mode: str,
    k_neighbors: int,
    epochs: int,
    images_per_node: int,
    run_label: str | None,
    seed: int,
    connectivity_mode: str = CONNECTIVITY_MODE,
    compressor: PCACompressor | None = None,
) -> dict[str, Any]:
    """Entrena y evalua un modelo.

    Split de validacion: 20% de los nodos de train reservados para
    medir sobre-ajuste sin tocar el split de test.
    """

    # Fast-path for sklearn-style models (e.g. KNNClassifierBaseline).
    # These have no gradient computation, no device movement, and no epoch loop.
    # They fit directly on node features extracted from the graph.
    from models.knn_baseline import KNNClassifierBaseline
    from models.logreg_baseline import LogRegClassifierBaseline
    if isinstance(model, (KNNClassifierBaseline, LogRegClassifierBaseline)):
        return _run_sklearn_model(
            model=model,
            graph=graph,
            model_type=model_type,
            dataset_mode=dataset_mode,
            train_subset_size=train_subset_size,
            test_subset_size=test_subset_size,
            run_mode=run_mode,
            k_neighbors=k_neighbors,
            connectivity_mode=connectivity_mode,
            epochs=epochs,
            images_per_node=images_per_node,
            run_label=run_label,
            seed=seed,
            compressor=compressor,
        )

    model = model.to(device)
    optimizer = make_optimizer(model, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()

    from training.splits import make_inductive_view, stratified_train_val_masks
    actual_train_mask, val_mask = stratified_train_val_masks(
        graph.y, graph.train_mask, seed=seed
    )

    # Patch 2: column-normalize graph features using train statistics.
    # Zero-mean unit-std per feature, then L2-normalize rows.
    # Applied here (not in graph_builder) so test nodes use train stats.
    graph = normalize_graph_features(
        graph, train_mask=actual_train_mask, l2_normalize_rows=True
    ).to(device)
    graph = make_inductive_view(graph, actual_train_mask).to(device)

    # Entrena usando solo los nodos de train efectivos (sin val).
    original_train_mask = graph.train_mask.clone()
    graph.train_mask = actual_train_mask

    loss_curve = train_with_logging(
        model=model,
        data=graph,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        epochs=epochs,
        label=f"{model_type}/{seed}",
        val_mask=val_mask,
    )
    train_loss = loss_curve[-1] if loss_curve else float('nan')

    graph.train_mask = original_train_mask

    train_accuracy = evaluate_accuracy(model, graph, actual_train_mask)
    val_accuracy = evaluate_accuracy(model, graph, val_mask)
    test_accuracy = evaluate_accuracy(model, graph, graph.test_mask)

    return {
        "run_mode": run_mode,
        "run_label": run_label,
        "seed": seed,
        "model_type": model_type,
        "dataset_mode": dataset_mode,
        "k_neighbors": k_neighbors,
        "connectivity_mode": connectivity_mode,
        "epochs": epochs,
        "train_subset_size": train_subset_size
        or int(graph.node_image_counts[graph.train_mask].sum().item()),
        "test_subset_size": test_subset_size
        or int(graph.node_image_counts[graph.test_mask].sum().item()),
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
        "test_accuracy": test_accuracy,
        "accuracy_gap": train_accuracy - test_accuracy,
        "val_test_gap": val_accuracy - test_accuracy,
        "inference_time_median_s": measure_inference_time(model, graph),
        "loss_curve": loss_curve,
        "num_parameters": count_trainable_parameters(model),
        "num_nodes": int(graph.num_nodes),
        "num_edges": int(graph.num_edges),
        "edge_stats": graph.edge_stats if hasattr(graph, "edge_stats") else {},
        "feature_dim": int(graph.num_node_features),
        "train_nodes": int(actual_train_mask.sum().item()),
        "val_nodes": int(val_mask.sum().item()),
        "test_nodes": int(graph.test_mask.sum().item()),
        "total_images": int(graph.node_image_counts.sum().item()),
        "images_per_node": images_per_node,
        "mean_images_per_node": float(graph.node_image_counts.float().mean().item()),
        "pca_enabled": compressor is not None,
        "pca_components": compressor.output_dim if compressor is not None else None,
        "pca_explained_variance": compressor.total_explained_variance if compressor is not None else None,
        "raw_feature_dim": 3 * 32 * 32,
    }


def _run_sklearn_model(
    model: Any,
    graph: Any,
    model_type: str,
    dataset_mode: str,
    train_subset_size: int | None,
    test_subset_size: int | None,
    run_mode: str,
    k_neighbors: int,
    connectivity_mode: str,
    epochs: int,
    images_per_node: int,
    run_label: str | None,
    seed: int,
    compressor: Any,
) -> dict[str, Any]:
    """Evalua un modelo sklearn (KNN, etc.) directamente sobre features de nodo.

    No necesita device, epochs, optimizador ni loop de entrenamiento.
    Fit sobre nodos de train, score sobre nodos de train y test.
    El val split se omite porque sklearn no tiene epoch loop que regularizar.
    """
    from utils.metrics import count_trainable_parameters
    from training.evaluator import measure_inference_time

    x = graph.x.cpu()
    y = graph.y.cpu()
    train_mask = graph.train_mask.cpu()
    test_mask = graph.test_mask.cpu()

    x_train = x[train_mask]
    y_train = y[train_mask]
    x_test  = x[test_mask]
    y_test  = y[test_mask]

    model.fit(x_train, y_train)
    train_accuracy = model.score(x_train, y_train)
    test_accuracy  = model.score(x_test,  y_test)

    # Latencia: tiempo de prediccion sobre todo el grafo (inferencia en el borde)
    import time, statistics as _stats
    warmup, runs = 3, 10
    for _ in range(warmup):
        model.score(x_test, y_test)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model.score(x_test, y_test)
        times.append(time.perf_counter() - t0)
    latency = _stats.median(times)

    return {
        "run_mode": run_mode,
        "run_label": run_label,
        "seed": seed,
        "model_type": model_type,
        "dataset_mode": dataset_mode,
        "k_neighbors": k_neighbors,
        "connectivity_mode": connectivity_mode,
        "epochs": 0,
        "train_subset_size": train_subset_size or int(x_train.shape[0]),
        "test_subset_size":  test_subset_size  or int(x_test.shape[0]),
        "train_loss": float("nan"),
        "train_accuracy": train_accuracy,
        "val_accuracy": float("nan"),
        "test_accuracy": test_accuracy,
        "accuracy_gap": train_accuracy - test_accuracy,
        "val_test_gap": float("nan"),
        "inference_time_median_s": latency,
        "num_parameters": 0,
        "num_nodes": int(graph.num_nodes),
        "num_edges": int(graph.num_edges),
        "edge_stats": getattr(graph, "edge_stats", {}),
        "feature_dim": int(graph.num_node_features),
        "train_nodes": int(x_train.shape[0]),
        "val_nodes": 0,
        "test_nodes": int(x_test.shape[0]),
        "total_images": int(graph.node_image_counts.sum().item()),
        "images_per_node": images_per_node,
        "mean_images_per_node": float(graph.node_image_counts.float().mean().item()),
        "pca_enabled": compressor is not None,
        "pca_components": compressor.output_dim if compressor is not None else None,
        "pca_explained_variance": compressor.total_explained_variance if compressor is not None else None,
        "raw_feature_dim": 3 * 32 * 32,
    }


def _aggregate_seed_results(
    all_seed_results: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """Agrega resultados de multiples semillas (media +/- std)."""

    aggregated: dict[str, dict[str, Any]] = {}
    for model_name, seed_list in all_seed_results.items():
        if len(seed_list) == 1:
            aggregated[model_name] = seed_list[0]
        else:
            base = dict(seed_list[0])
            numeric_keys = [
                "train_accuracy", "val_accuracy", "test_accuracy",
                "accuracy_gap", "val_test_gap", "train_loss",
                "inference_time_median_s",
            ]
            for key in numeric_keys:
                values = [r[key] for r in seed_list if key in r]
                if values:
                    base[f"{key}_mean"] = statistics.mean(values)
                    base[f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
                    base[key] = base[f"{key}_mean"]
            base["num_seeds"] = len(seed_list)
            aggregated[model_name] = base

    return aggregated


def _print_graph_summary(
    graph: Data,
    cifar_data: CifarData,
    model_name: str,
    dataset_mode: str,
    run_mode: str,
    k_neighbors: int,
    connectivity_mode: str,
    images_per_node: int,
    compressor: PCACompressor | None = None,
) -> None:
    y = graph.y.detach().cpu()
    present_class_ids = torch.unique(y).tolist()
    present_classes = [
        f"{class_id}: {cifar_data.class_names[class_id]}" for class_id in present_class_ids
    ]
    edge_stats = graph.edge_stats if hasattr(graph, "edge_stats") else {}

    print(f"\nResumen del grafo CIFAR-10 para {model_name}")
    print(f"Modo: {run_mode} | Datos: {dataset_mode} | K={k_neighbors} | Conectividad: {connectivity_mode}")
    print(
        f"Nodos: {graph.num_nodes} | Aristas: {graph.num_edges} | Features: {graph.num_node_features}D"
    )
    if edge_stats:
        print(
            f"  train-train: {edge_stats.get('train_train', '?')} | "
            f"train-test: {edge_stats.get('train_test', '?')} | "
            f"total: {edge_stats.get('total', '?')}"
        )
    print(f"Train: {int(graph.train_mask.sum())} nodos | Test: {int(graph.test_mask.sum())} nodos")
    print(f"Imagenes/nodo configuradas: {images_per_node} | Media real: {graph.node_image_counts.float().mean():.2f}")
    print(f"Clases: {', '.join(present_classes)}")
    print("Protocolo: INDUCTIVO (KNN solo sobre train; test adjuntado sin alterar aristas train-train)")
    if compressor is not None:
        print(
            f"Features: flatten -> PCA {3 * 32 * 32}-D -> {compressor.output_dim}-D "
            f"(varianza explicada: {compressor.total_explained_variance:.1%})"
        )
    else:
        # Check whether an encoder was used (set by caller via graph attribute)
        encoder_name = getattr(graph, "encoder_name", None)
        if encoder_name:
            print(f"Features: pretrained encoder ({encoder_name}) -> "
                  f"{graph.num_node_features}-D embeddings (offline preprocessing)")
        else:
            print(f"Features: flatten (raw pixels, {3 * 32 * 32}-D, PCA disabled)")


def _print_results_summary(results: dict[str, dict[str, Any]]) -> None:
    print("\n" + "=" * 60)
    print("COMPARACION DE MODELOS (mismo subconjunto de datos)")
    print("=" * 60)
    for model_name, m in results.items():
        print(f"\n{model_name} [{m['model_type']}] | dataset={m['dataset_mode']}")
        print(f"  Params entrenables : {m['num_parameters']}")
        print(f"  Train loss         : {m['train_loss']:.4f}")
        print(f"  Train acc          : {m['train_accuracy']:.4f}")
        if "val_accuracy" in m:
            std = m.get("val_accuracy_std", "")
            std_str = f" ± {std:.4f}" if std != "" else ""
            print(f"  Val acc            : {m['val_accuracy']:.4f}{std_str}")
        std = m.get("test_accuracy_std", "")
        std_str = f" ± {std:.4f}" if std != "" else ""
        print(f"  Test acc           : {m['test_accuracy']:.4f}{std_str}")
        print(f"  Val-Test gap       : {m.get('val_test_gap', float('nan')):.4f}")
        print(f"  Inference (median) : {m.get('inference_time_median_s', float('nan')):.6f} s")
        if "num_seeds" in m:
            print(f"  Semillas           : {m['num_seeds']}")


if __name__ == "__main__":
    run()
