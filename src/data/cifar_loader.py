"""Carga de CIFAR-10 y seleccion de subconjuntos."""

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms

from config import DATA_DIR, RANDOM_SEED, TEST_SUBSET_SIZE, TRAIN_SUBSET_SIZE


@dataclass(frozen=True)
class CifarSplit:
    """Imagenes y etiquetas de un split de CIFAR-10."""

    images: torch.Tensor
    labels: torch.Tensor


@dataclass(frozen=True)
class CifarData:
    """Datos de train/test preparados para el pipeline."""

    train: CifarSplit
    test: CifarSplit
    class_names: tuple[str, ...]


def get_cifar10_class_names() -> tuple[str, ...]:
    """Devuelve los nombres oficiales de clases de CIFAR-10."""

    return (
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )


def load_cifar10(
    data_dir: Path = DATA_DIR,
    train_subset_size: int | None = TRAIN_SUBSET_SIZE,
    test_subset_size: int | None = TEST_SUBSET_SIZE,
    seed: int = RANDOM_SEED,
    download: bool = True,
) -> CifarData:
    """Descarga/carga CIFAR-10 y devuelve subconjuntos como tensores."""

    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=download,
        transform=transform,
    )

    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=download,
        transform=transform,
    )

    train_indices = _select_subset_indices(
        dataset_size=len(train_dataset),
        subset_size=train_subset_size,
        seed=seed,
    )

    test_indices = _select_subset_indices(
        dataset_size=len(test_dataset),
        subset_size=test_subset_size,
        seed=seed + 1,
    )

    return CifarData(
        train=_subset_to_tensors(train_dataset, train_indices),
        test=_subset_to_tensors(test_dataset, test_indices),
        class_names=tuple(train_dataset.classes),
    )


def _select_subset_indices(
    dataset_size: int,
    subset_size: int | None,
    seed: int,
) -> list[int]:
    """Selecciona indices aleatorios reproducibles para un subconjunto."""

    if subset_size is None:
        return list(range(dataset_size))

    if subset_size <= 0:
        raise ValueError("subset_size debe ser mayor que 0.")

    if subset_size > dataset_size:
        raise ValueError(
            f"subset_size={subset_size} supera el tamano del dataset ({dataset_size})."
        )

    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(dataset_size, generator=generator)[:subset_size].tolist()


def _subset_to_tensors(dataset: datasets.CIFAR10, indices: list[int]) -> CifarSplit:
    """Convierte un subconjunto de CIFAR-10 a tensores."""

    subset = Subset(dataset, indices)
    images: list[torch.Tensor] = []
    labels: list[int] = []

    for image, label in subset:
        images.append(image)
        labels.append(label)

    return CifarSplit(
        images=torch.stack(images),
        labels=torch.tensor(labels, dtype=torch.long),
    )
