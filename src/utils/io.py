"""Funciones sencillas de entrada/salida."""

import json
from pathlib import Path
from typing import Any


def ensure_directories(*paths: Path) -> None:
    """Crea directorios si no existen."""

    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def save_json(data: dict[str, Any], path: Path) -> None:
    """Guarda un diccionario como JSON legible."""

    ensure_directories(path.parent)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
        file.write("\n")


def load_json(path: Path) -> dict[str, Any]:
    """Carga un JSON desde disco."""

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)
