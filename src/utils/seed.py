"""Control de semillas para experimentos reproducibles."""

import os
import random


def set_seed(seed: int) -> None:
    """Fija semillas en librerias disponibles."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
