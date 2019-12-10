import logging
import multiprocessing
from pathlib import Path

import torch

VERSION = (0, 0, 1)
SAMPLES_DIR: Path = Path("samples")
OBSERVATIONS_DIR: Path = Path("observations")
MODELS_DIR: Path = Path("models")
CPUS_TO_USE: int = multiprocessing.cpu_count()
SHOW_SCREEN: bool = False
CREATE_PROGRESS_SAMPLES: bool = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s: %(message)s", level=logging.WARNING,
)
