import logging
from pathlib import Path

import torch

VERSION = (0, 0, 1)
SUPPORTED_GAMES = (
    "CarRacing-v0",
)
SAMPLES_DIR : Path = Path("samples")
OBSERVATIONS_DIR : Path = Path("observations")
MODELS_DIR : Path = Path("models")

logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    level=logging.WARNING,
)
