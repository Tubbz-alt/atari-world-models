import logging
from pathlib import Path

import torch

VERSION = (0, 0, 1)
SUPPORTED_GAMES = (
    "Pong-v0",
    "CarRacing-v0",
)
SAMPLES_PATH = Path("samples")

logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    level=logging.WARNING,
)
