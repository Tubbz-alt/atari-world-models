import logging

import torch

VERSION = (0, 0, 1)
SUPPORTED_GAMES = (
    "Pong-v0",
    "CarRacing-v0",
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

logging_handler = logging.StreamHandler()
logger.addHandler(logging_handler)
formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
logging_handler.setFormatter(formatter)

del logging_handler
del formatter

# Make torch behaviour more reproducible
torch.manual_seed(0)
