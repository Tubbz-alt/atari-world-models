import logging
from pathlib import Path

import torch

from . import MODELS_DIR

logger = logging.getLogger(__name__)


def spread(n, number_of_bins):
    """ Split *n* into *number_of_bins* almost equally sized numbers. Do not
    return bins that contain 0.

    This always holds: sum(spread(n, number_of_bins)) == n
    """
    count, remaining = divmod(n, number_of_bins)
    result = [count] * number_of_bins
    for i in range(remaining):
        result[i] += 1
    return [i for i in result if i != 0]


class StateSavingMixin:
    def _build_filename(self, stamp):
        if stamp is None:
            filename = "{}.torch".format(self.__class__.__name__.lower())
        else:
            filename = "{}-{}.torch".format(
                self.__class__.__name__.lower(), stamp
            )
        return filename

    def load_state(self, game, stamp=None):
        # If there is a state file - load it
        device = "cpu"
        state_file = MODELS_DIR / Path(game) / self._build_filename(stamp)
        if state_file.is_file():
            logger.info(
                "%s: Loading state for %s with stamp %s",
                self.__class__.__name__,
                game,
                stamp,
            )
            self.load_state_dict(torch.load(str(state_file), map_location=device))

    def save_state(self, game, stamp=None):
        logger.info(
            "%s: Saving state for %s with stamp %s", self.__class__.__name__, game, stamp
        )
        state_dir = MODELS_DIR / Path(game)
        state_dir.mkdir(parents=True, exist_ok=True)
        state_file = state_dir / self._build_filename(stamp)
        torch.save(self.state_dict(), str(state_file))
