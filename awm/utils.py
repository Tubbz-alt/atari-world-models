from pathlib import Path

import torch

from . import logger


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
    def _build_filename(self, game, stamp):
        if stamp is None:
            filename = "{}-{}.torch".format(game, self.__class__.__name__.lower())
        else:
            filename = "{}-{}-{}.torch".format(
                game, self.__class__.__name__.lower(), stamp
            )
        return filename

    def load_state(self, game, stamp=None):
        # If there is a state file - load it
        device = "cpu"
        models_dir = Path("models")
        state_file = models_dir / Path(self._build_filename(game, stamp))
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
        models_dir = Path("models")
        models_dir.mkdir(parents=True, exist_ok=True)
        state_file = models_dir / Path(self._build_filename(game, stamp))
        torch.save(self.state_dict(), str(state_file))
