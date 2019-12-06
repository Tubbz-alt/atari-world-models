import dataclasses
import logging
from abc import ABC, abstractmethod
from pathlib import Path

import torch

from . import DEVICE, MODELS_DIR, OBSERVATIONS_DIR, SAMPLES_DIR

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
    """ Add load_state() and save_state() methods to a class. Expects the
    class to have an attribute *models_dir* of type pathlib.Path and *game* of
    type GymGame.
    """

    def _build_filename(self, stamp):
        if stamp is None:
            filename = "{}.torch".format(self.__class__.__name__.lower())
        else:
            filename = "{}-{}.torch".format(self.__class__.__name__.lower(), stamp)
        return filename

    def load_state(self, stamp=None):
        # If there is a state file - load it
        state_file = self.models_dir / self.game.key / self._build_filename(stamp)
        if state_file.is_file():
            logger.info(
                "%s: Loading state for %s with stamp %s",
                self.__class__.__name__,
                self.game,
                stamp,
            )
            self.load_state_dict(torch.load(str(state_file), map_location=DEVICE))

    def save_state(self, stamp=None):
        logger.info(
            "%s: Saving state for %s with stamp %s",
            self.__class__.__name__,
            self.game,
            stamp,
        )
        state_dir = self.models_dir / self.game.key
        state_dir.mkdir(parents=True, exist_ok=True)
        state_file = state_dir / self._build_filename(stamp)
        torch.save(self.state_dict(), str(state_file))


class Step(ABC):
    """ A step to take to train the NN.

    The step objects are exposed to the commandline via the various subcommands.
    The hyperparams_key attribute is used to access the relevant hyperparameters
    from the HyperParams class.
    """

    hyperparams_key: str

    def __init__(
        self,
        game,
        observations_dir=OBSERVATIONS_DIR,
        models_dir=MODELS_DIR,
        samples_dir=SAMPLES_DIR,
    ):
        self.game = game
        self.observations_dir = observations_dir
        self.models_dir = models_dir
        self.samples_dir = samples_dir

        # Load associated hyperparams if possible
        if self.hyperparams_key is not None:
            self.hyperparams = getattr(game.hyperparams, self.hyperparams_key)
        else:
            self.hyperparams = None

        super().__init__()

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


def merge_args_with_hyperparams(args, hyperparams):
    result = dataclasses.asdict(hyperparams)

    for k, v in args.items():
        if v is not None:
            result[k] = v

    return result
