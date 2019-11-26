from pathlib import Path

import torch


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
    def load_state(self, game):
        # If there is a state file - load it
        device = "cpu"
        models_dir = Path("models")
        state_file = models_dir / Path(
            "{}-{}.torch".format(game, self.__class__.__name__.lower())
        )
        if state_file.is_file():
            self.load_state_dict(torch.load(str(state_file), map_location=device))

    def save_state(self, game):
        models_dir = Path("models")
        models_dir.mkdir(parents=True, exist_ok=True)

        state_file = models_dir / Path(
            "{}-{}.torch".format(game, self.__class__.__name__.lower())
        )
        torch.save(self.state_dict(), str(state_file))
