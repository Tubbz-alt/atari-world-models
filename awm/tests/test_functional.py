import os
import tempfile
from pathlib import Path

from xvfbwrapper import Xvfb

from ..controller import TrainController
from ..games import CarRacing
from ..mdn_rnn import TrainMDNRNN
from ..observations import GatherObservationsPooled
from ..play import PlayGame
from ..vae import PrecomputeZValues, TrainVAE


def test_functional_car_racing():
    """ This test performs all the necessary stages for training, but with
    a minimal set of data.

    This is not intended to check if training the NN works, but a basic
    sanity check, that nothing is horribly b0rken.
    """
    game = CarRacing()
    working_dir = Path(tempfile.mkdtemp())
    observations_directory = working_dir / "observations"
    models_directory = working_dir / "models"
    samples_directory = working_dir / "samples"

    def build(klass):
        return klass(game, observations_directory, models_directory, samples_directory)

    # Gather observations - after gathering a number of observation files
    # must exist
    build(GatherObservationsPooled)(
        number_of_plays=1, steps_per_play=100, action_every_steps=10
    )
    dir_contents = os.listdir(observations_directory / game.key)
    # 1 play -> 1 worker
    assert len(dir_contents) == 1
    # 100 steps -> 100 observations
    assert len(os.listdir(observations_directory / game.key / dir_contents[0])) == 100

    # Train the VAE - after training a state file must exist
    assert not (models_directory / game.key / "vae.torch").is_file()
    build(TrainVAE)(number_of_epochs=2)
    assert (models_directory / game.key / "vae.torch").is_file()

    build(PrecomputeZValues)()

    # Train the MDN_RNN - after training a state file must exist
    assert not (models_directory / game.key / "mdn_rnn.torch").is_file()
    build(TrainMDNRNN)(number_of_epochs=2)
    assert (models_directory / game.key / "mdn_rnn.torch").is_file()

    # Train the Controller - after training a state file must exist. We set
    # the threshold really high, s.t. training will finish in epoch 1
    assert not (models_directory / game.key / "controller-1.torch").is_file()
    build(TrainController)(step_limit=5, reward_threshold=2000)
    assert (models_directory / game.key / "controller-1.torch").is_file()

    # Capture X11 in Xvfb
    vdisplay = Xvfb()
    vdisplay.start()
    build(PlayGame)(step_limit=10)
    vdisplay.stop()
