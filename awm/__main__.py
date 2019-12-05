import argparse
import logging
from argparse import RawDescriptionHelpFormatter
from pathlib import Path

from . import (
    MODELS_DIR,
    OBSERVATIONS_DIR,
    SAMPLES_DIR,
    VERSION,
    controller,
    mdn_rnn,
    observations,
    vae,
)
from .controller import TrainController
from .games import SUPPORTED_GAMES
from .mdn_rnn import TrainMDNRNN
from .observations import GatherObservationsPooled
from .play import PlayGame
from .utils import merge_args_with_hyperparams
from .vae import PrecomputeZValues, TrainVAE


def parse():
    description = """
awm - {}

Atari-World-Models - system to train a world-model style NN to play
Atari games.
    """.format(
        ".".join(map(str, VERSION))
    )
    parser = argparse.ArgumentParser(
        "awm", description=description, formatter_class=RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-v", help="Be more verbose", dest="verbose", action="store_true", default=False
    )
    parser.add_argument(
        "--observations-dir",
        type=Path,
        default=OBSERVATIONS_DIR,
        help="Directory for storing/loading the observations (default: %(default)s)",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=MODELS_DIR,
        help="Directory for storing/loading the models (default: %(default)s)",
    )
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=SAMPLES_DIR,
        help="Directory for storing the progress samples (default: %(default)s)",
    )
    parser.add_argument("game", help="Name of game", choices=SUPPORTED_GAMES.keys())

    subparsers = parser.add_subparsers(
        dest="subcommand", title="Subcommands", description="", help="",
    )
    subparsers.required = True

    observations_parser = subparsers.add_parser(
        "gather-observations",
        help="Gather random observations",
        description="Gather random observations",
    )
    observations_parser.add_argument(
        "--show-screen",
        action="store_true",
        default=observations.SHOW_SCREEN,
        help="Show the gym screen when playing (default: %(default)s)",
    )
    observations_parser.add_argument(
        "--cpus-to-use",
        type=int,
        default=observations.CPUS_TO_USE,
        help="CPUs to use in gathering observations (default: %(default)s)",
    )
    observations_parser.add_argument(
        "--number-of-plays",
        type=int,
        default=None,
        help="Number of plays to play (default: %(default)s)",
    )
    observations_parser.add_argument(
        "--steps-per-play",
        type=int,
        default=None,
        help="Steps per play to take (default: %(default)s)",
    )
    observations_parser.add_argument(
        "--action-every-steps",
        type=int,
        default=None,
        metavar="N",
        help="Take an action every N steps (default: %(default)s)",
    )
    observations_parser.set_defaults(klass=GatherObservationsPooled)

    train_vae_parser = subparsers.add_parser(
        "train-vae", help="Train the VAE", description="Train the VAE"
    )
    train_vae_parser.add_argument(
        "--number-of-epochs",
        type=int,
        default=None,
        help="Number of epochs to train the VAE (default: %(default)s)",
    )
    train_vae_parser.add_argument(
        "--dont-create-progress-samples",
        action="store_false",
        default=vae.CREATE_PROGRESS_SAMPLES,
        dest="create_progress_samples",
        help="Don't create sample pictures to visualize the learning progress (default: %(default)s)",
    )
    train_vae_parser.set_defaults(klass=TrainVAE)

    precompute_z_values_parser = subparsers.add_parser("precompute-z-values",)
    precompute_z_values_parser.set_defaults(klass=PrecomputeZValues)

    train_mdn_rnn_parser = subparsers.add_parser(
        "train-mdn-rnn", help="Train the MDN-RNN", description="Train the MDN-RNN"
    )
    train_mdn_rnn_parser.add_argument(
        "--dont-create-progress-samples",
        action="store_false",
        default=mdn_rnn.CREATE_PROGRESS_SAMPLES,
        dest="create_progress_samples",
        help="Don't create sample pictures to visualize the learning progress (default: %(default)s)",
    )
    train_mdn_rnn_parser.add_argument(
        "--number-of-epochs",
        type=int,
        default=None,
        help="Number of epochs to train the MDN_RNN (default: %(default)s)",
    )
    train_mdn_rnn_parser.set_defaults(klass=TrainMDNRNN)

    train_controller_parser = subparsers.add_parser(
        "train-controller",
        help="Train the Controller",
        description="Train the Controller",
    )
    train_controller_parser.add_argument(
        "--reward-threshold",
        type=int,
        default=None,
        help="Threshold for the reward to stop training (default: %(default)s)",
    )
    train_controller_parser.add_argument(
        "--step-limit",
        type=int,
        default=None,
        help="Limit for the game steps to play - 0 means no limit (default: %(default)s)",
    )
    train_controller_parser.add_argument(
        "--show-screen",
        action="store_true",
        default=controller.SHOW_SCREEN,
        help="Show the gym screen when training (default: %(default)s)",
    )
    train_controller_parser.set_defaults(klass=TrainController)

    play_game_parser = subparsers.add_parser(
        "play-game",
        help="Play the game using the World Model",
        description="Play the game using the World Model",
    )
    play_game_parser.add_argument("--stamp", type=int, default=None)
    play_game_parser.set_defaults(klass=PlayGame)

    args = vars(parser.parse_args())

    # Remove keys not mapping directly to a kwarg
    klass = args.pop("klass")
    game = SUPPORTED_GAMES[args.pop("game")]
    verbose = args.pop("verbose")
    observations_dir = args.pop("observations_dir")
    models_dir = args.pop("models_dir")
    samples_dir = args.pop("samples_dir")

    if verbose:
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
    del args["subcommand"]

    obj = klass(game, observations_dir, models_dir, samples_dir)

    # Default to preconfigured hyperparams if not overridden on commandline
    hyperparams = getattr(game.hyperparams, klass.hyperparams_key)
    args = merge_args_with_hyperparams(args, hyperparams)

    obj(**args)


parse()
