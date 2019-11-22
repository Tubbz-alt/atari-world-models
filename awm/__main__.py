import argparse
import logging
from argparse import RawDescriptionHelpFormatter
from pathlib import Path

from . import SUPPORTED_GAMES, VERSION, logger, observations, vae
from .mdn_rnn import train_mdn_rnn
from .observations import gather_observations_pooled
from .vae import precompute_z_values, train_vae


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
    parser.add_argument("game", help="Name of game", choices=SUPPORTED_GAMES)
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
        "--observation-directory",
        type=Path,
        default=observations.OBSERVATION_DIRECTORY,
        help="Path where the observations should be saved (default: %(default)s)",
    )
    observations_parser.add_argument(
        "--number-of-plays",
        type=int,
        default=observations.NUMBER_OF_PLAYS,
        help="Number of plays to play (default: %(default)s)",
    )
    observations_parser.add_argument(
        "--steps-per-play",
        type=int,
        default=observations.STEPS_PER_PLAY,
        help="Steps per play to take (default: %(default)s)",
    )
    observations_parser.add_argument(
        "--action-every-steps",
        type=int,
        default=observations.ACTION_EVERY_STEPS,
        metavar="N",
        help="Take an action every N steps (default: %(default)s)",
    )
    observations_parser.set_defaults(func=gather_observations_pooled)

    train_vae_parser = subparsers.add_parser(
        "train-vae", help="Train the VAE", description="Train the VAE"
    )
    train_vae_parser.add_argument(
        "--observation-directory",
        default=observations.OBSERVATION_DIRECTORY,
        help="Path to load the observations from (default: %(default)s)",
    )
    train_vae_parser.add_argument(
        "--number-of-epochs",
        type=int,
        default=vae.NUMBER_OF_EPOCHS,
        help="Number of epochs to train the VAE (default: %(default)s)",
    )
    train_vae_parser.add_argument(
        "--create-progress-samples",
        action="store_false",
        default=vae.CREATE_PROGRESS_SAMPLES,
        help="Create sample pictures to visualize the learning progress (default: %(default)s)",
    )
    train_vae_parser.set_defaults(func=train_vae)

    precompute_z_values_parser = subparsers.add_parser("precompute-z-values",)
    precompute_z_values_parser.set_defaults(func=precompute_z_values)

    train_mdn_rnn_parser = subparsers.add_parser(
        "train-mdn-rnn", help="Train the MDN-RNN", description="Train the MDN-RNN"
    )
    train_mdn_rnn_parser.set_defaults(func=train_mdn_rnn)

    args = vars(parser.parse_args())

    # Remove keys not mapping directly to a kwarg
    func = args.pop("func")
    game = args.pop("game")
    verbose = args.pop("verbose")
    if verbose:
        logger.setLevel(logging.DEBUG)
    del args["subcommand"]

    func(game, **args)


parse()
