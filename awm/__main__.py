import argparse
import logging
from argparse import RawDescriptionHelpFormatter
from pathlib import Path

from . import SUPPORTED_GAMES, VERSION, controller, mdn_rnn, observations, vae, OBSERVATIONS_DIR
from .controller import train_controller
from .mdn_rnn import train_mdn_rnn
from .observations import gather_observations_pooled
from .play import play_game
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
        "--observations-directory",
        type=Path,
        default=OBSERVATIONS_DIR,
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
        "--observations-directory",
        default=OBSERVATIONS_DIR,
        help="Path to load the observations from (default: %(default)s)",
    )
    train_vae_parser.add_argument(
        "--number-of-epochs",
        type=int,
        default=vae.NUMBER_OF_EPOCHS,
        help="Number of epochs to train the VAE (default: %(default)s)",
    )
    train_vae_parser.add_argument(
        "--dont-create-progress-samples",
        action="store_false",
        default=vae.CREATE_PROGRESS_SAMPLES,
        dest="create_progress_samples",
        help="Don't create sample pictures to visualize the learning progress (default: %(default)s)",
    )
    train_vae_parser.set_defaults(func=train_vae)

    precompute_z_values_parser = subparsers.add_parser("precompute-z-values",)
    precompute_z_values_parser.set_defaults(func=precompute_z_values)

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
        default=mdn_rnn.NUMBER_OF_EPOCHS,
        help="Number of epochs to train the MDN_RNN (default: %(default)s)",
    )
    train_mdn_rnn_parser.set_defaults(func=train_mdn_rnn)

    train_controller_parser = subparsers.add_parser(
        "train-controller",
        help="Train the Controller",
        description="Train the Controller",
    )
    train_controller_parser.add_argument(
        "--reward-threshold",
        type=int,
        default=controller.REWARD_THRESHOLD,
        help="Threshold for the reward to stop training (default: %(default)s)",
    )
    train_controller_parser.add_argument(
        "--show-screen",
        action="store_true",
        default=controller.SHOW_SCREEN,
        help="Show the gym screen when training (default: %(default)s)",
    )
    train_controller_parser.set_defaults(func=train_controller)

    play_game_parser = subparsers.add_parser(
        "play-game",
        help="Play the game using the World Model",
        description="Play the game using the World Model",
    )
    play_game_parser.add_argument("--stamp", type=int, default=None)
    play_game_parser.set_defaults(func=play_game)

    args = vars(parser.parse_args())

    # Remove keys not mapping directly to a kwarg
    func = args.pop("func")
    game = args.pop("game")
    verbose = args.pop("verbose")
    if verbose:
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
    del args["subcommand"]

    func(game, **args)


parse()
